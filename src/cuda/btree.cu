#include "cuda/btree.cuh"

#include "cuda/utils.cuh"
#include "cuda/soa_btree_nodes.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_partition.cuh>

// Computes the longes common prefix between
// the bits of two unsigned integers
__device__ __forceinline__ int lcp_safe(const morton_t *codes, int i, int j)
{
    // Does this allow coalescing?
    return __clzll(codes[i] ^ codes[j]) - 1; // Removing leading 0
}

__device__ __forceinline__ int lcp(const morton_t *codes, int i, int j, int n)
{
    // i index is always in range
    if (j < 0 || j > n - 1)
        return -3;
    else 
        return lcp_safe(codes, i, j);
}

__device__ __forceinline__ int sign(int x)
{
    return (x > 0) - (x < 0);
}

__device__ int find_split(const morton_t *codes,
                          int first,
                          int last,
                          int node_lcp,
                          int dir,
                          int num_leaves)
{
    int step = 0;
    int length = (last - first) * dir;

    do {
        length = (length + 1) >> 1;

        if (lcp(
               codes, first, first + (step + length) * dir, num_leaves) >
               node_lcp) { 
            step += length;
        }
    } while (length > 1);

    return first + step * dir + min(dir, 0);
}

__device__ int search_lr(const morton_t *codes,
                         const int *leaf_depth,
                         int dir,
                         int idx,
                         int depth_delta,
                         int num_leaves)
{
    int l = 0;
    int l_max = 2;

    int target_depth = leaf_depth[idx] - depth_delta;

    // Determine upper bound
    while (lcp(codes, idx, idx + dir * l_max, num_leaves) / 3 >=
           target_depth) {
        l_max = l_max << 1;
    }

    // Binary search
    while (l_max > 0) {
        l_max = l_max >> 1;
        if (lcp(codes, idx, idx + dir * (l + l_max), num_leaves) / 3 >=
            target_depth) {
            l += l_max;
        }
    }

    return l;
}

__device__ void set_node_child(const morton_t *codes,
                               SoABtreeNodes nodes,
                               int *child,
                               int parent_idx,
                               int child_idx,
                               int first_idx,
                               int parent_lcp)
{
    bool is_leaf = child_idx == first_idx;

    child[2 * parent_idx + 1] = 2 * child_idx + !is_leaf;

    // Lcp of leaves covered by child node
    int lcp = lcp_safe(codes, child_idx, first_idx);
    // When delta > 1, the node represent successive
    // descendants of octree nodes each having one child,
    // we compress the octree by ignoring the intermediate nodes
    // with no siblings
    int delta = min(1, lcp / 3 - parent_lcp / 3 + is_leaf);

    nodes.parent(2 * child_idx + !is_leaf) = 2 * parent_idx + 1;
    nodes.edge_delta(2 * child_idx + !is_leaf) = delta;
    nodes.depth(2 * child_idx + !is_leaf) = delta;

    if (is_leaf) {
        nodes.lcp(2 * child_idx) = parent_lcp + 3;
        nodes.left(2 * child_idx) = 0;
        nodes.right(2 * child_idx) = 0;
        nodes.leaves_begin(2 * child_idx) = child_idx;
        nodes.leaves_end(2 * child_idx) = child_idx;
    }

}

__global__ void reduce_leaf_depth(const morton_t *codes,
                                  int *leaf_depth,
                                  const int *num_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_leaves) {
        return;
    }

    leaf_depth[idx] = max(lcp(codes, idx, idx - 1, *num_leaves) / 3,
                          lcp(codes, idx, idx + 1, *num_leaves) / 3) + 1;
}

__global__ void merge_leaves(const morton_t *codes,
                             const int *leaf_depth,
                             const int *leaf_first_in,
                             int *leaf_first_out,
                             int *leaf_flagged,
                             int max_num_codes_per_leaf,
                             const int *num_leaves,
                             int max_num_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > max_num_leaves) {
        return;
    }

    // Making sure unused array rear contains 0, since
    // compaction will be called using max_num_leaves as size
    leaf_flagged[idx] = 0;
    leaf_first_out[idx] = *num_leaves;

    if (idx >= *num_leaves) {
        return;
    }

    leaf_first_out[idx] = idx;

    int left_old = 0;
    int left = 0;
    int right = 0;
    int depth_delta = 0;

    int num_codes = leaf_first_in[idx + 1] - leaf_first_in[idx];

    // Attempts to raise a leaf to a smaller depth
    while (num_codes <= max_num_codes_per_leaf) {
        depth_delta++;
        left_old = left;

        // Searching for first and last codes covered
        // by the candidate merged leaf
        left = search_lr(
            codes, leaf_depth, -1, idx, depth_delta, *num_leaves);
        right = search_lr(
            codes, leaf_depth, 1, idx, depth_delta, *num_leaves);
        // Number of codes contained in the candidate merged leaf
        num_codes = leaf_first_in[idx + right + 1] -
                    leaf_first_in[idx - left];
    }

    // Keeping only leaf entries at the start of each merged leaf
    leaf_flagged[idx] = left_old == 0;
}

// TODO: improve coalescence
__global__ void build_radix_tree(const morton_t *codes,
                                 SoABtreeNodes nodes,
                                 int *tmp_ranges,
                                 const int *num_leaves_,
                                 int max_num_internal,
                                 int max_num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_leaves = *num_leaves_;
    int num_internal = num_leaves - 1;

    if (idx < max_num_internal) {
        // TODO: coalesce it in a separate kernel
        // Filling tmp ranges used to later sort tree nodes
#pragma unroll
        for (int i = 0; i < 3; i++) {
            tmp_ranges[2 * idx + max_num_nodes * i] = 2 * idx;
            tmp_ranges[2 * idx + 1 + max_num_nodes * i] = 2 * idx + 1;
        }
    }

    if (idx >= num_internal && idx < max_num_internal) {
        // WARNING: in order to obtain the correct number
        // of required octree nodes by indexing the scanned
        // edge_delta at position num_leaves-1 we have to be
        // sure that the unused btree nodes, sorted by depth,
        // are placed at the end of the array.
        // This is also necessary if we're rebuilding the tree
        // and the number of unique leaves may have changed
        // from the previous iteration

        nodes.depth(2 * idx + 1) = 2 * num_leaves; // +1 just to be sure
        nodes.depth(2 * idx + 2) = 2 * num_leaves; // +1 just to be sure
    }

    if (idx >= num_internal) {
        return;
    }

    if (idx == 0) {
        nodes.parent(1) = 0;
        nodes.depth(1) = 0;
        // The root node is a valid octree node
        nodes.edge_delta(1) = 1;

#pragma unroll
        for (int i = 0; i < 3; i++) {
            tmp_ranges[max_num_nodes * (i + 1) - 1] = max_num_nodes - 1;
        }
    }

    // Determines whether the left or right
    // leaf is part of the current internal node
    int d = sign(lcp(codes, idx, idx + 1, num_leaves) -
                 lcp(codes, idx, idx - 1, num_leaves));

    // Minimum length of the common prefix between the
    // leaves covered by the current internal node, it
    // is obtained by computing lcp on the non-sibling
    // neighbouring node
    int lcp_min = lcp(codes, idx, idx - d, num_leaves);

    // Computes upper bound for the length of prefix
    // covered by the current internal node by doubling
    // the search range until a leaf whose lcp is <= delta_min
    int max_length = 2;
    while (lcp(codes, idx, idx + max_length * d, num_leaves) > lcp_min) {
        max_length = max_length << 1;
    }

    // Uses iterative binary search for the exact end of the
    // range of leaves covered by the current internal node
    int length = 0;
    int step = max_length;
    do {
        // Half the step size
        step = step >> 1;

        if (lcp(codes, idx, idx + (length + step) * d, num_leaves) >
            lcp_min) {
            length += step;
        }
    } while (step > 1);
    // End of the range of leaves covered
    int last = idx + length * d;

    // Length of prefix covered by the internal node
    int node_lcp = lcp_safe(codes, idx, last);
    nodes.lcp(2 * idx + 1) = node_lcp;

    int split = find_split(codes, idx, last, node_lcp, d, num_leaves);

    int min_leaf = min(idx, last);
    int max_leaf = max(idx, last);
    // Range of leaves covered by the internal node
    nodes.leaves_begin(2 * idx + 1) = min_leaf;
    nodes.leaves_end(2 * idx + 1) = max_leaf;

    set_node_child(codes,
                   nodes,
                   nodes.left(),
                   idx,
                   split,
                   min_leaf,
                   node_lcp);

    set_node_child(codes,
                   nodes,
                   nodes.right(),
                   idx,
                   split + 1,
                   max_leaf,
                   node_lcp);
}

__global__ void compute_nodes_depth(const int *parent_in,
                                    const int *depth_in,
                                    int *parent_out,
                                    int *depth_out,
                                    const int *num_leaves_,
                                    int max_num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_leaves = *num_leaves_;
    int num_nodes = 2 * num_leaves - 1;

    if (idx >= num_nodes && idx < max_num_nodes) {
        depth_out[idx] = depth_in[idx];
    }

    if (idx >= num_nodes) {
        return;
    }

    int curr_parent = parent_in[idx];
    int curr_depth = depth_in[idx];

    bool flag = curr_parent && curr_depth > 0;

    int next_depth = curr_depth + depth_in[curr_parent] * flag;
    int next_parent = flag ? parent_in[curr_parent] : curr_parent;

    parent_out[idx] = next_parent;
    depth_out[idx] = next_depth;
}

// Used to update pointers to children after sorting the nodes
__global__ void correct_child_pointers(int *left,
                                       int *right,
                                       const int *map,
                                       const int *num_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_nodes = 2 * *num_leaves - 1;

    if (idx >= num_nodes) {
        return;
    }

    int left_value = left[idx];
    int right_value = right[idx];

    // TODO: This check could be ignored
    if (left_value != 0 || right_value != 0) {
        left[idx] = map[left_value];
        right[idx] = map[right_value];
    }
}

Btree::Btree(int max_num_leaves) :
    init_max_num_leaves_(max_num_leaves),
    max_num_leaves_(max_num_leaves)
{
    cudaMalloc(&num_leaves_, sizeof(int));
    cudaMalloc(&range_, (max_num_leaves + 1) * sizeof(int));

    // Allocating device memory for internal nodes
    nodes_.alloc(get_max_num_nodes());
    tmp_nodes_.alloc(get_max_num_nodes());

    cudaMalloc(&octree_map_, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&leaf_first_code_idx_, (max_num_leaves + 1) * sizeof(int));

    // Allocating buffers to store intermediate computations
    // TODO: align inner buffers
    cudaMalloc(&tmp_, (max_num_leaves_ + 1) * 3 * sizeof(int));
    cudaMalloc(&tmp_morton_, (max_num_leaves_ + 1) * 3 * sizeof(morton_t));
    cudaMalloc(&tmp_ranges_, get_max_num_nodes() * 3 * sizeof(int));

    // Allocating tmp arrays for cub operations
    tmp_sort_ = nullptr;
    cub::DeviceMergeSort::StableSortPairs(
        tmp_sort_,
        tmp_sort_size_,
        nodes_.depth_,
        tmp_ranges_,
        get_max_num_nodes(),
        sort_op_);
    cudaMalloc(&tmp_sort_, tmp_sort_size_);

    tmp_compact_ = nullptr;
    cub::DevicePartition::Flagged(tmp_compact_,
                                  tmp_compact_size_,
                                  tmp_ranges_,
                                  tmp_,
                                  leaf_first_code_idx_,
                                  num_leaves_,
                                  max_num_leaves + 1);
    cudaMalloc(&tmp_compact_, tmp_compact_size_);

    thrust::sequence(thrust::device,
                     range_,
                     range_ + max_num_leaves + 1);

    int n = 1;
    num_launches_compute_nodes_depth_ = 0;
    while (n < get_max_num_nodes()) {
        n = n << 1;
        num_launches_compute_nodes_depth_++;
    }
    num_launches_compute_nodes_depth_ += 3;
}

void Btree::reset_max_num_leaves()
{
    max_num_leaves_ = init_max_num_leaves_;
}

void Btree::generate_leaves(const morton_t *d_sorted_codes,
                            int max_num_codes_per_leaf)
{
    int *leaf_depth = tmp_;
    // Sets the initial depth (octree relative) of each leaf
    // such that neighbouring leaves are not spatially overlapping
    reduce_leaf_depth<<<max_num_leaves_ / THREADS_PER_BLOCK +
                        (max_num_leaves_ % THREADS_PER_BLOCK > 0),
                        THREADS_PER_BLOCK>>>(d_sorted_codes,
                                             leaf_depth,
                                             num_leaves_);

    int *tmp_range_out = tmp_ + max_num_leaves_ + 1;
    int *leaf_flagged = tmp_ + 2 * (max_num_leaves_ + 1);
    // Merges adjacent leaves until each holds no more than
    // max_num_codes_per_leaf codes
    merge_leaves<<<(max_num_leaves_ + 1) / THREADS_PER_BLOCK +
                   ((max_num_leaves_ + 1) % THREADS_PER_BLOCK > 0),
                   THREADS_PER_BLOCK>>>(d_sorted_codes,
                                        leaf_depth,
                                        range_,
                                        tmp_range_out,
                                        leaf_flagged,
                                        max_num_codes_per_leaf,
                                        num_leaves_,
                                        max_num_leaves_);

    // Keeping only the first code of each merged leaf
    cub::DevicePartition::Flagged(tmp_compact_,
                                  tmp_compact_size_,
                                  tmp_range_out,
                                  leaf_flagged,
                                  leaf_first_code_idx_,
                                  num_leaves_,
                                  max_num_leaves_ + 1);
}

void Btree::build(const morton_t *d_sorted_codes)
{
    // WARNING: watch out when gathering values from
    // the rear of d_leaf_first_code, it should probably be
    // filled with 0
    morton_t *leaf_first_code = tmp_morton_;

    thrust::gather(thrust::device,
                   leaf_first_code_idx_,
                   leaf_first_code_idx_ + max_num_leaves_,
                   d_sorted_codes,
                   leaf_first_code);

    build_radix_tree<<<get_max_num_internal() / THREADS_PER_BLOCK +
                       (get_max_num_internal() % THREADS_PER_BLOCK > 0),
                       THREADS_PER_BLOCK>>>(leaf_first_code,
                                            nodes_,
                                            tmp_ranges_,
                                            num_leaves_,
                                            get_max_num_internal(),
                                            get_max_num_nodes());

    for (int i = 0; i < num_launches_compute_nodes_depth_; ++i) {
        // Maximizing the thread block size here
        compute_nodes_depth<<<
           get_max_num_nodes() / MAX_THREADS_PER_BLOCK +
           (get_max_num_nodes() % MAX_THREADS_PER_BLOCK > 0),
           MAX_THREADS_PER_BLOCK>>>(nodes_.parent_,
                                    nodes_.depth_,
                                    tmp_nodes_.parent_,
                                    tmp_nodes_.depth_,
                                    num_leaves_,
                                    get_max_num_nodes());

        // TODO: consider using std::swap
        swap_ptr(&nodes_.parent_, &tmp_nodes_.parent_);
        swap_ptr(&nodes_.depth_, &tmp_nodes_.depth_);
    }
}

void Btree::sort_to_bfs_order()
{
    // Sort arrays by depth

    int *tmp_range = tmp_ranges_;
    int *tmp_perm_in = tmp_ranges_ + get_max_num_nodes();
    int *tmp_perm_out = tmp_perm_in + get_max_num_nodes();

    // TODO: custom sort?

    cub::DeviceMergeSort::StableSortPairs(
        tmp_sort_,
        tmp_sort_size_,
        nodes_.depth_,
        tmp_perm_in,
        get_max_num_nodes(),
        sort_op_);

    int **in_ptrs[6];
    in_ptrs[0] = &nodes_.left_;
    in_ptrs[1] = &nodes_.right_;
    in_ptrs[2] = &nodes_.lcp_;
    in_ptrs[3] = &nodes_.edge_delta_;
    in_ptrs[4] = &nodes_.leaves_begin_;
    in_ptrs[5] = &nodes_.leaves_end_;

    int **tmp_ptrs[6];
    tmp_ptrs[0] = &tmp_nodes_.left_;
    tmp_ptrs[1] = &tmp_nodes_.right_;
    tmp_ptrs[2] = &tmp_nodes_.lcp_;
    tmp_ptrs[3] = &tmp_nodes_.edge_delta_;
    tmp_ptrs[4] = &tmp_nodes_.leaves_begin_;
    tmp_ptrs[5] = &tmp_nodes_.leaves_end_;

#pragma unroll
    for (int i = 0; i < 6; ++i) {
        // It's faster to gather the arrays to be sorted
        // based on the sorted permutation
        // TODO: gather on multiple streams?
        thrust::gather(thrust::device,
                       tmp_perm_in,
                       tmp_perm_in + get_max_num_nodes(),
                       *in_ptrs[i],
                       *tmp_ptrs[i]);

        swap_ptr(in_ptrs[i], tmp_ptrs[i]);
    }

    // Reverse tmp_perm_in mapping into tmp_perm_out
    // TODO: find a way to perform it using cub
    thrust::scatter(thrust::device,
                    tmp_range,
                    tmp_range + get_max_num_nodes(),
                    tmp_perm_in,
                    tmp_perm_out);

    correct_child_pointers<<<get_max_num_nodes() / THREADS_PER_BLOCK +
                             (get_max_num_nodes() % THREADS_PER_BLOCK > 0),
                             THREADS_PER_BLOCK>>>(nodes_.left_,
                                                  nodes_.right_,
                                                  tmp_perm_out,
                                                  num_leaves_);
}

void Btree::compute_octree_map()
{
    // Given an internal node i, if _edge_delta[i] > 0
    // _octree_map[i] will contain the index of the
    // corresponding octree node

    // TODO: use cub::DeviceScan
    thrust::exclusive_scan(thrust::device,
                           nodes_.edge_delta_,
                           nodes_.edge_delta_ + get_max_num_nodes(),
                           octree_map_);
}

Btree::~Btree()
{
    // Releasing device memory
    cudaFree(num_leaves_);

    nodes_.free();
    tmp_nodes_.free();

    cudaFree(leaf_first_code_idx_);
    cudaFree(octree_map_);

    cudaFree(tmp_);
    cudaFree(tmp_morton_);
    cudaFree(tmp_ranges_);
    cudaFree(range_);

    cudaFree(tmp_sort_);
    cudaFree(tmp_compact_);
}
