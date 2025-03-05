#include "btree_gpu.cuh"

#include "utils_gpu.cuh"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_partition.cuh>

__device__ __forceinline__ uint32_t _expand_bits(uint32_t u)
{
    u = (u * 0x00010001u) & 0xFF0000FFu;
    u = (u * 0x00000101u) & 0x0F00F00Fu;
    u = (u * 0x00000011u) & 0xC30C30C3u;
    u = (u * 0x00000005u) & 0x49249249u;

    return u;
}

// Computes 30-bit morton code by interleaving the bits
// of the coordinates, supposing that they are normalized
// in the range [0.0, 1.0]
__global__ void morton_encode(const Points *points, uint32_t *codes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Scale coordinates to [0, 2^10)
    uint32_t x = (uint32_t) (points->get_x(idx) * 1023.0f);
    uint32_t y = (uint32_t) (points->get_y(idx) * 1023.0f);
    uint32_t z = (uint32_t) (points->get_z(idx) * 1023.0f);

    x = _expand_bits(x);
    y = _expand_bits(y);
    z = _expand_bits(z);

    // Left shift x by 2 bits, y by 1 bit, then bitwise or
    codes[idx] = x * 4 + y * 2 + z;
}

// Computes the longes common prefix between
// the bits of two unsigned integers
__device__ __forceinline__ int _lcp_safe(const uint32_t *codes, int i, int j)
{
    // Does this allow coalescing?
    return __clz(codes[i] ^ codes[j]) - 2; // Removing leading 00
}

__device__ __forceinline__ int _lcp(const uint32_t *codes, int i, int j, int n)
{
    // i index is always in range
    if (j < 0 || j > n - 1)
        return -3;
    else 
        return _lcp_safe(codes, i, j);
}

__device__ __forceinline__ int _sign(int x)
{
    return (x > 0) - (x < 0);
}

__device__ int _find_split(const uint32_t *codes,
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

        if (_lcp(
                codes, first, first + (step + length) * dir, num_leaves) >
                node_lcp) { 
            step += length;
        }
    } while (length > 1);

    return first + step * dir + min(dir, 0);
}

__device__ int _search_lr(const uint32_t *codes,
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
    while (_lcp(codes, idx, idx + dir * l_max, num_leaves) / 3 >=
           target_depth) {
        l_max = l_max << 1;
    }

    // Binary search
    while (l_max > 0) {
        l_max = l_max >> 1;
        if (_lcp(codes, idx, idx + dir * (l + l_max), num_leaves) / 3 >=
            target_depth) {
            l += l_max;
        }
    }

    return l;
}

__device__ void _set_node_child(const uint32_t *codes,
                                struct Btree::Nodes nodes,
                                int *child,
                                int parent_idx,
                                int child_idx,
                                int first_idx,
                                int parent_lcp)
{
    bool is_leaf = child_idx == first_idx;

    // Pointers to leaf nodes are offset
    // by the number of internal nodes
    child[2 * parent_idx + 1] = 2 * child_idx + !is_leaf;
    //child_idx + num_internal * is_leaf;

    // Lcp of leaves covered by child node
    int lcp = _lcp_safe(codes, child_idx, first_idx);
    // When delta > 1, the node represent successive
    // descendants of octree nodes each having one child,
    // we compress the octree by ignoring the intermediate nodes
    // with no siblings
    int delta = min(1, lcp / 3 - parent_lcp / 3 + is_leaf);

    /*
    if (!is_leaf) {
        parent[child_idx] = parent_idx;
        edge_delta[child_idx] = delta;
        // Useful to compute octree nodes depth
        depth[child_idx] = delta;
    }
    */

    nodes.parent[2 * child_idx + !is_leaf] = 2 * parent_idx + 1;
    nodes.edge_delta[2 * child_idx + !is_leaf] = delta;
    nodes.depth[2 * child_idx + !is_leaf] = delta;

    if (is_leaf) {
        nodes.left[2 * child_idx] = 0;
        nodes.right[2 * child_idx] = 0;
        nodes.leaves_begin[2 * child_idx] = child_idx;
        nodes.leaves_end[2 * child_idx] = child_idx;
    }

}

__global__ void _reduce_leaf_depth(const uint32_t *codes,
                                   int *leaf_depth,
                                   const int *num_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_leaves) {
        return;
    }

    leaf_depth[idx] = max(_lcp(codes, idx, idx - 1, *num_leaves) / 3,
                          _lcp(codes, idx, idx + 1, *num_leaves) / 3) + 1;
}

__global__ void _merge_leaves(const uint32_t *codes,
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
        left = _search_lr(
            codes, leaf_depth, -1, idx, depth_delta, *num_leaves);
        right = _search_lr(
            codes, leaf_depth, 1, idx, depth_delta, *num_leaves);
        // Number of codes contained in the candidate merged leaf
        num_codes = leaf_first_in[idx + right + 1] -
                    leaf_first_in[idx - left];
    }

    // Keeping only leaf entries at the start of each merged leaf
    leaf_flagged[idx] = left_old == 0;
}

// TODO: improve coalescence
__global__ void _build_radix_tree(const uint32_t *codes,
                                  struct Btree::Nodes nodes,
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

        // WARNING: make sure this is correct
        nodes.depth[2 * idx + 1] = 2 * num_leaves; // +1 just to be sure
        nodes.depth[2 * idx + 2] = 2 * num_leaves; // +1 just to be sure
    }

    if (idx >= num_internal) {
        return;
    }

    if (idx == 0) {
        nodes.parent[1] = 0;
        nodes.depth[1] = 0;
        // The root node is a valid octree node
        nodes.edge_delta[1] = 1;

#pragma unroll
        for (int i = 0; i < 3; i++) {
            tmp_ranges[max_num_nodes * (i + 1) - 1] = max_num_nodes - 1;
        }
    }

    // Determines whether the left or right
    // leaf is part of the current internal node
    int d = _sign(_lcp(codes, idx, idx + 1, num_leaves) -
                  _lcp(codes, idx, idx - 1, num_leaves));

    // Minimum length of the common prefix between the
    // leaves covered by the current internal node, it
    // is obtained by computing lcp on the non-sibling
    // neighbouring node
    int lcp_min = _lcp(codes, idx, idx - d, num_leaves);

    // Computes upper bound for the length of prefix
    // covered by the current internal node by doubling
    // the search range until a leaf whose lcp is <= delta_min
    int max_length = 2;
    while (_lcp(codes, idx, idx + max_length * d, num_leaves) > lcp_min) {
        max_length = max_length << 1;
    }

    // Uses iterative binary search for the exact end of the
    // range of leaves covered by the current internal node
    int length = 0;
    int step = max_length;
    do {
        // Half the step size
        step = step >> 1;

        if (_lcp(codes, idx, idx + (length + step) * d, num_leaves) >
            lcp_min) {
            length += step;
        }
    } while (step > 1);
    // End of the range of leaves covered
    int last = idx + length * d;

    // Length of prefix covered by the internal node
    int node_lcp = _lcp_safe(codes, idx, last);
    int split = _find_split(codes, idx, last, node_lcp, d, num_leaves);

    int min_leaf = min(idx, last);
    int max_leaf = max(idx, last);
    // Range of leaves covered by the internal node
    nodes.leaves_begin[2 * idx + 1] = min_leaf;
    nodes.leaves_end[2 * idx + 1] = max_leaf;

    _set_node_child(codes,
                    nodes,
                    nodes.left,
                    idx,
                    split,
                    min_leaf,
                    node_lcp);

    _set_node_child(codes,
                    nodes,
                    nodes.right,
                    idx,
                    split + 1,
                    max_leaf,
                    node_lcp);
}

__global__ void _compute_nodes_depth(const int *parent_in,
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
__global__ void _correct_child_pointers(int *left,
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

    // Quite inefficient
    /*
    if (left_value < num_internal)
        left[idx] = map[left_value];

    if (right_value < num_internal)
        right[idx] = map[right_value];
    */
}

Btree::Btree(int max_num_leaves) : _max_num_leaves(max_num_leaves)
{
    cudaMalloc(&_num_leaves, sizeof(int));
    cudaMalloc(&_range, (max_num_leaves + 1) * sizeof(int));
    // Allocating device memory for internal nodes
    cudaMalloc(&_nodes.parent, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_nodes.depth, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_nodes.left, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_nodes.right, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_nodes.leaves_begin, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_nodes.leaves_end, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_nodes.edge_delta, get_max_num_nodes() * sizeof(int));

    cudaMalloc(&_tmp_nodes.parent, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_tmp_nodes.depth, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_tmp_nodes.left, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_tmp_nodes.right, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_tmp_nodes.leaves_begin, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_tmp_nodes.leaves_end, get_max_num_nodes() * sizeof(int));
    cudaMalloc(&_tmp_nodes.edge_delta, get_max_num_nodes() * sizeof(int));

    cudaMalloc(&_octree_map, get_max_num_nodes() * sizeof(int));

    // Allocating buffers to store intermediate computations
    // TODO: align inner buffers
    cudaMalloc(&_tmp, (_max_num_leaves + 1) * 3 * sizeof(int));
    cudaMalloc(&_tmp_ranges, get_max_num_nodes() * 3 * sizeof(int));

    // Allocating tmp arrays for cub operations
    _tmp_sort = nullptr;
    cub::DeviceMergeSort::StableSortPairs(
        _tmp_sort,
        _tmp_sort_size,
        _nodes.depth,
        _tmp_ranges,
        get_max_num_nodes(),
        _sort_op);
    cudaMalloc(&_tmp_sort, _tmp_sort_size);

    _tmp_compact = nullptr;
    cub::DevicePartition::Flagged(_tmp_compact,
                                  _tmp_compact_size,
                                  _tmp_ranges,
                                  _tmp,
                                  _tmp,
                                  _num_leaves,
                                  max_num_leaves + 1);
    cudaMalloc(&_tmp_compact, _tmp_compact_size);

    thrust::sequence(thrust::device,
                     _range,
                     _range + max_num_leaves + 1);

    int n = 1;
    _num_launches_compute_nodes_depth = 0;
    while (n < get_max_num_nodes()) {
        n = n << 1;
        _num_launches_compute_nodes_depth++;
    }
    _num_launches_compute_nodes_depth += 3;
}

void Btree::generate_leaves(const uint32_t *d_sorted_codes,
                            int *d_leaf_first_code_idx,
                            int max_num_codes_per_leaf)
{
    int *leaf_depth = _tmp;
    // Sets the initial depth (octree relative) of each leaf
    // such that neighbouring leaves are not spatially overlapping
    _reduce_leaf_depth<<<_max_num_leaves / THREADS_PER_BLOCK +
                         (_max_num_leaves % THREADS_PER_BLOCK > 0),
                         THREADS_PER_BLOCK>>>(d_sorted_codes,
                                              leaf_depth,
                                              _num_leaves);

    int *tmp_range_out = _tmp + _max_num_leaves + 1;
    int *leaf_flagged = _tmp + 2 * (_max_num_leaves + 1);
    // Merges adjacent leaves until each holds no more than
    // max_num_codes_per_leaf codes
    _merge_leaves<<<(_max_num_leaves + 1) / THREADS_PER_BLOCK +
                    ((_max_num_leaves + 1) % THREADS_PER_BLOCK > 0),
                    THREADS_PER_BLOCK>>>(d_sorted_codes,
                                         leaf_depth,
                                         _range,
                                         tmp_range_out,
                                         leaf_flagged,
                                         max_num_codes_per_leaf,
                                         _num_leaves,
                                         _max_num_leaves);

    // Keeping only the first code of each merged leaf
    cub::DevicePartition::Flagged(_tmp_compact,
                                  _tmp_compact_size,
                                  tmp_range_out,
                                  leaf_flagged,
                                  d_leaf_first_code_idx,
                                  _num_leaves,
                                  _max_num_leaves + 1);
}

void Btree::build(const uint32_t *d_sorted_codes,
                  const int *d_leaf_first_code_idx)
{
    // WARNING: watch out when gathering values from
    // the rear of d_leaf_first_code, it should probably be
    // filled with 0
    uint32_t *leaf_first_code = reinterpret_cast<uint32_t *>(_tmp);

    thrust::gather(thrust::device,
                   d_leaf_first_code_idx,
                   d_leaf_first_code_idx + _max_num_leaves,
                   d_sorted_codes,
                   leaf_first_code);

    _build_radix_tree<<<get_max_num_internal() / THREADS_PER_BLOCK +
                        (get_max_num_internal() % THREADS_PER_BLOCK > 0),
                        THREADS_PER_BLOCK>>>(leaf_first_code,
                                             _nodes,
                                             _tmp_ranges,
                                             _num_leaves,
                                             get_max_num_internal(),
                                             get_max_num_nodes());

    for (int i = 0; i < _num_launches_compute_nodes_depth; ++i) {
        // Maximizing the thread block size here
        _compute_nodes_depth<<<
            get_max_num_nodes() / MAX_THREADS_PER_BLOCK +
            (get_max_num_nodes() % MAX_THREADS_PER_BLOCK > 0),
            MAX_THREADS_PER_BLOCK>>>(_nodes.parent,
                                     _nodes.depth,
                                     _tmp_nodes.parent,
                                     _tmp_nodes.depth,
                                     _num_leaves,
                                     get_max_num_nodes());

        // TODO: consider using std::swap
        swap_ptr(&_nodes.parent, &_tmp_nodes.parent);
        swap_ptr(&_nodes.depth, &_tmp_nodes.depth);
    }
}

// TODO: sorting is costly, compare the traversal
// performance without bfs ordering
void Btree::sort_to_bfs_order()
{
    // Sort arrays by depth

    int *tmp_range = _tmp_ranges;
    int *tmp_perm_in = _tmp_ranges + get_max_num_nodes();
    int *tmp_perm_out = tmp_perm_in + get_max_num_nodes();

    // TODO: custom sort?

    // TODO: try to memcpy to host the actual number of leaves
    // and sort only the used nodes
    cub::DeviceMergeSort::StableSortPairs(
        _tmp_sort,
        _tmp_sort_size,
        _nodes.depth,
        tmp_perm_in,
        get_max_num_nodes(),
        _sort_op);

    int **in_ptrs[5];
    in_ptrs[0] = &_nodes.left;
    in_ptrs[1] = &_nodes.right;
    in_ptrs[2] = &_nodes.edge_delta;
    in_ptrs[3] = &_nodes.leaves_begin;
    in_ptrs[4] = &_nodes.leaves_end;

    int **tmp_ptrs[5];
    tmp_ptrs[0] = &_tmp_nodes.left;
    tmp_ptrs[1] = &_tmp_nodes.right;
    tmp_ptrs[2] = &_tmp_nodes.edge_delta;
    tmp_ptrs[3] = &_tmp_nodes.leaves_begin;
    tmp_ptrs[4] = &_tmp_nodes.leaves_end;

#pragma unroll
    for (int i = 0; i < 5; ++i) {
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

    _correct_child_pointers<<<get_max_num_nodes() / THREADS_PER_BLOCK +
                              (get_max_num_nodes() % THREADS_PER_BLOCK > 0),
                              THREADS_PER_BLOCK>>>(_nodes.left,
                                                   _nodes.right,
                                                   tmp_perm_out,
                                                   _num_leaves);
}

void Btree::compute_octree_map()
{
    // Given an internal node i, if _edge_delta[i] > 0
    // _octree_map[i] will contain the index of the
    // corresponding octree node

    // TODO: use cub::DeviceScan
    thrust::exclusive_scan(thrust::device,
                           _nodes.edge_delta,
                           _nodes.edge_delta + get_max_num_nodes(),
                           _octree_map);
}

Btree::~Btree()
{
    // Releasing device memory
    cudaFree(_num_leaves);

    cudaFree(_nodes.parent);
    cudaFree(_nodes.left);
    cudaFree(_nodes.right);
    cudaFree(_nodes.leaves_begin);
    cudaFree(_nodes.leaves_end);
    cudaFree(_nodes.edge_delta);

    cudaFree(_tmp_nodes.parent);
    cudaFree(_tmp_nodes.left);
    cudaFree(_tmp_nodes.right);
    cudaFree(_tmp_nodes.leaves_begin);
    cudaFree(_tmp_nodes.leaves_end);
    cudaFree(_tmp_nodes.edge_delta);

    cudaFree(_octree_map);

    cudaFree(_tmp);
    cudaFree(_tmp_ranges);
    cudaFree(_range);

    cudaFree(_tmp_sort);
    cudaFree(_tmp_compact);
}
