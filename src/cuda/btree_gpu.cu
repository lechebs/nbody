#include "btree_gpu.cuh"

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
__global__ void morton_encode(Points *points, uint32_t *codes)
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
__device__ __forceinline__ int _lcp_safe(uint32_t *codes, int i, int j)
{
    // Does this allow coalescing?
    return __clz(codes[i] ^ codes[j]) - 2; // Removing leading 00
}

__device__ __forceinline__ int _lcp(uint32_t *codes, int i, int j, int n)
{
    // i index is always in range
    if (j < 0 || j > n - 1)
        return -1;
    else 
        return _lcp_safe(codes, i, j);
}

__device__ __forceinline__ int _sign(int x)
{
    return (x > 0) - (x < 0);
}

__device__ int _find_split(uint32_t *codes,
                           int first,
                           int last,
                           int node_lcp,
                           int dir)
{
    int step = 0;
    int length = (last - first) * dir;

    do {
        length = (length + 1) >> 1;

        if (_lcp_safe(
                codes, first, first + (step + length) * dir) >
                node_lcp) {
            step += length;
        }
    } while (length > 1);

    return first + step * dir + min(dir, 0);
}

__device__ int _search_lr(uint32_t *codes,
                          int *leaf_depth,
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

__global__ void _reduce_leaf_depth(uint32_t *codes,
                                   int *leaf_depth,
                                   int *num_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_leaves) {
        return;
    }

    leaf_depth[idx] = max(_lcp(codes, idx, idx - 1, *num_leaves) / 3,
                          _lcp(codes, idx, idx + 1, *num_leaves) / 3) + 1;
}

__global__ void _merge_leaves(uint32_t *codes,
                              int *leaf_depth_in,
                              int *leaf_depth_out,
                              int *leaf_first_in,
                              int *leaf_first_out,
                              int *leaf_flagged,
                              int max_num_codes_per_leaf,
                              int *num_leaves,
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

    int left_old;
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
            codes, leaf_depth_in, -1, idx, depth_delta, *num_leaves);
        right = _search_lr(
            codes, leaf_depth_in, 1, idx, depth_delta, *num_leaves);
        // Number of codes contained in the candidate merged leaf
        num_codes = leaf_first_in[idx + right + 1] -
                    leaf_first_in[idx - left];
    }

    // Reduce delta_depth since the merge proposal
    // from the last iterations is always rejected
    leaf_depth_out[idx] = leaf_depth_in[idx] - max(0, depth_delta - 1);
    // Keeping only leaf entries at the start of each merged leaf
    leaf_flagged[idx] = left_old == 0;
}

// TODO: consider passing directly the relevant arrays
__global__ void _build_radix_tree(uint32_t *codes,
                                  int *parent,
                                  int *depth,
                                  int *edge_delta,
                                  Btree &btree)
{
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    int num_leaves = btree.get_num_leaves();

    if (first < btree.get_max_num_internal()) {
        // Setting maximum depth for unused nodes
        btree.set_depth(first, num_leaves);
    }

    if (first >= btree.get_num_internal()) {
        return;
    }

    // Filling tmp ranges used to later sort tree nodes
    btree.fill_tmp(first, first);
    // The root node is a valid octree node
    if (first == 0) {
        parent[first] = 0;
        depth[first] = 0;
        btree.set_edge_delta(first, 1);
    }

    // Determines whether the left or right
    // leaf is part of the current internal node
    int d = _sign(_lcp(codes, first, first + 1, num_leaves) -
                  _lcp(codes, first, first - 1, num_leaves));

    // Minimum length of the common prefix between the
    // leaves covered by the current internal node, it
    // is obtained by computing lcp on the non-sibling
    // neighbouring node
    int lcp_min = _lcp(codes, first, first - d, num_leaves);

    // Computes upper bound for the length of prefix
    // covered by the current internal node by doubling
    // the search range until a leaf whose lcp is <= delta_min
    int max_length = 2;
    while (_lcp(codes, first, first + max_length * d, num_leaves) > lcp_min) {
        max_length = max_length << 1;
    }

    // Uses iterative binary search for the exact end of the
    // range of leaves covered by the current internal node
    int length = 0;
    int step = max_length;
    do {
        // Half the step size
        step = step >> 1;
        if (_lcp(codes, first, first + (length + step) * d, num_leaves) >
            lcp_min) {
            length += step;
        }
    } while (step > 1);
    // End of the range of leaves covered
    int last = first + length * d;

    // Length of prefix covered by the internal node
    int node_lcp = _lcp_safe(codes, first, last);
    int split = _find_split(codes, first, last, node_lcp, d);

    int min_leaf = min(first, last);
    int max_leaf = max(first, last);

    bool is_left_leaf = min_leaf == split;
    bool is_right_leaf = max_leaf == split + 1;

    int left_lcp = _lcp_safe(codes, min_leaf, split);
    int right_lcp = _lcp_safe(codes, split + 1, max_leaf);

    // Record parent-child relationships
    btree.set_left(first, split, node_lcp, left_lcp, is_left_leaf);
    btree.set_right(first, split + 1, node_lcp, right_lcp, is_right_leaf);

    // TODO: prefill parent array with 0s

    if (!is_left_leaf) {
        depth[split] = edge_delta[split];
        parent[split] = first;
    }
    if (!is_right_leaf) {
        depth[split + 1] = edge_delta[split + 1];
        parent[split + 1] = first;
    }

    // btree.set_depth(first, node_lcp / 3);
    // printf("[%2d] node_lcp=%d\n", first, node_lcp);
    btree.set_leaves_range(first, min_leaf, max_leaf);
}

__global__ void _compute_nodes_depth(int *parent, int *depth, int *num_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_leaves - 1) {
        return;
    }

    int parent_idx = parent[idx];
    int curr_depth = depth[idx];
    // If parent is not the root
    if (parent_idx && curr_depth > 0) {
        depth[idx] = curr_depth + depth[parent_idx];
        parent[idx] = parent[parent_idx];
    }
}

__global__ void _copy_parent_depth(int *parent,
                                   int *depth,
                                   int *edge_delta,
                                   int *num_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_leaves - 1) {
        return;
    }

    int curr_idx = idx;
    while (edge_delta[curr_idx] == 0) {
        curr_idx = parent[curr_idx];
    }

    // printf("%2d -> %2d\n", idx, curr_idx);
    depth[idx] = depth[curr_idx];
}

// Used to update pointers to children after sorting the nodes
__global__ void _correct_child_pointers(int *left,
                                        int *right,
                                        int *map,
                                        Btree &btree)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_internal = btree.get_num_internal();

    if (idx >= num_internal) {
        return;
    }

    int left_value = left[idx];
    int right_value = right[idx];

    // Quite inefficient
    if (left_value < num_internal)
        left[idx] = map[left_value];

    if (right_value < num_internal)
        right[idx] = map[right_value];
}

Btree::Btree(int max_num_leaves) : _max_num_leaves(max_num_leaves)
{
    cudaMalloc(&_num_leaves, sizeof(int));
    // Allocating device memory for leaf nodes
    cudaMalloc(&_leaf_depth1, max_num_leaves * sizeof(int));
    cudaMalloc(&_leaf_depth2, max_num_leaves * sizeof(int));
    cudaMalloc(&_leaf_flagged, (max_num_leaves + 1) * sizeof(int));
    cudaMalloc(&_leaf_tmp_range, (max_num_leaves + 1) * sizeof(int));
    // Allocating device memory for internal nodes
    cudaMalloc(&_parent, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_tmp_parent, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_left, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_right, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_leaves_begin, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_leaves_end, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_depth, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_edge_delta, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_octree_map, get_max_num_internal() * sizeof(int));

    cudaMalloc(&_tmp_left, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_tmp_right, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_tmp_leaves_begin, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_tmp_leaves_end, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_tmp_edge_delta, get_max_num_internal() * sizeof(int));

    // Allocating device memory used for sorting operations
    // TODO: consider using cub::CachingDeviceAllocator
    cudaMalloc(&_tmp_range, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_tmp_perm1, get_max_num_internal() * sizeof(int));
    cudaMalloc(&_tmp_perm2, get_max_num_internal() * sizeof(int));

    // Allocating tmp arrays for cub operations
    _tmp_sort = nullptr;
    cub::DeviceMergeSort::StableSortPairs(
        _tmp_sort,
        _tmp_sort_size,
        _depth,
        thrust::make_zip_iterator(
            thrust::device_pointer_cast(_left),
            thrust::device_pointer_cast(_right),
            thrust::device_pointer_cast(_leaves_begin),
            thrust::device_pointer_cast(_leaves_end),
            thrust::device_pointer_cast(_edge_delta),
            thrust::device_pointer_cast(_tmp_perm1)),
        get_max_num_internal(),
        _sort_op);
    cudaMalloc(&_tmp_sort, _tmp_sort_size);

    _tmp_compact = nullptr;
    cub::DevicePartition::Flagged(_tmp_compact,
                                  _tmp_compact_size,
                                  _leaf_tmp_range,
                                  _leaf_flagged,
                                  _leaf_tmp_range,
                                  _num_leaves,
                                  max_num_leaves + 1);
    cudaMalloc(&_tmp_compact, _tmp_compact_size);

    // Allocating object copy in device memory
    cudaMalloc(&_d_this, sizeof(Btree));
    cudaMemcpy(_d_this, this, sizeof(Btree), cudaMemcpyHostToDevice);

    thrust::sequence(thrust::device,
                     _leaf_tmp_range,
                     _leaf_tmp_range + max_num_leaves + 1);
}

void Btree::generate_leaves(uint32_t *d_sorted_codes,
                            int *d_leaf_first_code,
                            int max_num_codes_per_leaf)
{
    // Sets the initial depth (octree relative) of each leaf
    // such that neighbouring leaves are not spatially overlapping
    _reduce_leaf_depth<<<_max_num_leaves / THREADS_PER_BLOCK +
                         (_max_num_leaves % THREADS_PER_BLOCK > 0),
                         THREADS_PER_BLOCK>>>(d_sorted_codes,
                                              _leaf_depth1,
                                              _num_leaves);

    // TODO: make class member, consider sharing it with sorting tmp
    static thrust::device_vector<int> tmp_range_out(_max_num_leaves + 1);

    // Merges adjacent leaves until each holds no more than
    // max_num_codes_per_leaf codes
    _merge_leaves<<<(_max_num_leaves + 1) / THREADS_PER_BLOCK +
                    ((_max_num_leaves + 1) % THREADS_PER_BLOCK > 0),
                    THREADS_PER_BLOCK>>>(d_sorted_codes,
                                         _leaf_depth1,
                                         _leaf_depth2,
                                         _leaf_tmp_range,
                                         thrust::raw_pointer_cast(
                                            &tmp_range_out[0]),
                                         _leaf_flagged,
                                         max_num_codes_per_leaf,
                                         _num_leaves,
                                         _max_num_leaves);

    // Keeping only the first code of each merged leaf
    cub::DevicePartition::Flagged(_tmp_compact,
                                  _tmp_compact_size,
                                  thrust::raw_pointer_cast(
                                    &tmp_range_out[0]),
                                  _leaf_flagged,
                                  d_leaf_first_code,
                                  _num_leaves,
                                  _max_num_leaves + 1);
}

void Btree::build(uint32_t *d_sorted_codes, int *d_leaf_first_code)
{
    // TODO: make class member, consider sharing with sorting tmp
    static thrust::device_vector<uint32_t> d_leaf_codes(_max_num_leaves);

    // WARNING: watch out when gathering values from
    // the rear of d_leaf_first_code
    thrust::gather(thrust::device,
                   d_leaf_first_code,
                   d_leaf_first_code + _max_num_leaves,
                   d_sorted_codes,
                   d_leaf_codes.begin());

    _build_radix_tree<<<get_max_num_internal() / THREADS_PER_BLOCK +
                        (get_max_num_internal() % THREADS_PER_BLOCK > 0),
                        THREADS_PER_BLOCK>>>(
                            d_sorted_codes,
                            _parent,
                            _depth,
                            _edge_delta,
                            //thrust::raw_pointer_cast(&d_leaf_codes[0]),
                            *_d_this);

    cudaMemcpy(_tmp_parent,
               _parent,
               get_max_num_internal() * sizeof(int),
               cudaMemcpyDeviceToDevice);

    int n = 1;
    int num_launches = 0;
    while (n < get_max_num_internal()) {
        n = n << 1;
        num_launches++;
    }

    std::cout << "num_launches=" << num_launches << std::endl;

    for (int i = 0; i < num_launches + 3; ++i) {
        _compute_nodes_depth<<<get_max_num_internal() / THREADS_PER_BLOCK +
                               (get_max_num_internal() % THREADS_PER_BLOCK > 0),
                               THREADS_PER_BLOCK>>>(_tmp_parent,
                                                    _depth,
                                                    _num_leaves);
    }

    /*
    _copy_parent_depth<<<get_max_num_internal() / THREADS_PER_BLOCK +
                         (get_max_num_internal() % THREADS_PER_BLOCK > 0),
                         THREADS_PER_BLOCK>>>(_parent,
                                              _depth,
                                              _edge_delta,
                                              _num_leaves);
    */
}

// TODO: sorting is costly, compare the traversal
// performance without bfs ordering
void Btree::sort_to_bfs_order()
{
    // Sort arrays by depth

    // TODO: custom sort?
    cub::DeviceMergeSort::StableSortPairs(
        _tmp_sort,
        _tmp_sort_size,
        _depth,
        // TODO: avoid creating zip_iterator on the fly
        // and test by sorting only _tmp_perm1 and gathering
        // the other arrays
        //_tmp_perm1,
        thrust::make_zip_iterator(
            thrust::device_pointer_cast(_left),
            thrust::device_pointer_cast(_right),
            thrust::device_pointer_cast(_leaves_begin),
            thrust::device_pointer_cast(_leaves_end),
            thrust::device_pointer_cast(_edge_delta),
            thrust::device_pointer_cast(_tmp_perm1)),
        get_max_num_internal(),
        _sort_op);

    /*
    thrust::gather(thrust::device,
                   _tmp_perm1,
                   _tmp_perm1 + get_max_num_internal(),
                   _left,
                   _tmp_left);
    */

    /*
    tmp_ptr = _left;
    _left = _tmp_left;
    _tmp_left = tmp_ptr;

    thrust::gather(thrust::device,
                   _tmp_perm1,
                   _tmp_perm1 + get_max_num_internal(),
                   _right,
                   _tmp_right);
    thrust::gather(thrust::device,
                   _tmp_perm1,
                   _tmp_perm1 + get_max_num_internal(),
                   _leaves_begin,
                   _tmp_leaves_begin);
    thrust::gather(thrust::device,
                   _tmp_perm1,
                   _tmp_perm1 + get_max_num_internal(),
                   _leaves_end,
                   _tmp_leaves_end);
    thrust::gather(thrust::device,
                   _tmp_perm1,
                   _tmp_perm1 + get_max_num_internal(),
                   _edge_delta,
                   _tmp_edge_delta);

    tmp_ptr = _right;
    _right = _tmp_right;
    _tmp_right = tmp_ptr;

    tmp_ptr = _leaves_begin;
    _leaves_begin = _tmp_leaves_begin;
    _tmp_leaves_begin = tmp_ptr;

    tmp_ptr = _leaves_end;
    _leaves_end = _tmp_leaves_end;
    _tmp_leaves_end = tmp_ptr;

    tmp_ptr = _edge_delta;
    _edge_delta = _tmp_edge_delta;
    _tmp_edge_delta = tmp_ptr;
    */

    // Reverse _tmp_perm1 mapping into _tmp_perm2
    // TODO: find a way to perform it using cub
    thrust::scatter(thrust::device,
                    _tmp_range,
                    _tmp_range + get_max_num_internal(),
                    _tmp_perm1,
                    _tmp_perm2);

    _correct_child_pointers<<<get_max_num_internal() / THREADS_PER_BLOCK +
                              (get_max_num_internal() % THREADS_PER_BLOCK > 0),
                              THREADS_PER_BLOCK>>>(_left,
                                                   _right,
                                                   _tmp_perm2,
                                                   *_d_this);
}

void Btree::compute_octree_map()
{
    // Given an internal node i, if _edge_delta[i] > 0
    // _octree_map[i] will contain the index of the
    // corresponding octree node

    // TODO: use cub::DeviceScan
    thrust::exclusive_scan(thrust::device,
                           _edge_delta,
                           _edge_delta + get_max_num_internal(),
                           _octree_map);
}

Btree::~Btree()
{
    // Releasing device memory
    cudaFree(_num_leaves);

    cudaFree(_leaf_depth1);
    cudaFree(_leaf_depth2);
    cudaFree(_leaf_flagged);
    cudaFree(_leaf_tmp_range);
    cudaFree(_tmp_compact);

    cudaFree(_parent);
    cudaFree(_tmp_parent);

    cudaFree(_left);
    cudaFree(_right);
    cudaFree(_leaves_begin);
    cudaFree(_leaves_end);
    cudaFree(_depth);
    cudaFree(_edge_delta);
    cudaFree(_octree_map);

    cudaFree(_tmp_left);
    cudaFree(_tmp_right);
    cudaFree(_tmp_leaves_begin);
    cudaFree(_tmp_leaves_end);
    cudaFree(_tmp_edge_delta);

    cudaFree(_tmp_sort);
    cudaFree(_tmp_perm1);
    cudaFree(_tmp_perm2);
    cudaFree(_tmp_range);

    cudaFree(_d_this);
}
