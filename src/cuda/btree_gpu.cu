#include "btree_gpu.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scatter.h>

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
    return __clz(codes[i] ^ codes[j]);
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

__global__ void build_radix_tree(uint32_t *codes, Btree *btree)
{
    int first = blockIdx.x * blockDim.x + threadIdx.x;

    int num_leaves = btree->get_num_leaves();
    if (first > num_leaves - 2) return;

    // Fill tmp arrays used for later computations
    btree->set_tmp_ranges(first, first);

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

    bool is_left_leaf = min(first, last) == split;
    bool is_right_left = max(first, last) == split + 1;

    // Record parent-child relationships
    btree->set_left(first, split, node_lcp, is_left_leaf);
    btree->set_right(first, split + 1, node_lcp, is_right_left);

    btree->set_depth(first, node_lcp / 3);
}

Btree::Btree(int num_leaves) : _num_leaves(num_leaves)
{
    // Allocating device memory for internal nodes
    cudaMalloc(&_left, (num_leaves - 1) * sizeof(int));
    cudaMalloc(&_right, (num_leaves - 1) * sizeof(int));
    cudaMalloc(&_depth, (num_leaves - 1) * sizeof(int));
    // Allocating device memory for internal nodes and leaf nodes
    cudaMalloc(&_edge_delta, (2 * num_leaves - 1) * sizeof(int));

    // Allocating device memory used for intermediate computations
    cudaMalloc(&_tmp_range1, (num_leaves - 1) * sizeof(int));
    cudaMalloc(&_tmp_range2, (num_leaves - 1) * sizeof(int));

    // Allocating object copy in device memory
    cudaMalloc(&_d_this, sizeof(Btree));
    cudaMemcpy(_d_this, this, sizeof(Btree), cudaMemcpyHostToDevice);
}

void Btree::build(uint32_t *d_sorted_codes)
{
    build_radix_tree<<<_num_leaves / THREADS_PER_BLOCK,
                       THREADS_PER_BLOCK>>>(d_sorted_codes, _d_this);
}

void Btree::sort_to_bfs_order()
{
    // Sort arrays by depth
    thrust::stable_sort_by_key(
        thrust::device,
        _depth,
        _depth + _num_leaves - 1,
        // Consider using device_vector for all arrays
        thrust::make_zip_iterator(
            thrust::device_pointer_cast(_left),
            thrust::device_pointer_cast(_right),
            thrust::device_pointer_cast(_edge_delta),
            thrust::device_pointer_cast(_tmp_range1)));

    // Update child pointers

    // TODO: scatter_if _left1 and _right1 with condition
    // _left1[i] (_right1[i]) < num_leaves - 1 using as
    // map _tmp_range1 into _left2 and _right2, then copy_if
    // _left1 and _right1 into _left2 and _right2 using
    // the previous condition negated
}

Btree::~Btree()
{
    // Releasing device memory
    cudaFree(_left);
    cudaFree(_right);
    cudaFree(_depth);
    cudaFree(_edge_delta);

    cudaFree(_d_this);
}


