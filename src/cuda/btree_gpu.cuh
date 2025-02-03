#ifndef BTREE_GPU_CUH
#define BTREE_GPU_CUH

#include "utils_gpu.cuh"

typedef unsigned int uint32_t;

// SoA to store the binary radix tree
class Btree
{
public:
    Btree(int num_leaves);

    // Builds the binary radix tree given the sorted morton encoded codes
    void build(uint32_t *d_sorted_codes);

    // Sorts internal nodes by depth to allow efficient bfs traversal
    void sort_to_bfs_order();

    ~Btree();

    __device__ __forceinline__
    bool is_leaf(int node) {
        // Pointers to leaf nodes are offset by the
        // number of internal nodes
        return node >= _num_leaves - 1;
    }

    __device__ __forceinline__
    bool is_octree_node(int node) {
        return _edge_delta[node] > 0;
    }

    __device__ __forceinline__
    int get_num_leaves() { return _num_leaves; }

    __device__ __forceinline__
    int get_left(int idx) { return _left[idx]; }

    __device__ __forceinline__
    int get_right(int idx) { return _right[idx]; }

    __device__ __forceinline__
    void set_left(int idx, int left, int lcp, bool is_leaf)
    {
        _set_child(_left, idx, left, lcp, is_leaf);
    }

    __device__ __forceinline__
    void set_right(int idx, int right, int lcp, bool is_leaf)
    {
        _set_child(_right, idx, right, lcp, is_leaf);
    }

    __device__ __forceinline__ 
    void set_depth(int idx, int depth)
    {
        _depth[idx] = depth;
    }

private:
    static constexpr int _MAX_LCP = 32;

    // Computes how many octree nodes correspond
    // to ascending edge of the given node
    __device__ __forceinline__
    void _compute_edge_delta(int node, int parent_lcp, bool is_leaf)
    {
        int node_lcp = is_leaf ? _MAX_LCP : parent_lcp + 1;
        _edge_delta[node] = node_lcp / 3 - parent_lcp / 3;
    }

    // Setter for either the left or right child of an internal node
    __device__ __forceinline__
    void _set_child(int *dest, int parent, int child, int lcp, bool is_leaf)
    {
        child += (_num_leaves - 1) * is_leaf;
        dest[parent] = child;
        _compute_edge_delta(child, lcp, is_leaf);
    }

    // Number of internal nodes is _num_leaves - 1
    int _num_leaves;

    // Arrays to store pointers (indices) to left and right children
    // of the internal nodes
    int *_left;
    int *_right;

    // Array to depth of the internal nodes
    int *_depth;
    // Array to store the number of octree nodes that correspond
    // to the ascending edge of each radix tree node (both internal
    // and leaves)
    int *_edge_delta;

    // Temporary arrays used for sorting
    int *_tmp_perm1;
    int *_tmp_perm2;
    int *_tmp_range;

    // Pointer to current object copy in device memory
    Btree *_d_this;
};

__global__ void morton_encode(Points *points, uint32_t *codes);

#endif
