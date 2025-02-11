#ifndef BTREE_GPU_CUH
#define BTREE_GPU_CUH

#include "utils_gpu.cuh"

typedef unsigned int uint32_t;

// SoA to store the binary radix tree
class Btree
{
// TODO: consider converting to unsigned int
public:
    Btree(int num_leaves);

    // Builds the binary radix tree given the sorted morton encoded codes
    void build(uint32_t *d_sorted_codes);

    // Sorts internal nodes by depth to allow efficient bfs traversal
    void sort_to_bfs_order();

    ~Btree();

    __device__ __forceinline__
    bool is_leaf(int node) {
        return node >= _num_leaves - 1;
    }

    __device__ __forceinline__
    bool is_octree_node(int idx) {
        return _edge_delta[idx] > 0;
    }

    __device__ __forceinline__
    int get_num_leaves() {
        return _num_leaves;
    }

    __device__ __forceinline__
    int get_left(int idx) {
        return _left[idx];
    }

    __device__ __forceinline__
    int get_right(int idx) {
        return _right[idx];
    }

    __device__ __forceinline__
    int get_octree_node(int idx) {
        return _octree_map[idx];
    }

    __device__ __forceinline__
    int get_leaf(int node)
    {
        // Removes offset from leaf node
        return node - _num_leaves + 1;
    }

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

    // Computes the map between radix tree nodes and octree nodes
    void _compute_octree_map();

    // Computes how many octree nodes correspond
    // to ascending edge of the given node
    __device__ __forceinline__
    void _compute_edge_delta(int node, int parent_lcp)
    {
        // int node_lcp = is_leaf ? _MAX_LCP : parent_lcp + 1;
        int node_lcp = parent_lcp + 1;

        // When _edge_delta[node] > 1, the parent has no siblings
        _edge_delta[node] = min(1, node_lcp / 3 - parent_lcp / 3);
    }

    // Setter for either the left or right child of an internal node
    __device__ __forceinline__
    void _set_child(int *dest, int parent, int child, int lcp, bool is_leaf)
    {
        // Pointers to leaf nodes are offset by the
        // number of internal nodes
        child += (_num_leaves - 1) * is_leaf;
        dest[parent] = child;
        if (!is_leaf) {
            _compute_edge_delta(child, lcp);
        }
    }

    // Number of internal nodes is _num_leaves - 1
    // TODO: const?
    int _num_leaves;

    // Arrays to store pointers (indices) to left and right children
    // of the internal nodes
    int *_left;
    int *_right;

    // Array to depth of the internal nodes
    int *_depth;
    // Array to store the number of octree nodes that correspond
    // to the ascending edge of each radix tree internal node
    int *_edge_delta;
    // Array to store the map between radix tree internal nodes
    // and octree nodes
    int *_octree_map;

    // Temporary arrays used for sorting
    int *_tmp_perm1;
    int *_tmp_perm2;
    int *_tmp_range;
};

extern __constant__ Btree d_btree;

__global__ void morton_encode(Points *points, uint32_t *codes);

#endif
