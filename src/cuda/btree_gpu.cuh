#ifndef BTREE_GPU_CUH
#define BTREE_GPU_CUH

#include "utils_gpu.cuh"

#include <iostream>
#include <vector>

typedef unsigned int uint32_t;

// SoA to store the binary radix tree
class Btree
{
// TODO: consider converting to unsigned int
public:
    Btree(int max_num_leaves);

    // Returns a pointer to the object copy in device memory
    Btree *get_dev_ptr()
    {
        return _d_this;
    }

    // Returns a pointer to the object copy _num_leaves
    int *get_dev_num_leaves_ptr()
    {
        return _num_leaves;
    }

    // Generates leaf nodes such that each contain no more than
    // max_num_points_per_leaf
    void generate_leaves(uint32_t *d_sorted_codes,
                         int *d_leaf_first_code,
                         int max_num_codes_per_leaf);

    // Builds the binary radix tree given the sorted morton encoded codes
    void build(uint32_t *d_sorted_codes, int *d_leaf_first_code);

    // Sorts internal nodes by depth to allow efficient bfs traversal
    void sort_to_bfs_order();

    // Computes the map between radix tree nodes and octree nodes
    void compute_octree_map();

    void print()
    {
        std::vector<int> left(get_max_num_internal());
        std::vector<int> right(get_max_num_internal());
        std::vector<int> begin(get_max_num_internal());
        std::vector<int> end(get_max_num_internal());
        std::vector<int> depth(get_max_num_internal());
        std::vector<int> edge(get_max_num_internal());
        std::vector<int> map(get_max_num_internal());

        cudaMemcpy(left.data(),
                   _left,
                   left.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(right.data(),
                   _right,
                   right.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(begin.data(),
                   _leaves_begin,
                   begin.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(end.data(),
                   _leaves_end,
                   end.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(depth.data(),
                   _depth,
                   depth.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(edge.data(),
                   _edge_delta,
                   edge.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(map.data(),
                   _octree_map,
                   map.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < get_max_num_internal(); ++i)
        {
            printf("%2d: %2d - %2d - depth: %d - edge: %d - "
                   "octree: %2d - range: %2d %2d\n",
                   i, left[i], right[i], depth[i], edge[i], map[i],
                   begin[i], end[i]);
        }
    }

    ~Btree();

    __device__ __forceinline__
    bool is_leaf(int node) {
        return node >= get_num_internal();
    }

    __device__ __forceinline__
    bool is_octree_node(int idx) {
        return _edge_delta[idx] > 0;
    }

    __device__ __forceinline__
    int get_num_leaves() {
        return *_num_leaves;
    }

    __device__ __forceinline__
    int get_num_internal() {
        return get_num_leaves() - 1;
    }

    __host__ __device__ __forceinline__
    int get_max_num_internal() {
        return _max_num_leaves - 1;
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
    int get_leaves_begin(int idx)
    {
        return _leaves_begin[idx];
    }

    __device__ __forceinline__
    int get_leaves_end(int idx)
    {
        return _leaves_end[idx];
    }

    __device__ __forceinline__
    int get_octree_node(int idx) {
        return _octree_map[idx];
    }

    __device__ __forceinline__
    int get_leaf(int node)
    {
        // Removes offset from leaf node
        return node - get_num_internal();
    }

    __device__ __forceinline__
    int get_num_octree_nodes()
    {
        return _octree_map[get_num_internal() - 1];
    }

    __device__ __forceinline__
    void set_left(int idx, int left, int parent_lcp, int lcp, bool is_leaf)
    {
        _set_child(_left, idx, left, parent_lcp, lcp, is_leaf);
    }

    __device__ __forceinline__
    void set_right(int idx, int right, int parent_lcp, int lcp, bool is_leaf)
    {
        _set_child(_right, idx, right, parent_lcp, lcp, is_leaf);
    }

    __device__ __forceinline__ 
    void set_depth(int idx, int depth)
    {
        _depth[idx] = depth;
    }

    __device__ __forceinline__
    void set_edge_delta(int idx, int edge_delta)
    {
        _edge_delta[idx] = edge_delta;
    }

    __device__ __forceinline__
    void set_leaves_range(int idx, int begin, int end)
    {
        _leaves_begin[idx] = begin;
        _leaves_end[idx] = end;
    }

    __device__ __forceinline__
    void fill_tmp(int idx, int value)
    {
        _tmp_perm1[idx] = value;
        _tmp_perm2[idx] = value;
        _tmp_range[idx] = value;
    }

private:
    static constexpr int _MAX_LCP = 32;

    // Computes how many octree nodes correspond
    // to ascending edge of the given node
    __device__ __forceinline__
    void _compute_edge_delta(int node, int parent_lcp, int node_lcp)
    {
        // When _edge_delta[node] > 1, the node represent successive
        // descendants of octree nodes each having one child,
        // we compress the octree by ignoring the intermediate nodes
        // with no siblings
        _edge_delta[node] = min(1, node_lcp / 3 - parent_lcp / 3);
    }

    // Setter for either the left or right child of an internal node
    __device__ __forceinline__
    void _set_child(int *dest,
                    int parent,
                    int child,
                    int parent_lcp,
                    int child_lcp,
                    bool is_leaf)
    {
        // Pointers to leaf nodes are offset by the
        // number of internal nodes
        child += get_num_internal() * is_leaf;
        dest[parent] = child;
        if (!is_leaf) {
            _compute_edge_delta(child, parent_lcp, child_lcp);
        }
    }

    int _max_num_leaves;
    int *_num_leaves;

    // Arrays used to generate leaf nodes containing
    // a given maximum number of codes (~points)
    int *_leaf_depth1;
    int *_leaf_depth2;
    int *_leaf_flagged;
    int *_leaf_tmp_range;
    // Storage used for leaves compaction
    int *_tmp_compact;
    size_t _tmp_compact_size;

    // Arrays to store pointers (indices) to left and right children
    // of the internal nodes
    int *_left;
    int *_right;
    // Arrays to store the range of leaves covered by the internal nodes
    int *_leaves_begin;
    int *_leaves_end;
    // Array to depth of the internal nodes
    int *_depth;
    // Array to store the number of octree nodes that correspond
    // to the ascending edge of each radix tree internal node
    int *_edge_delta;
    // Array to store the map between radix tree internal nodes
    // and octree nodes
    int *_octree_map;

    // Data used to sort internal nodes
    int *_tmp_sort;
    size_t _tmp_sort_size;
    LessOp _sort_op;
    int *_tmp_perm1;
    int *_tmp_perm2;
    int *_tmp_range;

    // Pointer to object copy in device memory
    Btree *_d_this;
};

__global__ void morton_encode(Points *points, uint32_t *codes);

#endif
