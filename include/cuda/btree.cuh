#ifndef BTREE_CUH
#define BTREE_CUH

#include "cuda/utils.cuh"
#include "cuda/soa_btree_nodes.cuh"

#include <iostream>
#include <vector>

class Btree
{
// TODO: consider converting to unsigned int
public:
    Btree(int max_num_leaves);

    const SoABtreeNodes &get_d_nodes() const
    {
        return _nodes;
    }

    int *get_d_num_leaves_ptr() const
    {
        return _num_leaves;
    }

    const int *get_d_leaf_first_code_idx_ptr() const
    {
        return _leaf_first_code_idx;
    }

    const int *get_d_octree_map_ptr() const
    {
        return _octree_map;
    }

    int get_num_leaves() const
    {
        int num_leaves;

        cudaMemcpy(&num_leaves,
                   _num_leaves,
                   sizeof(int),
                   cudaMemcpyDeviceToHost);

        return num_leaves;
    }

    int get_max_num_internal() const
    {
        return _max_num_leaves - 1;
    }

    int get_max_num_nodes() const
    {
        return 2 * _max_num_leaves - 1;
    }

    void set_max_num_leaves(int max_num_leaves)
    {
        _max_num_leaves = max_num_leaves;
    }

    void reset_max_num_leaves();

    // Generates leaf nodes such that each contain no more than
    // max_num_points_per_leaf
    void generate_leaves(const morton_t *d_sorted_codes,
                         int max_num_codes_per_leaf);

    // Builds the binary radix tree given the sorted morton encoded codes
    void build(const morton_t *d_sorted_codes);

    // Sorts internal nodes by depth to allow efficient bfs traversal
    void sort_to_bfs_order();

    // Computes the map between radix tree nodes and octree nodes
    void compute_octree_map();

    void print()
    {
        std::vector<int> left(get_max_num_nodes());
        std::vector<int> right(get_max_num_nodes());
        std::vector<int> begin(get_max_num_nodes());
        std::vector<int> end(get_max_num_nodes());
        std::vector<int> depth(get_max_num_nodes());
        std::vector<int> edge(get_max_num_nodes());
        std::vector<int> map(get_max_num_nodes());
        std::vector<int> parent(get_max_num_nodes());
        std::vector<int> perm(get_max_num_nodes());

        cudaMemcpy(left.data(),
                   _nodes._left,
                   left.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(right.data(),
                   _nodes._right,
                   right.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(begin.data(),
                   _nodes._leaves_begin,
                   begin.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(end.data(),
                   _nodes._leaves_end,
                   end.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(depth.data(),
                   _nodes._depth,
                   depth.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(edge.data(),
                   _nodes._edge_delta,
                   edge.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(map.data(),
                   _octree_map,
                   map.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(parent.data(),
                   _nodes._parent,
                   parent.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(perm.data(),
                   _tmp_ranges + get_max_num_nodes(),
                   perm.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);


        for (int i = 0; i < get_max_num_nodes(); ++i)
        {
            printf("%2d -> %2d: %2d - %2d - depth: %d - edge: %d - "
                   "octree: %2d - range: %2d %2d - parent: %d\n",
                   perm[i], i, left[i], right[i], depth[i], edge[i], map[i],
                   begin[i], end[i], parent[i]);
        }
    }

    ~Btree();

private:
    const int _init_max_num_leaves;

    int _max_num_leaves;
    int *_num_leaves;

    SoABtreeNodes _nodes;
    SoABtreeNodes _tmp_nodes;

    // Contains the index of the first code covered by each leaf
    int *_leaf_first_code_idx;
    // Stores the index of the corresponding octree node
    // for the btree nodes whose edge_delta is > 0
    int *_octree_map;

    // Buffers used for temporary storage
    int *_tmp;
    morton_t *_tmp_morton;
    int *_tmp_ranges;

    // Data used to sort internal nodes
    int *_tmp_sort;
    size_t _tmp_sort_size;
    LessOp _sort_op;
    // Storage used for leaves compaction
    int *_range;
    int *_tmp_compact;
    size_t _tmp_compact_size;

    int _num_launches_compute_nodes_depth;
};

#endif
