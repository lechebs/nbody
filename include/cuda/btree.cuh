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
        return nodes_;
    }

    int *get_d_num_leaves_ptr() const
    {
        return num_leaves_;
    }

    const int *get_d_leaf_first_code_idx_ptr() const
    {
        return leaf_first_code_idx_;
    }

    const int *get_d_octree_map_ptr() const
    {
        return octree_map_;
    }

    int get_num_leaves() const
    {
        int num_leaves;

        cudaMemcpy(&num_leaves,
                   num_leaves_,
                   sizeof(int),
                   cudaMemcpyDeviceToHost);

        return num_leaves;
    }

    int get_max_num_internal() const
    {
        return max_num_leaves_ - 1;
    }

    int get_max_num_nodes() const
    {
        return 2 * max_num_leaves_ - 1;
    }

    void set_max_num_leaves(int max_num_leaves)
    {
        max_num_leaves_ = max_num_leaves;
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

    ~Btree();

private:
    const int init_max_num_leaves_;

    int max_num_leaves_;
    int *num_leaves_;

    SoABtreeNodes nodes_;
    SoABtreeNodes tmp_nodes_;

    // Contains the index of the first code covered by each leaf
    int *leaf_first_code_idx_;
    // Stores the index of the corresponding octree node
    // for the btree nodes whose edge_delta is > 0
    int *octree_map_;

    // Buffers used for temporary storage
    int *tmp_;
    morton_t *tmp_morton_;
    int *tmp_ranges_;

    // Data used to sort internal nodes
    int *tmp_sort_;
    size_t tmp_sort_size_;
    LessOp sort_op_;
    // Storage used for leaves compaction
    int *range_;
    int *tmp_compact_;
    size_t tmp_compact_size_;

    int num_launches_compute_nodes_depth_;
};

#endif
