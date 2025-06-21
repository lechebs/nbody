#ifndef OCTREE_CUH
#define OCTREE_CUH

#include "cuda/soa_octree_nodes.cuh"
#include "cuda/btree.cuh"
#include "cuda/points.cuh"
#include "cuda/utils.cuh"

#include <iostream>
#include <vector>

// SoA to store the octree
template<typename T> class Octree
{
public:
    Octree(int max_num_leaves, float domain_size);
    Octree(int max_num_leaves,
           float domain_size,
           T *d_barycenters_x,
           T *d_barycenters_y,
           T *d_barycenters_z,
           T *d_nodes_size);

    const SoAOctreeNodes get_d_nodes() const
    {
        return nodes_;
    }

    const T *get_d_nodes_size() const
    {
        return nodes_size_;
    }

    const T *get_d_nodes_mass() const
    {
        return nodes_mass_;
    }

    const SoAVec3<T> get_d_barycenters() const
    {
        return barycenters_;
    }

    const int *get_d_points_begin_ptr() const
    {
        return points_begin_;
    }

    const int *get_d_points_end_ptr() const
    {
        return points_end_;
    }

    int get_num_nodes()
    {
        int num_nodes;
        cudaMemcpy(&num_nodes,
                   num_nodes_,
                   sizeof(int),
                   cudaMemcpyDeviceToHost);
        return num_nodes;
    }

    void set_max_num_nodes(int max_num_nodes)
    {
        max_num_nodes_ = max_num_nodes;
    }

    void build(const Btree &btree);

    void compute_nodes_points_range(const int *d_leaf_first_code_idx,
                                    const int *d_code_first_point_idx);

    void compute_nodes_barycenter(const Points<T> &points);

    ~Octree();

private:
    void init_(int max_num_leaves);

    float domain_size_;

    int max_depth_;
    int max_num_nodes_;

    int *num_nodes_;

    SoAOctreeNodes nodes_;
    // Array to store the side length of each octree node
    T *nodes_size_;
    // Array to store the mass of each octree node
    T *nodes_mass_;
    // Barycenter coordinates of the points within each node
    SoAVec3<T> barycenters_;
    // Arrays to store the range of points covered by each node
    int *points_begin_;
    int *points_end_;

    bool gl_buffers_;
};

#endif

