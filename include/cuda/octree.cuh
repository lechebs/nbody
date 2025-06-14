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
    // max_depth ~= btree_height / 3
    Octree(int max_num_leaves, float domain_size);
    Octree(int max_num_leaves,
           float domain_size,
           T *d_barycenters_x,
           T *d_barycenters_y,
           T *d_barycenters_z,
           T *d_nodes_size);

    const SoAOctreeNodes get_d_nodes() const
    {
        return _nodes;
    }

    const T *get_d_nodes_size() const
    {
        return _nodes_size;
    }

    const SoAVec3<T> get_d_barycenters() const
    {
        return _barycenters;
    }

    const int *get_d_points_begin_ptr() const
    {
        return _points_begin;
    }

    const int *get_d_points_end_ptr() const
    {
        return _points_end;
    }

    int get_num_nodes()
    {
        int num_nodes;
        cudaMemcpy(&num_nodes,
                   _num_nodes,
                   sizeof(int),
                   cudaMemcpyDeviceToHost);
        return num_nodes;
    }

    void set_max_num_nodes(int max_num_nodes)
    {
        _max_num_nodes = max_num_nodes;
    }

    void build(const Btree &btree);

    void compute_nodes_points_range(const int *d_leaf_first_code_idx,
                                    const int *d_code_first_point_idx);

    void compute_nodes_barycenter(const Points<T> &points);

    void print()
    {
        std::vector<int> first_child(_max_num_nodes);
        std::vector<int> num_children(_max_num_nodes);
        std::vector<int> leaves_begin(_max_num_nodes);
        std::vector<int> leaves_end(_max_num_nodes);
        std::vector<float> x_barycenter(_max_num_nodes);
        std::vector<float> y_barycenter(_max_num_nodes);
        std::vector<float> z_barycenter(_max_num_nodes);
        std::vector<float> size(_max_num_nodes);

        cudaMemcpy(first_child.data(),
                   _nodes._first_child,
                   sizeof(int) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(num_children.data(),
                   _nodes._num_children,
                   sizeof(int) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(leaves_begin.data(),
                   _points_begin,
                   sizeof(int) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(leaves_end.data(),
                   _points_end,
                   sizeof(int) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(x_barycenter.data(),
                   _barycenters.x(),
                   sizeof(float) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(y_barycenter.data(),
                   _barycenters.y(),
                   sizeof(float) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(z_barycenter.data(),
                   _barycenters.z(),
                   sizeof(float) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(size.data(),
                   _nodes_size,
                   sizeof(float) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < _max_num_nodes; ++i)
        {
            printf("%2d: [", i);
            for (int j = 0; j < num_children[i]; ++j) {
                printf(" %3d", first_child[i] + j);
            }
            printf("] - (%f, %f, %f) - %3d %3d - %f\n",
                   x_barycenter[i], y_barycenter[i], z_barycenter[i],
                   leaves_begin[i], leaves_end[i], size[i]);

        }
        printf("\n");
    }


    ~Octree();

private:
    void _init(int max_num_leaves);

    float _domain_size;

    int _max_depth;
    int _max_num_nodes;

    int *_num_nodes;

    SoAOctreeNodes _nodes;
    // Array to store the side length of each octree node
    T *_nodes_size;
    // Barycenter coordinates of the points within each node
    SoAVec3<T> _barycenters;
    // Arrays to store the range of points covered by each node
    int *_points_begin;
    int *_points_end;

    bool _gl_buffers;
};

#endif

