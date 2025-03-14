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
    Octree(int max_num_leaves);
    Octree(int max_num_leaves, int *d_points_begin, int *d_points_end);

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

        for (int i = 0; i < min(100, _max_num_nodes); ++i)
        {
            printf("%2d: [", i);
            for (int j = 0; j < num_children[i]; ++j) {
                printf(" %3d", first_child[i] + j);
            }
            printf("] - (%.3f, %.3f, %.3f) - %3d %3d\n",
                   x_barycenter[i], y_barycenter[i], z_barycenter[i],
                   leaves_begin[i], leaves_end[i]);
        }
    }

    ~Octree();

private:
    int _max_depth;
    int _max_num_nodes;

    int *_num_nodes;

    SoAOctreeNodes _nodes;
    // Barycenter coordinates of the points within each node
    SoAVec3<T> _barycenters;
    // Arrays to store the range of points covered by each node
    int *_points_begin;
    int *_points_end;

    bool _gl_buffers;
};

#endif

