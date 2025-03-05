#ifndef OCTREE_GPU_CUH
#define OCTREE_GPU_CUH

#include "btree_gpu.cuh"
#include "utils_gpu.cuh"

#include <iostream>
#include <vector>

// TODO: consider having distinct cpu and gpu
// copies of the object

// SoA to store the octree
class Octree
{
public:
    struct Nodes {
        // Array to store pointers (indices) to the children of each node,
        // each groups of 8 siblings is contiguous in memory
        int *children;
        // Array to store the number of children of each internal node
        int *num_children;
        // Arrays to store the range of leaves covered by the internal nodes
        int *leaves_begin;
        int *leaves_end;
        // Barycenter coordinates of the points within each node
        float *x_barycenter;
        float *y_barycenter;
        float *z_barycenter;
    };

    // max_depth ~= btree_height / 3
    Octree(int max_num_nodes);

    void build(const Btree &btree);

    void compute_nodes_barycenter(const Points *points,
                                  const Points *scan_points,
                                  const int *leaf_first_code_idx,
                                  const int *scan_codes_occurrences);

    void print()
   {
        std::vector<int> children(_max_num_nodes * 8);
        std::vector<int> num_children(_max_num_nodes);
        std::vector<int> leaves_begin(_max_num_nodes);
        std::vector<int> leaves_end(_max_num_nodes);
        std::vector<float> x_barycenter(_max_num_nodes);
        std::vector<float> y_barycenter(_max_num_nodes);
        std::vector<float> z_barycenter(_max_num_nodes);

        cudaMemcpy(children.data(),
                   _nodes.children,
                   sizeof(int) * _max_num_nodes * 8,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(num_children.data(),
                   _nodes.num_children,
                   sizeof(int) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(leaves_begin.data(),
                   _nodes.leaves_begin,
                   sizeof(int) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(leaves_end.data(),
                   _nodes.leaves_end,
                   sizeof(int) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(x_barycenter.data(),
                   _nodes.x_barycenter,
                   sizeof(float) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(y_barycenter.data(),
                   _nodes.y_barycenter,
                   sizeof(float) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(z_barycenter.data(),
                   _nodes.z_barycenter,
                   sizeof(float) * _max_num_nodes,
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < min(100, _max_num_nodes); ++i)
        {
            printf("%2d: [", i);
            for (int j = 0; j < num_children[i]; ++j) {
                printf(" %3d", children[j * _max_num_nodes + i]);
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

    Nodes _nodes;
};

#endif

