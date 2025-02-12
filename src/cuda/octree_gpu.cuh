#ifndef OCTREE_GPU_CUH
#define OCTREE_GPU_CUH

#include "btree_gpu.cuh"

#include <iostream>

// SoA to store the octree
class Octree
{
public:
    // max_depth ~= btree_height / 3
    Octree(int max_depth);

    void build(Btree &btree);

    ~Octree();

    __device__ __forceinline__
    bool is_leaf(int node)
    {
        return node >= _max_num_internal;
    }

    __device__ __forceinline__
    void set_num_children(int idx, int num_children)
    {
        _num_children[idx] = num_children;
    }

    __device__ __forceinline__
    void add_child(int parent, int child, bool is_leaf)
    {
        int num_children = _num_children[parent];

        _children[num_children * _max_num_internal + parent] =
        // TODO:: doesn't show noticeable difference
        //_children[parent * 8 + num_children] =
            // Pointers to leaf nodes are offset by the
            // maximum number of internal nodes
            child + is_leaf * _max_num_internal;

        _num_children[parent] = num_children + 1;
    }

private:
    int _max_depth;
    int _max_num_internal;

    // Array to store pointers (indices) to the children of each node,
    // each groups of 8 siblings is contiguous in memory
    int *_children;
    // Array to store the number of children of each internal node
    int *_num_children;

    Octree *_d_this;
};

#endif

