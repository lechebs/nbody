#include "octree_gpu.cuh"

#include "btree_gpu.cuh"

#include <iostream>

// At most 15 nodes can be visited by traversing
// 3 levels of any subtree of the binary radix tree
#define _BUILD_STACK_SIZE 16

__global__ void _build_octree(Btree &btree, Octree &octree)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= btree.get_num_internal() ||
        !btree.is_octree_node(idx)) return;

    // TODO: can we avoid this?
    if (idx == 0) {
        octree.set_num_internal(btree.get_num_octree_nodes());
    }

    int parent = btree.get_octree_node(idx);

    octree.set_num_children(parent, 0);
    octree.set_leaves_range(parent,
                            btree.get_leaves_begin(idx),
                            btree.get_leaves_end(idx));

    // Stack used to traverse at most 3 levels
    int stack[_BUILD_STACK_SIZE];
    // Points to first and last+1 element on the stack
    int start = 0;
    int end = 0;
    stack[end++] = idx;

    do {
        int bin_node = stack[start++];
        int is_leaf = btree.is_leaf(bin_node);

        if (bin_node != idx &&
            (is_leaf || btree.is_octree_node(bin_node))) {

            int child = is_leaf ? btree.get_leaf(bin_node) :
                                  btree.get_octree_node(bin_node);

            octree.add_child(parent, child, is_leaf);

        } else {
            stack[end++] = btree.get_left(bin_node);
            stack[end++] = btree.get_right(bin_node);
        }

    } while (start != end);
}

Octree::Octree(int max_depth) : _max_depth(max_depth)
{
    _max_num_internal = 1;
    int num_nodes_at_depth = 1;
    // Nodes at max_depth are leaves
    for (int d = 0; d < max_depth - 1; ++d) {
        num_nodes_at_depth *= 8;
        _max_num_internal += num_nodes_at_depth;
    }

    // Allocating device memory to store octree nodes
    cudaMalloc(&_children, _max_num_internal * 8 * sizeof(int));
    cudaMalloc(&_num_children, _max_num_internal * sizeof(int));
    cudaMalloc(&_leaves_begin, _max_num_internal * sizeof(int));
    cudaMalloc(&_leaves_end, _max_num_internal * sizeof(int));

    // Allocating object copy in device memory
    cudaMalloc(&_d_this, sizeof(Octree));
    cudaMemcpy(_d_this, this, sizeof(Octree), cudaMemcpyHostToDevice);
}

void Octree::build(Btree &btree)
{
    _build_octree<<<btree.get_num_internal() / THREADS_PER_BLOCK +
                    (btree.get_num_internal() % THREADS_PER_BLOCK > 0),
                    THREADS_PER_BLOCK>>>(*btree.get_dev_ptr(), *_d_this);
}

Octree::~Octree()
{
    cudaFree(_children);
    cudaFree(_num_children);
}
