#include "octree_gpu.cuh"

#include "btree_gpu.cuh"

#define _BUILD_STACK_SIZE 15 // 2^1 + 2^2 + 2^3 + 1

__constant__ Octree _d_octree;

__global__ void _build_octree()
{
    // For each internal node and leaf node
    // The parent of the current node is the closest ancestor k
    // such that _edge_delta[k] > 0, in this case take the 
    // k-th value of the scan of _edge_delta
    // WARNING: requires storing btree nodes as pointers to parent
    // building octree may require synchronization

    // otherwise as follows

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!d_btree.is_octree_node(idx)) return;

    // Stack used to traverse at most 3 levels
    int stack[_BUILD_STACK_SIZE];
    // Points to first and last+1 element on the stack
    int start = 0;
    int end = 0;

    stack[end++] = idx;

    int parent = d_btree.get_octree_node(idx);

    do {
        int bin_node = stack[start++];
        int is_leaf = d_btree.is_leaf(bin_node);

        if (bin_node != idx &&
            (is_leaf || d_btree.is_octree_node(bin_node))) {

            int child = is_leaf ? d_btree.get_leaf(bin_node) :
                                  d_btree.get_octree_node(bin_node);

            _d_octree.add_child(parent, child, is_leaf);
        } else {
            stack[end++] = d_btree.get_left(bin_node);
            stack[end++] = d_btree.get_right(bin_node);
        }

    } while (start != end);
}

Octree::Octree(int max_depth) : _max_depth(max_depth)
{
    int num_nodes_at_depth = 1;
    int num_internal_nodes = 1;
    // Nodes at max_depth are leaves
    for (int d = 0; d < max_depth - 1; ++d) {
        num_nodes_at_depth << 3;
        num_internal_nodes += num_nodes_at_depth;
    }

    _max_num_internal = num_nodes - num_nodes_at_depth;

    // Allocating device memory to store octree nodes
    cudaMalloc(&_children, num_internal_nodes * 8 * sizeof(int));
    cudaMalloc(&_num_children, num_internal_nodes * sizeof(int));

    // Allocating object copy in device constant memory
    cudaMemcpyToSymbol(_d_octree,
                       this,
                       sizeof(Octree),
                       cudaMemcpyHostToDevice);
}

void Octree::build()
{
    // launch _build_octree() kernel
}

Octree::~Octree()
{
    cudaFree(_children);
    cudaFree(_num_children);
}
