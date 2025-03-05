#include "octree_gpu.cuh"

#include "btree_gpu.cuh"
#include "utils_gpu.cuh"

#include <iostream>

// At most 15 nodes can be visited by traversing
// 3 levels of any subtree of the binary radix tree
#define _BUILD_STACK_SIZE 16

__global__ void _build_octree(struct Btree::Nodes btree_nodes,
                              struct Octree::Nodes octree_nodes,
                              const int *btree_octree_map,
                              const int *btree_num_leaves,
                              int *octree_num_nodes,
                              int octree_max_num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int btree_num_nodes = 2 * *btree_num_leaves - 1;

    if (idx >= btree_num_nodes || !btree_nodes.edge_delta[idx]) {
        return;
    }

    if (idx == 0) {
        *octree_num_nodes =
            btree_octree_map[btree_num_nodes - 1] +
            btree_nodes.edge_delta[btree_num_nodes - 1];
    }

    int parent = btree_octree_map[idx];

    // Resetting number of children
    octree_nodes.num_children[parent] = 0;

    int node_leaves_begin = btree_nodes.leaves_begin[idx];
    int node_leaves_end = btree_nodes.leaves_end[idx];

    octree_nodes.leaves_begin[parent] = node_leaves_begin;
    octree_nodes.leaves_end[parent] = node_leaves_end;

    if (node_leaves_begin == node_leaves_end) {
        // Leaf octree node
        return;
    }

    // Stack used to traverse at most 3 levels
    int stack[_BUILD_STACK_SIZE];
    // Points to last+1 element on the stack
    int end = 0;
    stack[end++] = idx;

    int bin_node = btree_nodes.left[idx];
    while (end >= 0) {
        int is_octree_node = btree_nodes.edge_delta[bin_node];

        if (bin_node != idx && is_octree_node) {

            int child = btree_octree_map[bin_node];

            int num_children = octree_nodes.num_children[parent];

            octree_nodes.children[
                num_children * octree_max_num_nodes + parent] = child;
                // TODO:: doesn't show noticeable difference from
                //_children[parent * 8 + num_children] =

            octree_nodes.num_children[parent] = ++num_children;

            end--;
            if (end >= 0) {
                bin_node = btree_nodes.right[stack[end]];
            }

        } else {
            stack[end++] = bin_node;
            bin_node = btree_nodes.left[bin_node];
        }
    }
}

__global__ void _compute_octree_nodes_barycenter(const Points *points,
                                                 const Points *scan_points,
                                                 const int *leaf_first_code_idx,
                                                 const int *scan_codes_occurrences,
                                                 struct Octree::Nodes nodes,
                                                 const int *num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_nodes) return;

    int leaves_begin = nodes.leaves_begin[idx];
    int leaves_end = nodes.leaves_end[idx];

    int codes_begin = leaf_first_code_idx[leaves_begin];
    int codes_end = leaf_first_code_idx[leaves_end + 1] - 1;

    int points_begin = scan_codes_occurrences[codes_begin];
    int points_end = scan_codes_occurrences[codes_end];

    float x_barycenter = scan_points->get_x(points_end) -
                         scan_points->get_x(points_begin) +
                         points->get_x(points_end);

    float y_barycenter = scan_points->get_y(points_end) -
                         scan_points->get_y(points_begin) +
                         points->get_y(points_end);

    float z_barycenter = scan_points->get_z(points_end) -
                         scan_points->get_z(points_begin) +
                         points->get_z(points_end);

    // Works when all points have unit mass
    float mass_sum = points_end - points_begin + 1;

    x_barycenter /= mass_sum;
    y_barycenter /= mass_sum;
    z_barycenter /= mass_sum;

    nodes.x_barycenter[idx] = x_barycenter;
    nodes.y_barycenter[idx] = y_barycenter;
    nodes.z_barycenter[idx] = z_barycenter;

    // When dealing with different masses, multiply the x array 
    // with the mass array and then compute the prefix sum
    // The prefix sum of the mass array needs to be computed as well
}

Octree::Octree(int max_num_nodes) : _max_num_nodes(max_num_nodes)
{
    cudaMalloc(&_num_nodes, sizeof(int));
    // Allocating device memory to store octree nodes
    cudaMalloc(&_nodes.children, max_num_nodes * 8 * sizeof(int));
    cudaMalloc(&_nodes.num_children, max_num_nodes * sizeof(int));
    cudaMalloc(&_nodes.leaves_begin, max_num_nodes * sizeof(int));
    cudaMalloc(&_nodes.leaves_end, max_num_nodes * sizeof(int));
    cudaMalloc(&_nodes.x_barycenter, max_num_nodes * sizeof(float));
    cudaMalloc(&_nodes.y_barycenter, max_num_nodes * sizeof(float));
    cudaMalloc(&_nodes.z_barycenter, max_num_nodes * sizeof(float));
}

void Octree::build(const Btree &btree)
{
    _build_octree<<<btree.get_max_num_nodes() / THREADS_PER_BLOCK +
                    (btree.get_max_num_nodes() % THREADS_PER_BLOCK > 0),
                    THREADS_PER_BLOCK>>>(btree.get_d_nodes(),
                                         _nodes,
                                         btree.get_d_octree_map(),
                                         btree.get_d_num_leaves_ptr(),
                                         _num_nodes,
                                         _max_num_nodes);
}

void Octree::compute_nodes_barycenter(const Points *points,
                                      const Points *scan_points,
                                      const int *leaf_first_code,
                                      const int *scan_codes_occurrences)
{
    _compute_octree_nodes_barycenter<<<
        _max_num_nodes / THREADS_PER_BLOCK +
        (_max_num_nodes % THREADS_PER_BLOCK > 0),
        THREADS_PER_BLOCK>>>(points,
                             scan_points,
                             leaf_first_code,
                             scan_codes_occurrences,
                             _nodes,
                             _num_nodes);
}

Octree::~Octree()
{
    cudaFree(_num_nodes);

    cudaFree(_nodes.children);
    cudaFree(_nodes.num_children);
    cudaFree(_nodes.leaves_begin);
    cudaFree(_nodes.leaves_end);
    cudaFree(_nodes.x_barycenter);
    cudaFree(_nodes.y_barycenter);
    cudaFree(_nodes.z_barycenter);
}
