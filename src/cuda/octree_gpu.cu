#include "octree_gpu.cuh"

#include "btree_gpu.cuh"
#include "utils_gpu.cuh"

#include <iostream>

// At most 15 nodes can be visited by traversing
// 3 levels of any subtree of the binary radix tree
#define _BUILD_STACK_SIZE 16

__global__ void _build_octree(struct Btree::Nodes btree_internal,
                              struct Octree::Nodes octree_internal,
                              const int *btree_num_leaves,
                              int *octree_num_internal,
                              int octree_max_num_internal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int btree_num_internal = *btree_num_leaves - 1;

    int is_octree_node = btree_internal.edge_delta[idx];

    if (idx >= btree_num_internal || !is_octree_node) {
        return;
    }

    if (idx == 0) {
        *octree_num_internal =
            btree_internal.octree_map[btree_num_internal - 1];
    }

    int parent = btree_internal.octree_map[idx];

    // Resetting number of children
    octree_internal.num_children[parent] = 0;

    octree_internal.leaves_begin[parent] =
        btree_internal.leaves_begin[idx];
    octree_internal.leaves_end[parent] =
        btree_internal.leaves_end[idx];

    // Stack used to traverse at most 3 levels
    int stack[_BUILD_STACK_SIZE];
    // Points to last+1 element on the stack
    int end = 0;
    stack[end++] = idx;

    int bin_node = btree_internal.left[idx];
    while (end >= 0) {
        is_octree_node = btree_internal.edge_delta[bin_node];
        int is_leaf = bin_node >= btree_num_internal;

        if (bin_node != idx && (is_leaf || is_octree_node)) {

            // Removing offset for leaf pointers
            int child = is_leaf ? bin_node - btree_num_internal :
                                  btree_internal.octree_map[bin_node];

            int num_children = octree_internal.num_children[parent];

            octree_internal.children[
                num_children * octree_max_num_internal + parent] =
                // TODO:: doesn't show noticeable difference from
                //_children[parent * 8 + num_children] =

                // Pointers to leaf nodes are offset by the
                // maximum number of internal nodes
                child + is_leaf * octree_max_num_internal;

            octree_internal.num_children[parent] = ++num_children;

            end--;
            if (end >= 0) {
                bin_node = btree_internal.right[stack[end]];
            }

        } else {
            stack[end++] = bin_node;
            bin_node = btree_internal.left[bin_node];
        }
    }
}

__global__ void _compute_octree_nodes_barycenter(const Points *points,
                                                 const Points *scan_points,
                                                 const int *leaf_first_code_idx,
                                                 const int *scan_codes_occurrences,
                                                 struct Octree::Nodes internal,
                                                 const int *num_internal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_internal) return;

    int leaves_begin = internal.leaves_begin[idx];
    int leaves_end = internal.leaves_end[idx];

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

    internal.x_barycenter[idx] = x_barycenter;
    internal.y_barycenter[idx] = y_barycenter;
    internal.z_barycenter[idx] = z_barycenter;

    // When dealing with different masses, multiply the x array 
    // with the mass array and then compute the prefix sum
    // The prefix sum of the mass array needs to be computed as well
}

Octree::Octree(int max_num_internal) : _max_num_internal(max_num_internal)
{
    cudaMalloc(&_num_internal, sizeof(int));
    // Allocating device memory to store octree nodes
    cudaMalloc(&_internal.children, max_num_internal * 8 * sizeof(int));
    cudaMalloc(&_internal.num_children, max_num_internal * sizeof(int));
    cudaMalloc(&_internal.leaves_begin, max_num_internal * sizeof(int));
    cudaMalloc(&_internal.leaves_end, max_num_internal * sizeof(int));
    cudaMalloc(&_internal.x_barycenter, max_num_internal * sizeof(float));
    cudaMalloc(&_internal.y_barycenter, max_num_internal * sizeof(float));
    cudaMalloc(&_internal.z_barycenter, max_num_internal * sizeof(float));
}

void Octree::build(const Btree &btree)
{
    _build_octree<<<btree.get_max_num_internal() / THREADS_PER_BLOCK +
                    (btree.get_max_num_internal() % THREADS_PER_BLOCK > 0),
                    THREADS_PER_BLOCK>>>(btree.get_d_internal(),
                                         _internal,
                                         btree.get_d_num_leaves_ptr(),
                                         _num_internal,
                                         _max_num_internal);
}

void Octree::compute_nodes_barycenter(const Points *points,
                                      const Points *scan_points,
                                      const int *leaf_first_code,
                                      const int *scan_codes_occurrences)
{
    _compute_octree_nodes_barycenter<<<
        _max_num_internal / THREADS_PER_BLOCK +
        (_max_num_internal % THREADS_PER_BLOCK > 0),
        THREADS_PER_BLOCK>>>(points,
                             scan_points,
                             leaf_first_code,
                             scan_codes_occurrences,
                             _internal,
                             _num_internal);
}

Octree::~Octree()
{
    cudaFree(_num_internal);

    cudaFree(_internal.children);
    cudaFree(_internal.num_children);
    cudaFree(_internal.leaves_begin);
    cudaFree(_internal.leaves_end);
    cudaFree(_internal.x_barycenter);
    cudaFree(_internal.y_barycenter);
    cudaFree(_internal.z_barycenter);
}
