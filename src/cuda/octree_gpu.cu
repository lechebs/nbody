#include "cuda/octree_gpu.cuh"

#include "cuda/btree_gpu.cuh"
#include "cuda/points_gpu.cuh"
#include "cuda/utils_gpu.cuh"

#include <iostream>

// At most 15 nodes can be visited by traversing
// 3 levels of any subtree of the binary radix tree
#define _BUILD_STACK_SIZE 16

void SoAOctreeNodes::alloc(int num_nodes)
{
    cudaMalloc(&first_child, num_nodes * sizeof(int));
    cudaMalloc(&num_children, num_nodes * sizeof(int));
    cudaMalloc(&leaves_begin, num_nodes * sizeof(int));
    cudaMalloc(&leaves_end, num_nodes * sizeof(int));
}

void SoAOctreeNodes::free()
{
    cudaFree(first_child);
    cudaFree(num_children);
    cudaFree(leaves_begin);
    cudaFree(leaves_end);
}

__device__ __forceinline__ int _traverse(const int *child,
                                         const int *edge_delta,
                                         int start_node)
{
    int bin_node = start_node;

    do {
        bin_node = child[bin_node];
    } while (!edge_delta[bin_node]);

    return bin_node;
}

__global__ void _build_octree(SoABtreeNodes btree_nodes,
                              SoAOctreeNodes octree_nodes,
                              const int *btree_octree_map,
                              const int *btree_num_leaves,
                              int *octree_num_nodes)
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

    int first_bin_child = _traverse(btree_nodes.left,
                                    btree_nodes.edge_delta,
                                    idx);

    int last_bin_child = _traverse(btree_nodes.right,
                                   btree_nodes.edge_delta,
                                   idx);

    int first_child = btree_octree_map[first_bin_child];
    int last_child = btree_octree_map[last_bin_child];

    octree_nodes.first_child[parent] = first_child;
    octree_nodes.num_children[parent] = last_child - first_child + 1;
}

template<typename T> __global__ void
_compute_octree_nodes_barycenter(const SoAVec3<T> points,
                                 const SoAVec3<T> scan_points,
                                 const int *leaf_first_code_idx,
                                 const int *code_first_point_idx,
                                 SoAOctreeNodes nodes,
                                 SoAVec3<T> barycenters,
                                 const int *num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_nodes) return;

    int leaves_begin = nodes.leaves_begin[idx];
    int leaves_end = nodes.leaves_end[idx];

    int codes_begin = leaf_first_code_idx[leaves_begin];
    int codes_end = leaf_first_code_idx[leaves_end + 1] - 1;

    int points_begin = code_first_point_idx[codes_begin];
    int points_end = code_first_point_idx[codes_end];

    T x_barycenter = scan_points.x[points_end] -
                     scan_points.x[points_begin] +
                     points.x[points_end];

    T y_barycenter = scan_points.y[points_end] -
                     scan_points.y[points_begin] +
                     points.y[points_end];

    T z_barycenter = scan_points.z[points_end] -
                     scan_points.z[points_begin] +
                     points.z[points_end];

    // Works when all points have unit mass
    T mass_sum = points_end - points_begin + 1;

    x_barycenter /= mass_sum;
    y_barycenter /= mass_sum;
    z_barycenter /= mass_sum;

    barycenters.x[idx] = x_barycenter;
    barycenters.y[idx] = y_barycenter;
    barycenters.z[idx] = z_barycenter;

    // When dealing with different masses, multiply the x array 
    // with the mass array and then compute the prefix sum
    // The prefix sum of the mass array needs to be computed as well
}

template<typename T> Octree<T>::Octree(int max_num_nodes) :
    _max_num_nodes(max_num_nodes)
{
    cudaMalloc(&_num_nodes, sizeof(int));

    _nodes.alloc(max_num_nodes);
    _barycenters.alloc(max_num_nodes);
}

template<typename T> void Octree<T>::build(const Btree &btree)
{
    _build_octree<<<btree.get_max_num_nodes() / THREADS_PER_BLOCK +
                    (btree.get_max_num_nodes() % THREADS_PER_BLOCK > 0),
                    THREADS_PER_BLOCK>>>(btree.get_d_nodes(),
                                         _nodes,
                                         btree.get_d_octree_map_ptr(),
                                         btree.get_d_num_leaves_ptr(),
                                         _num_nodes);
}

template<typename T>
void Octree<T>::compute_nodes_barycenter(Points<T> &points,
                                         const int *leaf_first_code_idx)
{
    _compute_octree_nodes_barycenter<<<
        _max_num_nodes / THREADS_PER_BLOCK +
        (_max_num_nodes % THREADS_PER_BLOCK > 0),
        THREADS_PER_BLOCK>>>(points.get_d_pos(),
                             points.get_d_scan_pos(),
                             leaf_first_code_idx,
                             points.get_d_codes_first_point_idx_ptr(),
                             _nodes,
                             _barycenters,
                             _num_nodes);
}

template<typename T> Octree<T>::~Octree()
{
    cudaFree(_num_nodes);

    _nodes.free();
    _barycenters.free();
}

// Explicit templates instantiation

template class Octree<float>;
template class Octree<double>;
