#include "cuda/octree.cuh"

#include "cuda/soa_btree_nodes.cuh"
#include "cuda/soa_octree_nodes.cuh"
#include "cuda/btree.cuh"
#include "cuda/points.cuh"
#include "cuda/utils.cuh"

#include <iostream>
#include <cmath>

// For 3 levels DFS traversal
#define _BUILD_STACK_SIZE 4

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

__global__ void _build_octree(const SoABtreeNodes btree_nodes,
                              SoAOctreeNodes octree_nodes,
                              const int *btree_octree_map,
                              const int *btree_num_leaves,
                              int *octree_num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int btree_num_nodes = 2 * *btree_num_leaves - 1;

    if (idx >= btree_num_nodes || !btree_nodes.edge_delta(idx)) {
        return;
    }

    if (idx == 0) {
        *octree_num_nodes =
            btree_octree_map[btree_num_nodes - 1] +
            btree_nodes.edge_delta(btree_num_nodes - 1);
    }

    int parent = btree_octree_map[idx];

    // Resetting number of children
    octree_nodes.num_children(parent) = 0;

    int node_leaves_begin = btree_nodes.leaves_begin(idx);
    int node_leaves_end = btree_nodes.leaves_end(idx);

    octree_nodes.leaves_begin(parent) = node_leaves_begin;
    octree_nodes.leaves_end(parent) = node_leaves_end;

    if (node_leaves_begin == node_leaves_end) {
        // Leaf octree node
        return;
    }

    int first_bin_child = _traverse(btree_nodes.left(),
                                    btree_nodes.edge_delta(),
                                    idx);

    int last_bin_child = _traverse(btree_nodes.right(),
                                   btree_nodes.edge_delta(),
                                   idx);

    int first_child = btree_octree_map[first_bin_child];
    int last_child = btree_octree_map[last_bin_child];

    octree_nodes.first_child(parent) = first_child;
    octree_nodes.num_children(parent) = last_child - first_child + 1;
}

__global__ void
_compute_octree_nodes_points_range(const int *leaf_first_code_idx,
                                   const int *codes_first_point_idx,
                                   const int *leaves_begin_,
                                   const int *leaves_end_,
                                   int *points_begin,
                                   int *points_end,
                                   const int *num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_nodes) {
        return;
    }

    int leaves_begin = leaves_begin_[idx];
    int leaves_end = leaves_end_[idx];

    int codes_begin = leaf_first_code_idx[leaves_begin];
    int codes_end = leaf_first_code_idx[leaves_end + 1] - 1;

    points_begin[idx] = codes_first_point_idx[codes_begin];
    points_end[idx] = codes_first_point_idx[codes_end + 1] - 1;
}

template<typename T> __global__ void
_compute_octree_nodes_barycenter(const SoAVec3<T> points,
                                 const SoAVec3<T> scan_points,
                                 const int *points_begin,
                                 const int *points_end,
                                 SoAVec3<T> barycenters,
                                 const int *num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_nodes) {
        return;
    }

    int begin = points_begin[idx];
    int end = points_end[idx];

    T x_barycenter = scan_points.x(end) -
                     scan_points.x(begin) +
                     points.x(end);

    T y_barycenter = scan_points.y(end) -
                     scan_points.y(begin) +
                     points.y(end);

    T z_barycenter = scan_points.z(end) -
                     scan_points.z(begin) +
                     points.z(end);

    // Works when all points have unit mass
    T mass_sum = end - begin + 1;

    x_barycenter /= mass_sum;
    y_barycenter /= mass_sum;
    z_barycenter /= mass_sum;

    barycenters.x(idx) = x_barycenter;
    barycenters.y(idx) = y_barycenter;
    barycenters.z(idx) = z_barycenter;

    // When dealing with different masses, multiply the x array 
    // with the mass array and then compute the prefix sum
    // The prefix sum of the mass array needs to be computed as well
}

template<typename T> Octree<T>::Octree(int max_num_leaves) :
    _gl_buffers(false)
{
    _max_num_nodes = min(
        2 * max_num_leaves,
        geometric_sum(8, ceil(log2(max_num_leaves) / 3.0) + 1.0));

    cudaMalloc(&_num_nodes, sizeof(int));

    cudaMalloc(&_points_begin, _max_num_nodes * sizeof(int));
    cudaMalloc(&_points_end, _max_num_nodes * sizeof(int));

    _nodes.alloc(_max_num_nodes);
    _barycenters.alloc(_max_num_nodes);
}

template<typename T> Octree<T>::Octree(int max_num_leaves,
                                       int *d_points_begin,
                                       int *d_points_end) :
    _gl_buffers(true)
{
    _max_num_nodes = min(
        2 * max_num_leaves,
        geometric_sum(8, ceil(log2(max_num_leaves) / 3.0) + 1.0));

    cudaMalloc(&_num_nodes, sizeof(int));

    _points_begin = d_points_begin;
    _points_end = d_points_end;

    _nodes.alloc(_max_num_nodes);
    _barycenters.alloc(_max_num_nodes);
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
void Octree<T>::compute_nodes_points_range(const int *d_leaf_first_code_idx,
                                           const int *d_codes_first_point_idx)
{
    _compute_octree_nodes_points_range<<<
        _max_num_nodes / THREADS_PER_BLOCK +
        (_max_num_nodes % THREADS_PER_BLOCK > 0),
        THREADS_PER_BLOCK>>>(d_leaf_first_code_idx,
                             d_codes_first_point_idx,
                             _nodes._leaves_begin,
                             _nodes._leaves_end,
                             _points_begin,
                             _points_end,
                             _num_nodes);
}

template<typename T>
void Octree<T>::compute_nodes_barycenter(const Points<T> &points)
{
    _compute_octree_nodes_barycenter<<<
        _max_num_nodes / THREADS_PER_BLOCK +
        (_max_num_nodes % THREADS_PER_BLOCK > 0),
        THREADS_PER_BLOCK>>>(points.get_d_pos(),
                             points.get_d_scan_pos(),
                             _points_begin,
                             _points_end,
                             _barycenters,
                             _num_nodes);
}

template<typename T> Octree<T>::~Octree()
{
    cudaFree(_num_nodes);

    if (!_gl_buffers) {
        cudaFree(_points_begin);
        cudaFree(_points_end);
    }

    _nodes.free();
    _barycenters.free();
}

// Explicit templates instantiation

template class Octree<float>;
template class Octree<double>;
