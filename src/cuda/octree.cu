#include "cuda/octree.cuh"

#include "cuda/soa_vec3.cuh"
#include "cuda/soa_btree_nodes.cuh"
#include "cuda/soa_octree_nodes.cuh"
#include "cuda/btree.cuh"
#include "cuda/points.cuh"
#include "cuda/utils.cuh"

#include <iostream>
#include <cmath>

__device__ __forceinline__ int traverse(const int *child,
                                        const int *edge_delta,
                                        int start_node)
{
    int bin_node = start_node;

    do {
        bin_node = child[bin_node];
    } while (!edge_delta[bin_node]);

    return bin_node;
}

template<typename T>
__global__ void build_octree(const SoABtreeNodes btree_nodes,
                             SoAOctreeNodes octree_nodes,
                             T *octree_nodes_size,
                             const int *btree_octree_map,
                             const int *btree_num_leaves,
                             float domain_size,
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

    int node_level = btree_nodes.lcp(idx) / 3;
    // Computing the side length of the cube spanned by the octree node
    octree_nodes_size[parent] = domain_size / (1 << node_level);

    int node_leaves_begin = btree_nodes.leaves_begin(idx);
    int node_leaves_end = btree_nodes.leaves_end(idx);

    octree_nodes.leaves_begin(parent) = node_leaves_begin;
    octree_nodes.leaves_end(parent) = node_leaves_end;

    octree_nodes.depth(parent) = btree_nodes.depth(idx);

    if (node_leaves_begin == node_leaves_end) {
        // Leaf octree node
        return;
    }

    int first_bin_child = traverse(btree_nodes.left(),
                                   btree_nodes.edge_delta(),
                                   idx);

    int last_bin_child = traverse(btree_nodes.right(),
                                  btree_nodes.edge_delta(),
                                  idx);

    int first_child = btree_octree_map[first_bin_child];
    int last_child = btree_octree_map[last_bin_child];

    octree_nodes.first_child(parent) = first_child;
    octree_nodes.num_children(parent) = last_child - first_child + 1;
}

__global__ void
compute_octree_nodes_points_range(const int *leaf_first_code_idx,
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
    int codes_end = leaf_first_code_idx[leaves_end + 1];

    points_begin[idx] = codes_first_point_idx[codes_begin];
    points_end[idx] = codes_first_point_idx[codes_end] - 1;
}

template<typename T> __global__
void compute_octree_leaves_weighted_pos(const SoAOctreeNodes nodes,
                                        const SoAVec3<T> points_pos,
                                        const T *points_mass,
                                        const int *points_begin,
                                        const int *points_end,
                                        SoAVec3<T> barycenters,
                                        T *mass,
                                        int *num_nodes)
{
    int idx = blockIdx.x;
    if (idx >= *num_nodes) {
        return;
    }

    int num_children = nodes.num_children(idx);
    if (num_children > 0) {
        return;
    }

    int begin = points_begin[idx];
    int end = points_end[idx];

    int num_points = end - begin + 1;

    T weighted_x = 0.0;
    T weighted_y = 0.0;
    T weighted_z = 0.0;
    T tot_mass = 0.0;

    for (int i = 0; i + threadIdx.x < num_points; i += 32) {
        T m = points_mass[begin + i + threadIdx.x];

        T x = points_pos.x(begin + i + threadIdx.x) * m;
        T y = points_pos.y(begin + i + threadIdx.x) * m;
        T z = points_pos.z(begin + i + threadIdx.x) * m;

        // Warp scan
        #pragma unroll 5
        for (int delta = 16; delta > 0; delta >>= 1) {
            x += __shfl_down_sync(0xffffffff, x, delta);
            y += __shfl_down_sync(0xffffffff, y, delta);
            z += __shfl_down_sync(0xffffffff, z, delta);
            m += __shfl_down_sync(0xffffffff, m, delta);
        }

        weighted_x += x;
        weighted_y += y;
        weighted_z += z;
        tot_mass += m;
    }

    if (threadIdx.x == 0) {
        barycenters.x(idx) = weighted_x;
        barycenters.y(idx) = weighted_y;
        barycenters.z(idx) = weighted_z;
        mass[idx] = tot_mass;
    }
}

template<typename T>
__global__ void compute_octree_nodes_weighted_pos(const SoAOctreeNodes nodes,
                                                  SoAVec3<T> barycenters,
                                                  T *mass,
                                                  int *num_nodes,
                                                  int curr_depth)
{
    int node_idx = (blockDim.x * blockIdx.x + threadIdx.x) / 8;
    if (node_idx >= *num_nodes) {
        return;
    }

    int num_children = nodes.num_children(node_idx);
    int depth = nodes.depth(node_idx);
    if (num_children == 0 || depth != curr_depth) {
        return;
    }

    int first_child = nodes.first_child(node_idx);

    T m = 0.0;
    T wx = 0.0;
    T wy = 0.0;
    T wz = 0.0;

    int lane_idx = threadIdx.x % 8;
    if (lane_idx < num_children) {
        int children = first_child + lane_idx;
        m = mass[children];
        wx = barycenters.x(children);
        wy = barycenters.y(children);
        wz = barycenters.z(children);
    }

    #pragma unroll 3
    for (int delta = 4; delta > 0; delta >>= 1) {
        m += __shfl_down_sync(0xffffffff, m, delta, 8);
        wx += __shfl_down_sync(0xffffffff, wx, delta, 8);
        wy += __shfl_down_sync(0xffffffff, wy, delta, 8);
        wz += __shfl_down_sync(0xffffffff, wz, delta, 8);
    }

    if (lane_idx == 0) {
        mass[node_idx] = m;
        barycenters.x(node_idx) = wx;
        barycenters.y(node_idx) = wy;
        barycenters.z(node_idx) = wz;
    }
}

template<typename T> Octree<T>::Octree(int max_num_leaves,
                                       float domain_size) :
    domain_size_(domain_size),
    gl_buffers_(false)
{
    init_(max_num_leaves);

    barycenters_.alloc(max_num_nodes_);

    cudaMalloc(&nodes_size_, max_num_nodes_ * sizeof(T));
}

template<typename T> Octree<T>::Octree(int max_num_leaves,
                                       float domain_size,
                                       T *d_barycenters_x,
                                       T *d_barycenters_y,
                                       T *d_barycenters_z,
                                       T *d_nodes_size) :
    domain_size_(domain_size),
    gl_buffers_(true)
{
    init_(max_num_leaves);

    barycenters_.x() = d_barycenters_x;
    barycenters_.y() = d_barycenters_y;
    barycenters_.z() = d_barycenters_z;

    nodes_size_ = d_nodes_size;
}

template<typename T> void Octree<T>::init_(int max_num_leaves)
{
    max_num_nodes_ = min(
        2 * max_num_leaves,
        geometric_sum(8, ceil(log2(max_num_leaves) / 3.0) + 1.0));

    cudaMalloc(&num_nodes_, sizeof(int));
    cudaMalloc(&points_begin_, max_num_nodes_ * sizeof(int));
    cudaMalloc(&points_end_, max_num_nodes_ * sizeof(int));

    nodes_.alloc(max_num_nodes_);
    cudaMalloc(&nodes_mass_, max_num_nodes_ * sizeof(T));
}

template<typename T> void Octree<T>::build(const Btree &btree)
{
    build_octree<<<btree.get_max_num_nodes() / THREADS_PER_BLOCK +
                   (btree.get_max_num_nodes() % THREADS_PER_BLOCK > 0),
                   THREADS_PER_BLOCK>>>(btree.get_d_nodes(),
                                        nodes_,
                                        nodes_size_,
                                        btree.get_d_octree_map_ptr(),
                                        btree.get_d_num_leaves_ptr(),
                                        domain_size_,
                                        num_nodes_);
}

template<typename T>
void Octree<T>::compute_nodes_points_range(const int *d_leaf_first_code_idx,
                                           const int *d_codes_first_point_idx)
{
    compute_octree_nodes_points_range<<<
       max_num_nodes_ / THREADS_PER_BLOCK +
       (max_num_nodes_ % THREADS_PER_BLOCK > 0),
       THREADS_PER_BLOCK>>>(d_leaf_first_code_idx,
                            d_codes_first_point_idx,
                            nodes_.leaves_begin_,
                            nodes_.leaves_end_,
                            points_begin_,
                            points_end_,
                            num_nodes_);
}

template<typename T>
__global__ void compute_octree_nodes_barycenter(SoAVec3<T> barycenters,
                                                T *mass,
                                                int *num_nodes)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= *num_nodes) {
        return;
    }

    T m = mass[idx];
    T wx = barycenters.x(idx);
    T wy = barycenters.y(idx);
    T wz = barycenters.z(idx);

    barycenters.x(idx) = wx / m;
    barycenters.y(idx) = wy / m;
    barycenters.z(idx) = wz / m;
}

template<typename T>
void Octree<T>::compute_nodes_barycenter(const Points<T> &points)
{
    compute_octree_leaves_weighted_pos<<<
        max_num_nodes_, 32>>>(nodes_,
                              points.get_d_pos(),
                              points.get_d_mass(),
                              points_begin_,
                              points_end_,
                              barycenters_,
                              nodes_mass_,
                              num_nodes_);

    int max_depth;
    cudaMemcpy(&max_depth,
               nodes_.depth_ + get_num_nodes() - 1,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int d = max_depth - 1; d >= 0; --d) {
        compute_octree_nodes_weighted_pos<<<
            (max_num_nodes_ * 8 - 1) / MAX_THREADS_PER_BLOCK + 1,
            MAX_THREADS_PER_BLOCK>>>(nodes_,
                                     barycenters_,
                                     nodes_mass_,
                                     num_nodes_,
                                     d);
    }

    compute_octree_nodes_barycenter<<<
        (max_num_nodes_ - 1) / MAX_THREADS_PER_BLOCK + 1,
        MAX_THREADS_PER_BLOCK>>>(barycenters_,
                                 nodes_mass_,
                                 num_nodes_);
}

template<typename T> Octree<T>::~Octree()
{
    cudaFree(num_nodes_);
    cudaFree(points_begin_);
    cudaFree(points_end_);

    nodes_.free();

    if (!gl_buffers_) {
        barycenters_.free();
        cudaFree(nodes_size_);
    }

    cudaFree(nodes_mass_);
}

// Explicit templates instantiation

template class Octree<float>;
template class Octree<double>;
