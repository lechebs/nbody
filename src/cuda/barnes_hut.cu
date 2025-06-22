#include "cuda/barnes_hut.cuh"

#include <cassert>

#include "cuda/soa_vec3.cuh"
#include "cuda/soa_octree_nodes.cuh"
#include "cuda/physics_common.cuh"
#include "cuda/octree.cuh"

#define WARP_SIZE 32

__constant__ int QUEUE_SIZE;
__constant__ int GROUP_SIZE;
__constant__ int NUM_WARPS;

// TODO: try removing inlining to reduce registers usage
__device__ __forceinline__ int warp_scan(int var, int lane_idx)
{
    int res = var;
    #pragma unroll
    for (int i = 0, delta = 1; i < 5; ++i, delta <<= 1) {
        int value = __shfl_up_sync(0xffffffff, res, delta);
        if (lane_idx >= delta) {
            res += value;
        }
    }

    return res - var;
}

template<typename T> __device__ __forceinline__
bool approx_crit(float size, T dist_sq, float theta)
{
    return size * size / dist_sq < theta * theta;
}

template<typename T> __device__ __forceinline__
T compute_group_to_node_min_dist(const T *x_group,
                                 const T *y_group,
                                 const T *z_group,
                                 T x_node,
                                 T y_node,
                                 T z_node)
{
    // WARNING: coordinates are within the unit cube
    T domain_size = PhysicsCommon<T>::get_domain_size();
    T min_dist_sq = domain_size * domain_size * (T) 3.0;

    #pragma unroll
    for (int i = 0; i < GROUP_SIZE; ++i) {
        T dist_sq = PhysicsCommon<T>::
                    compute_dist_sq(x_group[i], y_group[i], z_group[i],
                                    x_node, y_node, z_node);

        if (dist_sq <= min_dist_sq) {
            min_dist_sq = dist_sq;
        }
    }

    return min_dist_sq;
}

__device__ __forceinline__ void append_to_queue(const int *open_buff,
                                                const int *nchildren_buff,
                                                int *queue,
                                                int open_buff_size,
                                                int queue_size,
                                                int num_children)
{
    for (int j = 0; j + threadIdx.x < num_children; j += GROUP_SIZE) {

        // Divergence-less binary search

        int left = 0;
        int step = open_buff_size;
        do {
            step = (step + 1) >> 1;
            if (left + step < open_buff_size &&
                j + threadIdx.x >= nchildren_buff[left + step]) {
                left += step;
            }
        } while (step > 1);

        int queue_dst = queue_size + j + threadIdx.x;

        __syncwarp(__activemask());

        queue[queue_dst] = open_buff[left] +
                           j +
                           threadIdx.x -
                           nchildren_buff[left];
    }
}

template<typename T> __device__ __forceinline__
int evaluate_approx(const SoAVec3<T> bodies_pos,
                    const SoAVec3<T> nodes_barycenter,
                    const T *nodes_mass,
                    const int *bodies_begin,
                    const int *bodies_end,
                    T px, T py, T pz,
                    T &fx, T &fy, T &fz,
                    int *approx_buff,
                    T *x_buff,
                    T *y_buff,
                    T *z_buff,
                    T *m_buff,
                    int approx_buff_size,
                    const T softening_factor_sq)
{
    int start_buff_idx = max(0, approx_buff_size - GROUP_SIZE);

    if (threadIdx.x < approx_buff_size) {
        int node = approx_buff[start_buff_idx + threadIdx.x];
        m_buff[threadIdx.x] = nodes_mass[node];
        x_buff[threadIdx.x] = nodes_barycenter.x(node);
        y_buff[threadIdx.x] = nodes_barycenter.y(node);
        z_buff[threadIdx.x] = nodes_barycenter.z(node);
    }

    __syncthreads();

    int chunk_size = approx_buff_size - start_buff_idx;
    // #pragma unroll
    // TODO: try unrolling
    for (int k = 0; k < chunk_size; ++k) {
        PhysicsCommon<T>::
        accumulate_pairwise_force(px,
                                  py,
                                  pz,
                                  x_buff[k],
                                  y_buff[k],
                                  z_buff[k],
                                  m_buff[k],
                                  fx,
                                  fy,
                                  fz,
                                  softening_factor_sq);
    }

    return start_buff_idx;
}

template<typename T> __device__ __forceinline__
void evaluate_leaf(const SoAVec3<T> bodies_pos,
                   const T *bodies_mass,
                   const int *bodies_begin,
                   const int *bodies_end,
                   int leaf,
                   T *x_buff,
                   T *y_buff,
                   T *z_buff,
                   T *m_buff,
                   T px,
                   T py,
                   T pz,
                   T &fx,
                   T &fy,
                   T &fz,
                   const T softening_factor_sq)
{
    int leaf_first_body = bodies_begin[leaf];
    int leaf_num_bodies = bodies_end[leaf] - leaf_first_body + 1;

    // Load leaf bodies into shared memory in chunks
    for (int k = 0; k < leaf_num_bodies; k += GROUP_SIZE) {

        if (k + threadIdx.x < leaf_num_bodies) {
            int body_idx = leaf_first_body + k + threadIdx.x;
            x_buff[threadIdx.x] = bodies_pos.x(body_idx);
            y_buff[threadIdx.x] = bodies_pos.y(body_idx);
            z_buff[threadIdx.x] = bodies_pos.z(body_idx);
            m_buff[threadIdx.x] = bodies_mass[body_idx];

        }
        __syncthreads();

        int chunk_size = min(k + GROUP_SIZE, leaf_num_bodies) - k;
        // #pragma unroll
        // TODO: try unrolling when possible
        for (int b = 0; b < chunk_size; ++b) {

            PhysicsCommon<T>::
            accumulate_pairwise_force(px,
                                      py,
                                      pz,
                                      x_buff[b],
                                      y_buff[b],
                                      z_buff[b],
                                      m_buff[b],
                                      fx,
                                      fy,
                                      fz,
                                      softening_factor_sq);
        }
    }
}

template<typename T>
__global__ void barnes_hut_traverse(const SoAVec3<T> bodies_pos,
                                    const T *bodies_mass,
                                    SoAVec3<T> bodies_acc,
                                    const SoAOctreeNodes nodes,
                                    const SoAVec3<T> nodes_barycenter,
                                    const T *nodes_size,
                                    const T *nodes_mass,
                                    const int *bodies_begin,
                                    const int *bodies_end,
                                    const int *codes_first_point_idx,
                                    const int *leaf_first_code_idx,
                                    int *queue,
                                    int *next_queue,
                                    int queue_size,
                                    float theta,
                                    int num_leaves,
                                    int num_bodies)
{
    int body_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // WARNING: number of bodies should be multiple of 32
    if (body_idx >= num_bodies) {
        return;
    }

    extern __shared__ int shmem[];

    T *x_buff = (T *) shmem;
    T *y_buff = &x_buff[GROUP_SIZE];
    T *z_buff = &y_buff[GROUP_SIZE];
    T *m_buff = &z_buff[GROUP_SIZE];

    int *approx_buff = (int *) &m_buff[GROUP_SIZE];
    int *open_buff = &approx_buff[GROUP_SIZE * 2];
    int *nchildren_buff = &open_buff[GROUP_SIZE];

    int *approx_blk_scan = &nchildren_buff[GROUP_SIZE];
    int *open_blk_scan = &approx_blk_scan[WARP_SIZE];
    int *nchildren_blk_scan = &open_blk_scan[WARP_SIZE];

    unsigned int *sh_leaf_mask =
        (unsigned int *) &nchildren_blk_scan[WARP_SIZE];

    queue += blockIdx.x * QUEUE_SIZE;
    next_queue += blockIdx.x * QUEUE_SIZE;

    int approx_buff_size = 0;
    int open_buff_size = 0;
    int tot_num_children = 0;

    int lane_idx = threadIdx.x % WARP_SIZE;
    int warp_idx = threadIdx.x / WARP_SIZE;

    T px = bodies_pos.x(body_idx);
    T py = bodies_pos.y(body_idx);
    T pz = bodies_pos.z(body_idx);
    T m = bodies_mass[body_idx];

    T fx = (T) 0.0f;
    T fy = (T) 0.0f;
    T fz = (T) 0.0f;

    T softening_factor_sq = PhysicsCommon<T>::get_softening_factor_sq();

    if (threadIdx.x == 0) {
        // TODO: prefill with nodes from lower levels
        queue[0] = 0;
    }

    while (queue_size > 0) {
        int next_queue_size = 0;
        // Process each node in the queue in round-robin fashion
        for (int i = 0; i < queue_size; i += GROUP_SIZE) {

            x_buff[threadIdx.x] = px;
            y_buff[threadIdx.x] = py;
            z_buff[threadIdx.x] = pz;

            // Ensure memory ordering
            //__syncwarp();
            __syncthreads();

            int node;
            int first_child;
            int num_children = 0;

            bool approx_node = 0;
            bool open_node = 0;
            bool is_leaf = 0;

            if (i + threadIdx.x < queue_size) {

                node = queue[threadIdx.x + i];

                T bx = nodes_barycenter.x(node);
                T by = nodes_barycenter.y(node);
                T bz = nodes_barycenter.z(node);

                T min_dist = compute_group_to_node_min_dist(x_buff,
                                                            y_buff,
                                                            z_buff,
                                                            bx,
                                                            by,
                                                            bz);

                num_children = nodes.num_children(node);
                first_child = nodes.first_child(node);
                float size = nodes_size[node];

                is_leaf = num_children == 0;
                approx_node = approx_crit(size, min_dist, theta) &&
                            !is_leaf;
                open_node = !is_leaf && !approx_node;
            }

            // In-warp exclusive scan to obtain scatter indices
            // int leaf_node_scatter = _warp_scan(is_leaf);
            int approx_node_scatter = warp_scan(approx_node, lane_idx);
            int open_node_scatter = warp_scan(open_node, lane_idx);
            int nchildren_scatter = warp_scan(open_node * num_children,
                                              lane_idx);

            // Aggregate warp scans to compute block scan
            if (lane_idx == WARP_SIZE - 1) {
                approx_blk_scan[warp_idx] = approx_node_scatter + approx_node;
                open_blk_scan[warp_idx] = open_node_scatter + open_node;
                nchildren_blk_scan[warp_idx] = nchildren_scatter +
                                               open_node * num_children;
            }

            __syncthreads();

            if (warp_idx == 0) {
                int value = warp_scan(lane_idx < NUM_WARPS ?
                                      approx_blk_scan[lane_idx] : 0,
                                      lane_idx);
                approx_blk_scan[lane_idx] = value;

                value = warp_scan(lane_idx < NUM_WARPS ?
                                  open_blk_scan[lane_idx] : 0,
                                  lane_idx);
                open_blk_scan[lane_idx] = value;

                value = warp_scan(lane_idx < NUM_WARPS ?
                                  nchildren_blk_scan[lane_idx] : 0,
                                  lane_idx);
                nchildren_blk_scan[lane_idx] = value;
            }

            __syncthreads();

            approx_node_scatter += approx_blk_scan[warp_idx];
            open_node_scatter += open_blk_scan[warp_idx];
            nchildren_scatter += nchildren_blk_scan[warp_idx];

            if (approx_node) {
                int idx = approx_buff_size + approx_node_scatter;
                approx_buff[idx] = node;
            } else if (open_node) {
                int idx = open_buff_size + open_node_scatter;
                // Buffer index of first child
                open_buff[idx] = first_child;
                nchildren_buff[idx] = nchildren_scatter;
            }

            __syncthreads();

            // Update offset to allow buffering scatter indices

            if (warp_idx == NUM_WARPS - 1) {
                // Let thread 32 compute the length of the buffers
                approx_buff_size +=
                    __shfl_sync(0xffffffff,
                                approx_node_scatter + approx_node,
                                31);
                open_buff_size =
                    __shfl_sync(0xffffffff,
                                open_node_scatter + open_node,
                                31);
                tot_num_children =
                    __shfl_sync(0xffffffff,
                                nchildren_scatter + open_node * num_children,
                                31);

                if (lane_idx == 0) {
                    // Reuse buffers to share sizes with other warps
                    approx_blk_scan[0] = approx_buff_size;
                    open_blk_scan[0] = open_buff_size;
                    nchildren_blk_scan[0] = tot_num_children;
                }
            }

            __syncthreads();

            if (warp_idx < NUM_WARPS - 1) {
                // Perhaps only one thread reads from shmem and
                // then uses __shfl??
                approx_buff_size = approx_blk_scan[0];
                open_buff_size = open_blk_scan[0];
                tot_num_children = nchildren_blk_scan[0];
            }

            // Append children nodes to next queue
            // TODO: try to evaluate when open_buff_size > 4, or more
            // e.g. when open_buff_size > 32
            if (open_buff_size > 0) {
                if (next_queue_size + tot_num_children > QUEUE_SIZE) {
                    if (threadIdx.x == 0) {
                        printf("[%04d:%04d] WARNING: not enough memory for "
                               "next traversal queue, "
                               "traversal will terminate early, "
                               "increase mem_traversal_queues.\n",
                               threadIdx.x, blockIdx.x);
                    }
                } else {
                    append_to_queue(open_buff,
                                    nchildren_buff,
                                    next_queue,
                                    open_buff_size,
                                    next_queue_size,
                                    tot_num_children);
                    // We can avoid __syncthreads() here as long as we do it
                    // at least once before the next iteration

                    next_queue_size += tot_num_children;
                }
                open_buff_size = 0;
            }

            for (int w = 0; w < NUM_WARPS; ++w) {

                __syncthreads();

                unsigned int leaf_mask;
                // Reuse buffer
                int *leaves = approx_blk_scan;

                if (warp_idx == w) {
                    leaf_mask = __ballot_sync(0xffffffff, is_leaf);
                    if (lane_idx == 0) {
                        *sh_leaf_mask = leaf_mask;
                    }

                    leaves[lane_idx] = node;
                }

                __syncthreads();

                if (warp_idx != w) {
                    leaf_mask = *sh_leaf_mask;
                }

                unsigned int src_lane = 0;
                int n = 1;
                // Loop over the set bits in leaf_mask
                while ((src_lane = __fns(leaf_mask, 0, n)) < 32) {
                    int leaf = leaves[src_lane];
                    evaluate_leaf(bodies_pos,
                                  bodies_mass,
                                  bodies_begin,
                                  bodies_end,
                                  leaf,
                                  x_buff,
                                  y_buff,
                                  z_buff,
                                  m_buff,
                                  px,
                                  py,
                                  pz,
                                  fx,
                                  fy,
                                  fz,
                                  softening_factor_sq);
                    n++;
                }
            }

            // Evaluate cluster interaction list
            if (approx_buff_size >= GROUP_SIZE) {
                approx_buff_size = evaluate_approx(bodies_pos,
                                                   nodes_barycenter,
                                                   nodes_mass,
                                                   bodies_begin,
                                                   bodies_end,
                                                   px,
                                                   py,
                                                   pz,
                                                   fx,
                                                   fy,
                                                   fz,
                                                   approx_buff,
                                                   x_buff,
                                                   y_buff,
                                                   z_buff,
                                                   m_buff,
                                                   approx_buff_size,
                                                   softening_factor_sq);
            }

            // Shouldn't be strictly needed here
            __syncthreads();
        }
        // __syncthreads() here yes though

        int *tmp = queue;
        queue = next_queue;
        next_queue = tmp;

        queue_size = next_queue_size;
    }

    __syncthreads();

    if (approx_buff_size > 0) {
        approx_buff_size = evaluate_approx(bodies_pos,
                                           nodes_barycenter,
                                           nodes_mass,
                                           bodies_begin,
                                           bodies_end,
                                           px,
                                           py,
                                           pz,
                                           fx,
                                           fy,
                                           fz,
                                           approx_buff,
                                           x_buff,
                                           y_buff,
                                           z_buff,
                                           m_buff,
                                           approx_buff_size,
                                           softening_factor_sq);
    }

    T gravity = PhysicsCommon<T>::get_gravity();
    bodies_acc.x(body_idx) = gravity * fx;
    bodies_acc.y(body_idx) = gravity * fy;
    bodies_acc.z(body_idx) = gravity * fz;
}

template<typename T>
BarnesHut<T>::BarnesHut(SoAVec3<T> &bodies_pos,
                        int num_bodies,
                        float theta,
                        float dt,
                        size_t mem_queues,
                        int group_size) :
    pos_(bodies_pos),
    num_bodies_(num_bodies),
    theta_(theta),
    dt_(dt),
    mem_queues_(mem_queues),
    group_size_(group_size)
{
    // Allocate space for traversal queues
    cudaMalloc(&queues_, mem_queues);

    vel_.alloc(num_bodies);
    vel_half_.alloc(num_bodies);
    acc_.alloc(num_bodies);

    tmp_vel_.alloc(num_bodies);
    tmp_vel_half_.alloc(num_bodies);
    tmp_acc_.alloc(num_bodies);

    vel_.zeros(num_bodies);
    acc_.zeros(num_bodies);

    // Copy to constant memory
    int num_groups = (num_bodies - 1) / group_size + 1;
    int queue_size_per_group = mem_queues / (2 * sizeof(int) * num_groups);
    cudaMemcpyToSymbol(QUEUE_SIZE, &queue_size_per_group, sizeof(int));
    cudaMemcpyToSymbol(GROUP_SIZE, &group_size, sizeof(int));

    int num_warps = (group_size - 1) / WARP_SIZE + 1;
    cudaMemcpyToSymbol(NUM_WARPS, &num_warps, sizeof(int));
}

template<typename T>
void BarnesHut<T>::sort_bodies(const int *sort_indices)
{
    tmp_vel_.gather(vel_, sort_indices, num_bodies_);
    tmp_vel_half_.gather(vel_half_, sort_indices, num_bodies_);
    tmp_acc_.gather(acc_, sort_indices, num_bodies_);

    vel_.swap(tmp_vel_);
    vel_half_.swap(tmp_vel_half_);
    acc_.swap(tmp_acc_);
}

template<typename T>
void BarnesHut<T>::solve_pos(const Octree<T> &octree,
                             const int *codes_first_point_idx,
                             const int *leaf_first_code_idx,
                             int num_octree_leaves)
{
    leapfrog_integrate_pos<<<num_bodies_ / MAX_THREADS_PER_BLOCK +
                             (num_bodies_ % MAX_THREADS_PER_BLOCK > 0),
                             MAX_THREADS_PER_BLOCK>>>(pos_,
                                                      vel_,
                                                      vel_half_,
                                                      acc_,
                                                      dt_,
                                                      num_bodies_);
}

template<typename T>
void BarnesHut<T>::solve_vel(const Octree<T> &octree,
                             const T *bodies_mass,
                             const int *codes_first_point_idx,
                             const int *leaf_first_code_idx,
                             int num_octree_leaves)
{
    compute_forces(octree,
                   bodies_mass,
                   codes_first_point_idx,
                   leaf_first_code_idx,
                   num_octree_leaves);

    leapfrog_integrate_vel<<<num_bodies_ / MAX_THREADS_PER_BLOCK +
                             (num_bodies_ % MAX_THREADS_PER_BLOCK > 0),
                             MAX_THREADS_PER_BLOCK>>>(vel_,
                                                      vel_half_,
                                                      acc_,
                                                      dt_,
                                                      num_bodies_);
}

template<typename T>
void BarnesHut<T>::compute_forces(const Octree<T> &octree,
                                  const T *bodies_mass,
                                  const int *codes_first_point_idx,
                                  const int *leaf_first_code_idx,
                                  int num_octree_leaves)
{
    int shmem_size = 4 * sizeof(T) * group_size_ +
                     4 * sizeof(int) * group_size_ +
                     3 * sizeof(int) * WARP_SIZE +
                     sizeof(unsigned int);

    barnes_hut_traverse<<<num_bodies_ / group_size_ +
                          (num_bodies_ % group_size_ > 0),
                          group_size_,
                          shmem_size>>>
                               (pos_,
                                bodies_mass,
                                acc_,
                                octree.get_d_nodes(),
                                octree.get_d_barycenters(),
                                octree.get_d_nodes_size(),
                                octree.get_d_nodes_mass(),
                                octree.get_d_points_begin_ptr(),
                                octree.get_d_points_end_ptr(),
                                codes_first_point_idx,
                                leaf_first_code_idx,
                                queues_,
                                queues_ + mem_queues_ / sizeof(int) / 2,
                                1,
                                theta_,
                                num_octree_leaves,
                                num_bodies_);
}

template<typename T>
BarnesHut<T>::~BarnesHut()
{
    vel_.free();
    vel_half_.free();
    acc_.free();

    tmp_vel_.free();
    tmp_vel_half_.free();
    tmp_acc_.free();

    cudaFree(queues_);
}

template class BarnesHut<float>;
template class BarnesHut<double>;
