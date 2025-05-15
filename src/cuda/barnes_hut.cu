#include "cuda/barnes_hut.cuh"

#include <cassert>

#include "cuda/soa_vec3.cuh"
#include "cuda/soa_octree_nodes.cuh"
#include "cuda/octree.cuh"

#define WARP_SIZE 32
#define GROUP_SIZE 512
#define NUM_WARPS (GROUP_SIZE / WARP_SIZE)
#define EPS 1e-2f
#define GRAVITY 0.003f
#define DIST_SCALE 100.0f
#define VELOCITY_DAMPENING 0.98f

// TODO: try removing inlining to reduce registers usage
__device__ __forceinline__ int _warp_scan(int var, int lane_idx)
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
bool _approx_crit(float size, T dist_sq, float theta)
{
    return size * size / dist_sq < theta * theta;
}

template<typename T> __device__ __forceinline__
T _compute_group_to_node_min_dist(const T *x_group,
                                  const T *y_group,
                                  const T *z_group,
                                  T x_node,
                                  T y_node,
                                  T z_node)
{
    // WARNING: coordinates are within the unit cube
    T min_dist_sq = (T) 3.0f;

    #pragma unroll
    for (int i = 0; i < GROUP_SIZE; ++i) {
        // Shifting to avoid bank conflicts
        // int j = (i + threadIdx.x) % WARP_SIZE;
        T dx = x_group[i] - x_node;
        T dy = y_group[i] - y_node;
        T dz = z_group[i] - z_node;

        T dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq <= min_dist_sq) {
            min_dist_sq = dist_sq;
        }
    }

    return min_dist_sq;
}

template<typename T> __device__ __forceinline__
void _compute_pairwise_force(T p1x, T p1y, T p1z,
                             T p2x, T p2y, T p2z,
                             T mass,
                             T &dst_x, T &dst_y, T &dst_z)
{
    T dist_x = p2x - p1x;
    T dist_y = p2y - p1y;
    T dist_z = p2z - p1z;

    T dist_sq = dist_x * dist_x +
                dist_y * dist_y +
                dist_z * dist_z;

    T inv_den = DIST_SCALE * dist_sq + (T) EPS * EPS;
    inv_den = __frsqrt_rn(inv_den * inv_den * inv_den);

    dst_x += mass * (T) GRAVITY * dist_x * inv_den;
    dst_y += mass * (T) GRAVITY * dist_y * inv_den;
    dst_z += mass * (T) GRAVITY * dist_z * inv_den;
}

__device__ __forceinline__ void _append_to_queue(const int *open_buff,
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
        // TODO: check if this works
        do {
            step = (step + 1) >> 1;
            if (left + step < open_buff_size &&
                j + threadIdx.x >= nchildren_buff[left + step]) {
                left += step;
            }
        } while (step > 1);

        int queue_dst = queue_size + j + threadIdx.x;

        if (queue_dst > 8192)
        printf("%d %d\n", blockIdx.x, threadIdx.x);

        __syncwarp(__activemask());

        queue[queue_dst] = open_buff[left] +
                           j +
                           threadIdx.x -
                           nchildren_buff[left];
    }
}

template<typename T> __device__ __forceinline__
int _evaluate_approx(const SoAVec3<T> bodies_pos,
                     const SoAVec3<T> nodes_barycenter,
                     const int *bodies_begin,
                     const int *bodies_end,
                     T px, T py, T pz,
                     T &fx, T &fy, T &fz,
                     int *approx_buff,
                     T *x_buff,
                     T *y_buff,
                     T *z_buff,
                     int approx_buff_size)
{
    int start_buff_idx = max(0, approx_buff_size - GROUP_SIZE);

    if (threadIdx.x < approx_buff_size) {
        int node = approx_buff[start_buff_idx + threadIdx.x];
        // Reuse buffer to hold cluster mass
        approx_buff[start_buff_idx + threadIdx.x] =
            bodies_end[node] - bodies_begin[node] + 1;
        x_buff[threadIdx.x] = nodes_barycenter.x(node);
        y_buff[threadIdx.x] = nodes_barycenter.y(node);
        z_buff[threadIdx.x] = nodes_barycenter.z(node);
    }

    __syncthreads();

    int chunk_size = approx_buff_size - start_buff_idx;
    // #pragma unroll
    // TODO: try unrolling when possible
    for (int k = 0; k < chunk_size; ++k) {
        _compute_pairwise_force(px,
                                py,
                                pz,
                                x_buff[k],
                                y_buff[k],
                                z_buff[k],
                                (T) approx_buff[start_buff_idx + k],
                                fx,
                                fy,
                                fz);
    }

    return start_buff_idx;
}

template<typename T> __device__ __forceinline__
void _evaluate_leaf(const SoAVec3<T> bodies_pos,
                    const int *bodies_begin,
                    const int *bodies_end,
                    int leaf,
                    T *x_buff,
                    T *y_buff,
                    T *z_buff,
                    T px,
                    T py,
                    T pz,
                    T &fx,
                    T &fy,
                    T &fz)
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
        }
        __syncthreads();

        int chunk_size = min(k + GROUP_SIZE, leaf_num_bodies) - k;
        // #pragma unroll
        // TODO: try unrolling when possible
        for (int b = 0; b < chunk_size; ++b) {
            _compute_pairwise_force(px,
                                    py,
                                    pz,
                                    x_buff[b],
                                    y_buff[b],
                                    z_buff[b],
                                    (T) 1.0,
                                    fx,
                                    fy,
                                    fz);
        }
    }
}

template<typename T>
__global__ void _barnes_hut_traverse(const SoAVec3<T> bodies_pos,
                                     SoAVec3<T> bodies_acc,
                                     const SoAOctreeNodes nodes,
                                     const SoAVec3<T> nodes_barycenter,
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

    // TODO: on sm_89 the max num of blocks per sm is 24
    // and the total shared mem is ~100KB, so blocks of
    // size 32 don't saturate the sm. With blocks of 64
    // we can get ~4KB of shared memory per block and fully
    // saturate the sms. We can thus allocate the node queue
    // on the shared memory, and fallback to global memory
    // when needed, at least that's what they did for Bonsai
    // apparently. The control flow there looks more prone
    // to divergence when it comes to handling the type of node,
    // so perhaps don't bother buffering nodes to be opened.
    // Also, try to use shfl instructions to avoid
    // using shared memory to pass around bodies data when
    // evaluating p2p forces.

    __shared__ T x_buff[GROUP_SIZE];
    __shared__ T y_buff[GROUP_SIZE];
    __shared__ T z_buff[GROUP_SIZE];

    __shared__ int approx_buff[GROUP_SIZE * 2];
    __shared__ int open_buff[GROUP_SIZE];
    __shared__ int nchildren_buff[GROUP_SIZE];

    __shared__ int approx_blk_scan[WARP_SIZE];
    __shared__ int open_blk_scan[WARP_SIZE];
    __shared__ int nchildren_blk_scan[WARP_SIZE];

    __shared__ unsigned int sh_leaf_mask;

    //__shared__ int queue_buff[4096];

    queue += blockIdx.x * 8192;
    next_queue += blockIdx.x * 8192;

    //queue = queue_buff;
    //next_queue = queue_buff + 2048;

    /*
    queue = &queue_buff[0];
    next_queue = &queue_buff[4096];
    */

    int approx_buff_size = 0;
    int open_buff_size = 0;
    int tot_num_children = 0;

    //int tot_queue_size = 0;

    // We're considering each code as a single body during the
    // traversal, even though it may map to more than one body.
    // The position of the first point mapped to a given code is
    // taken as the position of the whole code.
    // Nevertheless, when forces need to be evaluated,
    // all points covered by the codes are considered.

    //int n_opened = 0;
    //int n_approx = 0;
    //int n_leaves = 0;

    int lane_idx = threadIdx.x % WARP_SIZE;
    int warp_idx = threadIdx.x / WARP_SIZE;

    T px = bodies_pos.x(body_idx);
    T py = bodies_pos.y(body_idx);
    T pz = bodies_pos.z(body_idx);

    T fx = (T) 0.0f;
    T fy = (T) 0.0f;
    T fz = (T) 0.0f;

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

                T min_dist = _compute_group_to_node_min_dist(x_buff,
                                                             y_buff,
                                                             z_buff,
                                                             bx,
                                                             by,
                                                             bz);

                num_children = nodes.num_children(node);
                first_child = nodes.first_child(node);
                float size = nodes.size(node);

                is_leaf = num_children == 0;
                approx_node = _approx_crit(size, min_dist, theta) &&
                            !is_leaf;
                open_node = !is_leaf && !approx_node;

                /*
                if (blockIdx.x == 10) {
                    printf("[%04d - %04d] node=%04d leaf=%d approx=%d open=%d size=%.3f "
                           "(%.3f, %.3f, %.3f) child=%d num_children=%d\n",
                           threadIdx.x, blockIdx.x, node, is_leaf, approx_node, open_node, size,
                           bx, by, bz, first_child, num_children);
                }
                */
            }

            // In-warp exclusive scan to obtain scatter indices
            // int leaf_node_scatter = _warp_scan(is_leaf);
            int approx_node_scatter = _warp_scan(approx_node, lane_idx);
            int open_node_scatter = _warp_scan(open_node, lane_idx);
            int nchildren_scatter = _warp_scan(open_node * num_children,
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
                int value = _warp_scan(lane_idx < NUM_WARPS ?
                                       approx_blk_scan[lane_idx] : 0,
                                       lane_idx);
                approx_blk_scan[lane_idx] = value;

                value = _warp_scan(lane_idx < NUM_WARPS ?
                                   open_blk_scan[lane_idx] : 0,
                                   lane_idx);
                open_blk_scan[lane_idx] = value;

                value = _warp_scan(lane_idx < NUM_WARPS ?
                                   nchildren_blk_scan[lane_idx] : 0,
                                   lane_idx);
                nchildren_blk_scan[lane_idx] = value;
            }

            __syncthreads();

            /*
            if (blockIdx.x == 0 && approx_node) {
                printf("[%2d] approx_node_scatter=%d, offset=%d\n",
                       threadIdx.x, approx_node_scatter, approx_blk_scan[warp_idx]);
            }
            */

            approx_node_scatter += approx_blk_scan[warp_idx];
            open_node_scatter += open_blk_scan[warp_idx];
            nchildren_scatter += nchildren_blk_scan[warp_idx];

            // Compaction into buffers
            /*if (is_leaf) {
                leaf_buff[leaf_node_scatter] = node;
            } else */

            if (approx_node) {
                int idx = approx_buff_size + approx_node_scatter;
                approx_buff[idx] = node;
                /*
                if (blockIdx.x == 0) {
                    printf("[%2d] %d - %d\n", threadIdx.x, idx, node);
                }
                */
            } else if (open_node) {
                int idx = open_buff_size + open_node_scatter;
                // Buffer index of first child
                open_buff[idx] = first_child;
                nchildren_buff[idx] = nchildren_scatter; //+ nchildren_offset;
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
                // Perhaps only one thread reads from shmem and then uses __shfl??
                approx_buff_size = approx_blk_scan[0];
                open_buff_size = open_blk_scan[0];
                tot_num_children = nchildren_blk_scan[0];
            }

            /*
            if (blockIdx.x == 10 && (threadIdx.x == 511)) {
                printf("approx_size=%d open_size=%d\n",
                       approx_buff_size, open_buff_size);
                printf("approx = ");
                for (int k = 0; k < approx_buff_size; ++k) {
                    printf(" %d", approx_buff[k]);
                }
                printf("\nopen = ");
                for (int k = 0; k < open_buff_size; ++k) {
                    printf(" %d", open_buff[k]);
                }
                printf("\n");
            }
            */

            // Append children nodes to next queue
            // TODO: try to evaluate when open_buff_size > 4, or more
            // e.g. when open_buff_size > 32
            if (open_buff_size > 0) {
                //n_opened += open_buff_size;

                _append_to_queue(open_buff,
                                 nchildren_buff,
                                 next_queue,
                                 open_buff_size,
                                 next_queue_size,
                                 tot_num_children);
                // We can avoid __syncthreads() here as long as we do it
                // at least once before the next iteration

                open_buff_size = 0;
                // nchildren_offset = 0;
                next_queue_size += tot_num_children;

                if (next_queue_size > 8192)
                    printf("ERROR: queue too small! %d %d\n", threadIdx.x, blockIdx.x);

                //tot_queue_size += tot_num_children;

                /*
                if (threadIdx.x == 0) {
                    printf("open = [");
                    for (int j = 0; j < next_queue_size; ++j) {
                        printf(" %d", next_queue[j]);
                    }
                    printf(" ]\n");
                }
                */
            }

            for (int w = 0; w < NUM_WARPS; ++w) {

                __syncthreads();

                unsigned int leaf_mask;
                // Reuse buffer
                int *leaves = approx_blk_scan;

                if (warp_idx == w) {
                    leaf_mask = __ballot_sync(0xffffffff, is_leaf);
                    if (lane_idx == 0) {
                        sh_leaf_mask = leaf_mask;
                    }

                    leaves[lane_idx] = node;
                }

                __syncthreads();

                if (warp_idx != w) {
                    leaf_mask = sh_leaf_mask;
                }

                unsigned int src_lane = 0;
                int n = 1;
                // Loop over the set bits in leaf_mask
                while ((src_lane = __fns(leaf_mask, 0, n)) < 32) {
                    int leaf = leaves[src_lane];//__shfl_sync(0xffffffff, node, src_lane);
                    _evaluate_leaf(bodies_pos,
                                   bodies_begin,
                                   bodies_end,
                                   leaf,
                                   x_buff,
                                   y_buff,
                                   z_buff,
                                   px,
                                   py,
                                   pz,
                                   fx,
                                   fy,
                                   fz);
                    n++;
                    //n_leaves++;
                }
            }

            // Evaluate cluster interaction list
            if (approx_buff_size >= GROUP_SIZE) {
                //n_approx += GROUP_SIZE;
                approx_buff_size = _evaluate_approx(bodies_pos,
                                                    nodes_barycenter,
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
                                                    approx_buff_size);
            }

            // Shouldn't be strictly needed here
            __syncthreads();

            //if (threadIdx.x == 0 && blockIdx.x == 10) printf("\n");
        }
        // __syncthreads() here yes though

        int *tmp = queue;
        queue = next_queue;
        next_queue = tmp;

        queue_size = next_queue_size;
    }

    __syncthreads();

    if (approx_buff_size > 0) {
        //n_approx += approx_buff_size;
        approx_buff_size = _evaluate_approx(bodies_pos,
                                            nodes_barycenter,
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
                                            approx_buff_size);
    }

    /*
    _compute_pairwise_force(px,
                            py,
                            pz,
                            (T) 0.5f,
                            (T) 0.5f,
                            (T) 0.5f,
                            (T) 100000.0f,
                            fx,
                            fy,
                            fz);
    */

    bodies_acc.x(body_idx) += fx;
    bodies_acc.y(body_idx) += fy;
    bodies_acc.z(body_idx) += fz;

    /*
    if (blockIdx.x < 10 && threadIdx.x == 0)
        printf("%d, %d\n", tot_queue_size, leaves_eval);
        */

    /*
    if (body_idx == 0) {
        printf("opened=%d, approx=%d, leaves=%d\n", n_opened, n_approx, n_leaves);
    }
    */
}

template<typename T>
__device__ void _impose_boundary_conditions(T &x, T &y, T &z,
                                            T &vx, T &vy, T &vz)
{
    if (x < 0.0f || x > 1.0f) {
        vx *= -1.0f / 2;
        x = max(0.0f, min(1.0f, x));
    }

    if (y < 0.0f || y > 1.0f) {
        vy *= -1.0f / 2;
        y = max(0.0f, min(1.0f, y));
    }

    if (z < 0.0f || z > 1.0f) {
        vz *= -1.0f / 2;
        z = max(0.0f, min(1.0f, z));
    }
}

template<typename T>
__global__ void _semi_implicit_euler_integrate(SoAVec3<T> bodies_pos,
                                               SoAVec3<T> bodies_vel,
                                               const SoAVec3<T> bodies_acc,
                                               T dt,
                                               int num_bodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) {
        return;
    }

    T x = bodies_pos.x(idx);
    T y = bodies_pos.y(idx);
    T z = bodies_pos.z(idx);

    T vx = bodies_vel.x(idx);
    T vy = bodies_vel.y(idx);
    T vz = bodies_vel.z(idx);

    bodies_vel.x(idx) = vx + bodies_acc.x(idx) * dt;
    bodies_vel.y(idx) = vy + bodies_acc.y(idx) * dt;
    bodies_vel.z(idx) = vz + bodies_acc.z(idx) * dt;

    x += vx * dt;
    y += vy * dt;
    z += vz * dt;

    _impose_boundary_conditions(x, y, z, vx, vy, vz);

    bodies_pos.x(idx) = x;
    bodies_pos.y(idx) = y;
    bodies_pos.z(idx) = z;

    bodies_vel.x(idx) = vx;
    bodies_vel.y(idx) = vy;
    bodies_pos.z(idx) = vz;
}

template<typename T>
__global__ void _leapfrog_integrate_pos(SoAVec3<T> pos,
                                        SoAVec3<T> vel,
                                        const SoAVec3<T> acc,
                                        float dt,
                                        int num_bodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) {
        return;
    }

    T x = pos.x(idx);
    T y = pos.y(idx);
    T z = pos.z(idx);

    T vx = vel.x(idx);
    T vy = vel.y(idx);
    T vz = vel.z(idx);

    x += vx * dt + 0.5f * acc.x(idx) * dt * dt;
    y += vy * dt + 0.5f * acc.y(idx) * dt * dt;
    z += vz * dt + 0.5f * acc.z(idx) * dt * dt;

    _impose_boundary_conditions(x, y, z, vx, vy, vz);

    pos.x(idx) = x;
    pos.y(idx) = y;
    pos.z(idx) = z;

    vel.x(idx) = vx;
    vel.y(idx) = vy;
    vel.z(idx) = vz;
}

template<typename T>
__global__ void _leapfrog_integrate_vel(SoAVec3<T> vel,
                                        const SoAVec3<T> acc,
                                        float dt,
                                        int num_bodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) {
        return;
    }

    vel.x(idx) += 0.5f * acc.x(idx) * dt;
    vel.y(idx) += 0.5f * acc.y(idx) * dt;
    vel.z(idx) += 0.5f * acc.z(idx) * dt;

    vel.x(idx) *= VELOCITY_DAMPENING;
    vel.y(idx) *= VELOCITY_DAMPENING;
    vel.z(idx) *= VELOCITY_DAMPENING;
}


template<typename T>
BarnesHut<T>::BarnesHut(SoAVec3<T> &bodies_pos,
                        int num_bodies,
                        float theta,
                        float dt) :
    _pos(bodies_pos),
    _num_bodies(num_bodies),
    _theta(theta),
    _dt(dt)
{
    // Allocate space for traversal queues
    cudaMalloc(&_queues, 2 * (num_bodies / GROUP_SIZE) * 8192 * sizeof(int));

    _vel.alloc(num_bodies);
    _acc.alloc(num_bodies);

    _vel.zeros(num_bodies);
}

template<typename T>
void BarnesHut<T>::solve_pos(const Octree<T> &octree,
                             const int *codes_first_point_idx,
                             const int *leaf_first_code_idx,
                             int num_octree_leaves)
{
    static int init = 0;
    if (!init) {
        init = 1;
        //_vel.tangent(_pos, _num_bodies);
        //_vel.rand(_num_bodies);
        //_vel.hubble(_pos, _num_bodies, 1000);
    }

    _acc.zeros(_num_bodies);

    _compute_forces(octree,
                    codes_first_point_idx,
                    leaf_first_code_idx,
                    num_octree_leaves);

    _update_pos();
}

template<typename T>
void BarnesHut<T>::solve_vel(const Octree<T> &octree,
                             const int *codes_first_point_idx,
                             const int *leaf_first_code_idx,
                             int num_octree_leaves)
{
    _compute_forces(octree,
                    codes_first_point_idx,
                    leaf_first_code_idx,
                    num_octree_leaves);

    _update_vel();
}


template<typename T>
void BarnesHut<T>::_compute_forces(const Octree<T> &octree,
                                   const int *codes_first_point_idx,
                                   const int *leaf_first_code_idx,
                                   int num_octree_leaves)
{
    _barnes_hut_traverse<<<_num_bodies / GROUP_SIZE +
                           (_num_bodies % GROUP_SIZE > 0),
                           GROUP_SIZE>>>
        /*num_octree_leaves, 32>>>*/(_pos,
                                 _acc,
                                 octree.get_d_nodes(),
                                 octree.get_d_barycenters(),
                                 octree.get_d_points_begin_ptr(),
                                 octree.get_d_points_end_ptr(),
                                 codes_first_point_idx,
                                 leaf_first_code_idx,
                                 _queues,
                                 _queues + (_num_bodies / GROUP_SIZE) * 8192,
                                 1,
                                 _theta,
                                 num_octree_leaves,
                                 _num_bodies);
}

template<typename T>
void BarnesHut<T>::_update_pos()
{
    _leapfrog_integrate_pos<<<_num_bodies / MAX_THREADS_PER_BLOCK +
                              (_num_bodies % MAX_THREADS_PER_BLOCK > 0),
                              MAX_THREADS_PER_BLOCK>>>(_pos,
                                                       _vel,
                                                       _acc,
                                                       _dt,
                                                       _num_bodies);
}

template<typename T>
void BarnesHut<T>::_update_vel()
{
    _leapfrog_integrate_vel<<<_num_bodies / MAX_THREADS_PER_BLOCK +
                              (_num_bodies % MAX_THREADS_PER_BLOCK > 0),
                              MAX_THREADS_PER_BLOCK>>>(_vel,
                                                       _acc,
                                                       _dt,
                                                       _num_bodies);
}

template<typename T>
BarnesHut<T>::~BarnesHut()
{
    cudaFree(_queues);
}

template class BarnesHut<float>;
template class BarnesHut<double>;
