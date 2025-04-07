#include "cuda/barnes_hut.cuh"

#include "cuda/soa_vec3.cuh"
#include "cuda/soa_octree_nodes.cuh"
#include "cuda/octree.cuh"

#define WARP_SIZE 32
#define EPS 1e-2
#define GRAVITY 0.1

__device__ __forceinline__ int _warp_scan(int var)
{
    int res = var;
    #pragma unroll
    for (int i = 0, delta = 1; i < 5; ++i, delta <<= 1) {
        int value = __shfl_up_sync(0xffffffff, res, delta);
        if (threadIdx.x >= delta) {
            res += value;
        }
    }

    return res - var;
}

template<typename T> __device__ __forceinline__
bool _approx_crit(float size, T dist, float theta)
{
    return dist > size / theta;
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
    T min_dist_sq = 3;

    #pragma unroll
    for (int i = 0; i < WARP_SIZE; ++i) {
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

    return sqrt(min_dist_sq);
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

    T den = dist_sq + EPS;
    den = sqrt(den * den * den);

    dst_x += mass * GRAVITY * dist_x / den;
    dst_y += mass * GRAVITY * dist_y / den;
    dst_z += mass * GRAVITY * dist_z / den;
}

__device__ __forceinline__ void _append_to_queue(const int *open_buff,
                                                 const int *nchildren_buff,
                                                 int *queue,
                                                 int open_buff_size,
                                                 int queue_size,
                                                 int num_nodes)
{
    for (int j = 0; j + threadIdx.x < num_nodes; j += WARP_SIZE) {

        // Iterative binary search

        int left = 0;
        int step = open_buff_size >> 1;
        // TODO: check if this works
        while (step > 0) {
            if (j + threadIdx.x >= nchildren_buff[left + step]) {
                left += step;
            }
            step >>= 1;
        }

        /*
        int left = 0;

        while (left < open_buff_size &&
               j + threadIdx.x >= nchildren_buff[left]) {
            left++;
        }
        left--;
        */

        int queue_dst = queue_size + j + threadIdx.x;

        __syncwarp(__activemask());

        queue[queue_dst] = open_buff[left] +
                           j +
                           threadIdx.x -
                           nchildren_buff[left];
    }
}

template<typename T> __device__ __forceinline__
int _evaluate_approx(const SoAVec3<T> bodies_pos,
                     //SoAVec3<T> bodies_acc,
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
                     //int group_first_body,
                     //int group_num_bodies)
{
    int start_buff_idx = max(0, approx_buff_size - 32);

    if (threadIdx.x < approx_buff_size) {
        int node = approx_buff[start_buff_idx + threadIdx.x];
        // Reuse buffer to hold cluster mass
        approx_buff[threadIdx.x] = bodies_end[node] -
                                   bodies_begin[node] + 1;
        x_buff[threadIdx.x] = nodes_barycenter.x(node);
        y_buff[threadIdx.x] = nodes_barycenter.y(node);
        z_buff[threadIdx.x] = nodes_barycenter.z(node);
    }

    __syncwarp();

    /*
    // Loop over points in warp in chunks
    for (int j = 0; j + threadIdx.x < group_num_bodies; j += WARP_SIZE) {
        int idx = group_first_body + j + threadIdx.x;

        T px = bodies_pos.x(idx);
        T py = bodies_pos.y(idx);
        T pz = bodies_pos.z(idx);

        T fx = 0.0;
        T fy = 0.0;
        T fz = 0.0;
        */

    int chunk_size = approx_buff_size - start_buff_idx;
    // #pragma unroll
    for (int k = 0; k < chunk_size; ++k) {
        _compute_pairwise_force(px,
                                py,
                                pz,
                                x_buff[k],
                                y_buff[k],
                                z_buff[k],
                                (T) approx_buff[k],
                                fx,
                                fy,
                                fz);
    }

    /*
        bodies_acc.x(idx) += fx;
        bodies_acc.y(idx) += fy;
        bodies_acc.z(idx) += fz;
    }
    */

    return start_buff_idx;
}

template<typename T> __device__ __forceinline__
void _evaluate_leaf(const SoAVec3<T> bodies_pos,
                    // SoAVec3<T> bodies_acc,
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
                    T &fz
                    /*
                    int group_first_body,
                    int group_num_bodies
                    */)
{
    int leaf_first_body = bodies_begin[leaf];
    int leaf_num_bodies = bodies_end[leaf] - leaf_first_body + 1;

    // Loop over points in warp in chunks
    /*
    for (int j = 0; j < group_num_bodies; j += WARP_SIZE) {
        int idx = group_first_body + j + threadIdx.x;

        T px, py, pz;

        T fx = 0.0;
        T fy = 0.0;
        T fz = 0.0;

        if (j + threadIdx.x < group_num_bodies) {
            px = bodies_pos.x(idx);
            py = bodies_pos.y(idx);
            pz = bodies_pos.z(idx);
        }
    */

    // Load leaf bodies into shared memory in chunks
    for (int k = 0; k < leaf_num_bodies; k += WARP_SIZE) {

        if (k + threadIdx.x < leaf_num_bodies) {
            int body_idx = leaf_first_body + k + threadIdx.x;
            x_buff[threadIdx.x] = bodies_pos.x(body_idx);
            y_buff[threadIdx.x] = bodies_pos.y(body_idx);
            z_buff[threadIdx.x] = bodies_pos.z(body_idx);
        }
        __syncwarp();

        /*
            if (j + threadIdx.x >= group_num_bodies) {
                continue;
            }
        */

        int chunk_size = min(k + 32, leaf_num_bodies) - k;
        // #pragma unroll
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

    /*
        if (j + threadIdx.x < group_num_bodies) {
            bodies_acc.x(idx) += fx;
            bodies_acc.y(idx) += fy;
            bodies_acc.z(idx) += fz;
        }
    }
    */
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

    __shared__ T x_buff[WARP_SIZE];
    __shared__ T y_buff[WARP_SIZE];
    __shared__ T z_buff[WARP_SIZE];

    __shared__ int approx_buff[WARP_SIZE * 2];
    // __shared__ int leaf_buff[WARP_SIZE];
    __shared__ int open_buff[WARP_SIZE + 4];
    __shared__ int nchildren_buff[WARP_SIZE + 4];

    // __shared__ int queue_buff[512];

    queue += blockIdx.x * 8192;
    next_queue += (num_bodies / 32) * 8192 + blockIdx.x * 8192;

    /*
    queue = &queue_buff[0];
    next_queue = &queue_buff[4096];
    */

    int approx_buff_size = 0;
    int open_buff_size = 0;
    int leaves_eval = 0;

    int tot_queue_size = 0;

    /*
    int first_code_idx = leaf_first_code_idx[blockIdx.x];
    int end_code_idx = leaf_first_code_idx[blockIdx.x + 1];

    int group_first_body = codes_first_point_idx[first_code_idx];
    int group_end_body = codes_first_point_idx[end_code_idx];
    int group_num_bodies = group_end_body - group_first_body;
    */

    // We're considering each code as a single body during the
    // traversal, even though it may map to more than one body.
    // The position of the first point mapped to a given code is
    // taken as the position of the whole code.
    // Nevertheless, when forces need to be evaluated,
    // all points covered by the codes are considered.

    /*
    int num_codes = end_code_idx - first_code_idx;
    // Index of the first point covered by the current code
    int code_pos_idx =
        codes_first_point_idx[first_code_idx + threadIdx.x % num_codes];

    T cx = bodies_pos.x(code_pos_idx);
    T cy = bodies_pos.y(code_pos_idx);
    T cz = bodies_pos.z(code_pos_idx);
    */

    T px = bodies_pos.x(body_idx);
    T py = bodies_pos.y(body_idx);
    T pz = bodies_pos.z(body_idx);

    T fx = 0.0;
    T fy = 0.0;
    T fz = 0.0;

    // TODO: prefill with nodes from lower levels
    queue[0] = 0;

    while (queue_size > 0) {
        int next_queue_size = 0;
        // Process each node in the queue in round-robin fashion
        for (int i = 0; i < queue_size; i += WARP_SIZE) {

            x_buff[threadIdx.x] = px;
            y_buff[threadIdx.x] = py;
            z_buff[threadIdx.x] = pz;

            // Ensure memory ordering
            __syncwarp();

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

                T min_dist_sq = _compute_group_to_node_min_dist(x_buff,
                                                                y_buff,
                                                                z_buff,
                                                                bx,
                                                                by,
                                                                bz);

                num_children = nodes.num_children(node);
                first_child = nodes.first_child(node);
                float size = nodes.size(node);

                is_leaf = num_children == 0;
                approx_node = _approx_crit(size, min_dist_sq, theta) &&
                            !is_leaf;
                open_node = !is_leaf && !approx_node;

                /*
                if (blockIdx.x == 0) {
                    printf("[%04d] node=%04d leaf=%d approx=%d open=%d, size=%.3f\n",
                           threadIdx.x, node, is_leaf, approx_node, open_node, size);
                }
                */
            }

            // In-warp exclusive scan to obtain scatter indices
            // int leaf_node_scatter = _warp_scan(is_leaf);
            int approx_node_scatter = _warp_scan(approx_node);
            int open_node_scatter = _warp_scan(open_node);
            int nchildren_scatter = _warp_scan(open_node * num_children);

            /*
            if (blockIdx.x == 0) {
                printf("%2d %d %d %d\n",
                       threadIdx.x, nchildren_scatter,
                       first_child, open_node_scatter);
            }
            */

            // Compaction into buffers
            /*if (is_leaf) {
                leaf_buff[leaf_node_scatter] = node;
            } else */

            if (approx_node) {
                int idx = approx_buff_size + approx_node_scatter;
                approx_buff[idx] = node;
            } else if (open_node) {
                int idx = open_buff_size + open_node_scatter;
                // Buffer index of first child
                open_buff[idx] = first_child;
                nchildren_buff[idx] = nchildren_scatter; //+ nchildren_offset;
            }

            // Update offset to allow buffering scatter indices
            // TODO: there has to be a simpler way, just get the
            // last element of the inclusive scan
            unsigned int open_mask = __ballot_sync(0xffffffff, open_node);
            int last_nchildren = 0;
            int last_open_lane = 32 - __ffs(__brev(open_mask));
            if (last_open_lane >= 0) {
                last_nchildren = __shfl_sync(0xffffffff,
                                             num_children,
                                             last_open_lane);
            }
            // nchildren_offset += last_nchildren;

            // Let warp 32 compute the length of the buffers
            approx_buff_size +=
                __shfl_sync(0xffffffff,
                            approx_node_scatter + approx_node,
                            31);
            open_buff_size =
                __shfl_sync(0xffffffff,
                            open_node_scatter + open_node,
                            31);

            /*
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                printf("approx_size=%d open_size=%d\n",
                       approx_buff_size, open_buff_size);
                for (int k = 0; k < open_buff_size; ++k) {
                    printf(" %d", nchildren_buff[k]);
                }
                printf("\n");
            }
            */

            // Append children nodes to next queue
            // TODO: try to evaluate when open_buff_size > 4, or more
            // e.g. when open_buff_size > 32
            if (open_buff_size > 0) {
                int num_nodes = nchildren_buff[open_buff_size - 1] +
                                last_nchildren;

                _append_to_queue(open_buff,
                                 nchildren_buff,
                                 next_queue,
                                 open_buff_size,
                                 next_queue_size,
                                 num_nodes);

                open_buff_size = 0;
                // nchildren_offset = 0;
                next_queue_size += num_nodes;
                tot_queue_size += num_nodes;

                /*
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    printf("[");
                    for (int j = 0; j < next_queue_size; ++j) {
                        printf(" %d", next_queue[j]);
                    }
                    printf(" ]\n");
                }
                */
            }

            unsigned int leaf_mask = __ballot_sync(0xffffffff, is_leaf);
            unsigned int src_lane = 0;
            int n = 1;
            // Loop over the set bits in leaf_mask
            while ((src_lane = __fns(leaf_mask, 0, n)) < 32) {
                int leaf = __shfl_sync(0xffffffff, node, src_lane);
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
                leaves_eval++;
            }

            // Evaluate cluster interaction list
            if (approx_buff_size >= 32) {
                approx_buff_size = _evaluate_approx(bodies_pos,
                                                    //bodies_acc,
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
                                                    //group_first_body,
                                                    //group_num_bodies);
            }
        }

        int *tmp = queue;
        queue = next_queue;
        next_queue = tmp;

        queue_size = next_queue_size;
    }

    if (approx_buff_size > 0) {
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
                                                    //group_first_body,
                                                    //group_num_bodies);
        approx_buff_size = 0;
    }

    _compute_pairwise_force(px,
                            py,
                            pz,
                            (T) 0.5,
                            (T) 0.5,
                            (T) 0.5,
                            (T) 2000.0,
                            fx,
                            fy,
                            fz);

    bodies_acc.x(body_idx) += fx;
    bodies_acc.y(body_idx) += fy;
    bodies_acc.z(body_idx) += fz;

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("%d, %d\n", tot_queue_size, leaves_eval);

    /*
    for (int j = 0; j + threadIdx.x < group_num_bodies; j += WARP_SIZE) {
        int idx = group_first_body + j + threadIdx.x;

        T px = bodies_pos.x(idx);
        T py = bodies_pos.y(idx);
        T pz = bodies_pos.z(idx);

        T fx = 0.0;
        T fy = 0.0;
        T fz = 0.0;

        _compute_pairwise_force(px,
                                py,
                                pz,
                                (T) 0.5,
                                (T) 0.5,
                                (T) 0.5,
                                (T) 100.0,
                                fx,
                                fy,
                                fz);

        bodies_acc.x(idx) += fx;
        bodies_acc.y(idx) += fy;
        bodies_acc.z(idx) += fz;
    }
    */
}

template<typename T>
__device__ void _impose_boundary_conditions(T &x, T &y, T &z,
                                            T &vx, T &vy, T &vz)
{
    if (x < 0 || x > 1.0) {
        vx *= -0.5;
        x = max(0.0, min(1.0, x));
    }

    if (y < 0 || y > 1.0) {
        vy *= -0.5;
        y = max(0.0, min(1.0, y));
    }

    if (z < 0 || z > 1.0) {
        vz *= -0.5;
        z = max(0.0, min(1.0, z));
    }
}

template<typename T>
__global__ void _integrate_semi_implicit_euler(SoAVec3<T> bodies_pos,
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

    x += vx * dt + 0.5 * acc.x(idx) * dt * dt;
    y += vy * dt + 0.5 * acc.y(idx) * dt * dt;
    z += vz * dt + 0.5 * acc.z(idx) * dt * dt;

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

    vel.x(idx) += 0.5 * acc.x(idx) * dt;
    vel.y(idx) += 0.5 * acc.y(idx) * dt;
    vel.z(idx) += 0.5 * acc.z(idx) * dt;
}


template<typename T>
BarnesHut<T>::BarnesHut(SoAVec3<T> bodies_pos,
                        int num_bodies,
                        float theta,
                        float dt) :
    _pos(bodies_pos),
    _num_bodies(num_bodies),
    _theta(theta),
    _dt(dt)
{
    // Allocate space for traversal queues
    cudaMalloc(&_queues, 2 * (num_bodies / 16) * 8192 * sizeof(int));

    _vel.alloc(num_bodies);
    _acc.alloc(num_bodies);

    // _vel.rand(num_bodies);
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
        _vel.plummer_vel(_pos, _num_bodies, 0.2);
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
    _barnes_hut_traverse<<<_num_bodies / 32 + (_num_bodies % 32 > 0), 32>>>
        /*num_octree_leaves, 32>>>*/(_pos,
                                 _acc,
                                 octree.get_d_nodes(),
                                 octree.get_d_barycenters(),
                                 octree.get_d_points_begin_ptr(),
                                 octree.get_d_points_end_ptr(),
                                 codes_first_point_idx,
                                 leaf_first_code_idx,
                                 _queues,
                                 _queues + (_num_bodies / 32) * 8192,
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
