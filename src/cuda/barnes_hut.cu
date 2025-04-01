#include "cuda/barnes_hut.cuh"

#include "cuda/soa_vec3.cuh"
#include "cuda/soa_octree_nodes.cuh"
#include "cuda/octree.cuh"

#define WARP_SIZE 32
#define EPS 1e-6

__device__ __forceinline__ int _warp_scan(int var)
{
    #pragma unroll
    for (int i = 0, delta = 1; i < 5; ++i, delta <<= 1) {
        int value = __shfl_up_sync(0xffffffff, var, delta);
        if (threadIdx.x >= delta) {
            var += prev_value;
        }
    }
    return var;
}

template<typename T> __device__ __forceinline__
bool _opening_crit(T bx, T by, T bz, T dist, T theta)
{
    return true;
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
        int j = (i + threadIdx.x) % WARP_SIZE;
        T dx = x_buff[j] - x_node;
        T dy = y_buff[j] - y_node;
        T dz = z_buff[j] - z_node;

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

    T dist = sqrt(dist_x * dist_x +
                  dist_y * dist_y +
                  dist_z * dist_z) / mass + EPS;

    dst_x = dist_x / dist;
    dst_y = dist_y / dist;
    dst_z = dist_z / dist;
}

__device__ __forceinline__ int _append_to_queue(const int *open_buff,
                                                const int *open_size_buff,
                                                int *queue,
                                                int queue_size,
                                                int num_nodes)
{
    for (int j = 0; j + threadIdx.x < num_nodes; j += WARP_SIZE) {

        // Iterative binary search
        int start = min(j + 32, num_nodes);
        int length = num_nodes - start;

        int left = 0;
        int step = length >> 1;
        // TODO: check if this works
        while (step > 0) {
            if (j + threadIdx.x >= open_size_buff[left + step]) {
                left += step;
            }
            step >>= 1;
        }

        int queue_dst = next_queue_size + j + threadIdx.x;

        __syncwarp(__activemask());

        next_queue[queue_dst] = open_buff[left];
                                threadIdx.x -
                                open_size_buff[left];
    }
}

template<typename T> __device__ __forceinline__
int _evaluate_approx(const SoAVec3<T> bodies_pos,
                     SoAVec3<T> bodies_acc,
                     const SoAVec3<T> nodes_barycenter,
                     const int *bodies_begin,
                     const int *bodies_end,
                     int *cluster_buff,
                     T *x_buff,
                     T *y_buff,
                     T *z_buff,
                     int cluster_buff_size,
                     int first_point_idx,
                     int num_points)
{
    int start_buff_idx = cluster_buff_size - 32;
    int cluster = cluster_buff[start_buff_idx + threadIdx.x];
    // Reuse buffer to hold cluster mass
    cluster_buff[threadIdx.x] = bodies_end[cluster] -
                                bodies_begin[cluster];
    x_buff[threadIdx.x] = nodes_barycenter.x(cluster);
    y_buff[threadIdx.x] = nodes_barycenter.y(cluster);
    z_buff[threadIdx.x] = nodes_barycenter.z(cluster);

    __syncwarp();

    // Loop over points in warp in chunks
    for (int j = 0; j + threadIdx.x < num_points; j += WARP_SIZE) {
        int idx = first_point_idx + j + threadIdx.x;

        T px = bodies_pos.x(idx);
        T py = bodies_pos.y(idx);
        T pz = bodies_pos.z(idx);

        T fx = 0.0;
        T fy = 0.0;
        T fz = 0.0;

        #pragma unroll
        for (int k = 0; k < WARP_SIZE; ++k) {
            _compute_pairwise_force(x_buff[k],
                                    y_buff[k],
                                    z_buff[k],
                                    px,
                                    py,
                                    pz,
                                    cluster_buff[k]
                                    fx,
                                    fy,
                                    fz);
        }

        bodies_acc.x(idx) += fx;
        bodies_acc.y(idx) += fy;
        bodies_acc.z(idx) += fz;
    }

    return start_buff_idx;
}

template<typename T> __device__ __forceinline__
void _evaluate_leaf(const SoAVec3<T> bodies_pos,
                    SoAVec3<T> bodies_acc,
                    const int *bodies_begin,
                    const int *bodies_end,
                    int leaf,
                    T *x_buff,
                    T *y_buff,
                    T *z_buff,
                    int group_first_body,
                    int group_num_bodies)
{
    int leaf_first_body = bodies_begin[node];
    int leaf_num_bodies = bodies_end[node] - leaf_first_body;

    // Loop over points in warp in chunks
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

        // Load bodies into shared memory in chunks
        for (int k = 0; k < leaf_num_bodies; k += WARP_SIZE) {
            __syncwarp();

            if (k + threadIdx.x < leaf_num_bodies) {
                int idx = leaf_first_body + k + threadIdx.x;
                x_buff[threadIdx.x] = bodies_pos.x(idx)
                y_buff[threadIdx.x] = bodies_pos.y(idx);
                z_buff[threadIdx.x] = bodies_pos.z(idx);
            }

            __syncwarp();

            if (j + threadIdx.x >= group_num_bodies) {
                continue;
            }

            bool is_same_leaf_chunk = leaf_first_body + k ==
                                      group_first_body + j;

            #pragma unroll
            for (int b = 0; b < WARP_SIZE; ++b) {
                if (b + k >= leaf_num_bodies) {
                    break;
                }

                if (!(is_same_leaf_chunk && b == threadIdx.x)) {
                    _compute_pairwise_force(x_buff[b],
                                            y_buff[b],
                                            z_buff[b],
                                            px,
                                            py,
                                            pz,
                                            1.0,
                                            fx,
                                            fy,
                                            fz);
                }
            }
        }
    }
}

template<typename T>
__global__ void _barnes_hut_traverse(const SoAVec3<T> bodies_pos,
                                     T *bodies_acc,
                                     const SoAOctreeNodes nodes,
                                     const SoAVec3<T> nodes_barycenter,
                                     const int *bodies_begin,
                                     const int *bodies_end,
                                     const int *queue,
                                     int *next_queue,
                                     int queue_size,
                                     T theta,
                                     int num_bodies)
{
    if (blockIdx.x >= *num_leaves) {
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
    __shared__ int open_buff[WARP_SIZE + 4];
    __shared__ int nchildren_buff[WARP_SIZE + 4];

    int approx_buff_size = 0;
    int open_buff_size = 0;

    int nchildren_offset = 0;

    int first_code_idx = leaf_first_code_idx[blockIdx.x];
    int end_code_idx = leaf_first_code_idx[blockIdx.x + 1];

    int group_first_body = codes_first_point_idx[first_code_idx];
    int group_end_body = codes_first_point_idx[end_code_idx];
    int group_num_bodies = group_end_body - group_first_body;

    // We're considering each code as a single body during the
    // traversal, even though it may map to more than one body.
    // The position of the first point mapped to a given code is
    // taken as the position of the whole code.
    // Nevertheless, when forces need to be evaluated,
    // all points covered by the codes are considered.

    int num_codes = end_code_idx - first_code_idx;
    // Index of the first point covered by the current code
    int code_pos_idx =
        codes_first_point_idx[first_code_idx + threadIdx.x % num_codes];

    T cx = bodies_pos.x(code_pos_idx);
    T cy = bodies_pos.y(code_pos_idx);
    T cz = bodies_pos.z(code_pos_idx);

    while (queue_size > 0) {
        int next_queue_size = 0;
        // Process each node in the queue in round-robin fashion
        for (int i = 0; i < queue_size; i += WARP_SIZE) {

            x_buff[threadIdx.x] = cx;
            y_buff[threadIdx.x] = cy;
            z_buff[threadIdx.x] = cz;

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

                is_leaf = num_children == 0;
                open_node = _opening_crit(bx, by, bz, min_dist_sq, theta) &&
                            !is_leaf;
                approx_node = !is_leaf && !open_node;
            }

            // In-warp exclusive scan to obtain scatter indices
            int approx_node_scatter = _warp_scan(approximate_node);
            int open_node_scatter = _warp_scan(open_node);
            int nchildren_scatter = _warp_scan(open_node * num_children);
            // Compaction into buffers
            if (approx_node) {
                int idx = approx_buff_size + approx_node_scatter;
                approx_buff[idx] = node;
            } else if (open_node) {
                int idx = open_buff_size + open_node_scatter;
                // Buffer index of first child and 
                open_buff[idx] = first_child;
                nchildren_buff[idx] = nchildren_scatter + nchildren_offset;
            }

            // Update offset to allow buffering scatter indices
            unsigned int open_mask = __ballot_sync(0xffffffff, open_node);
            int last_nchildren = 0;
            int last_open_lane = 32 - __ffs(__brev(open_mask));
            if (last_open_lane >= 0) {
                last_nchildren = __shfl_sync(0xffffffff,
                                             num_children,
                                             last_open_lane);
            }
            nchildren_offset += last_nchildren;

            // Let warp 32 compute the length of the buffers
            cluster_buff_size =
                __shfl_sync(0xffffffff,
                            approx_buff_off + approx_node,
                            31);
            open_buff_size =
                __shfl_sync(0xffffffff,
                            open_buff_off + open_node,
                            31);

            // Append children nodes to next queue
            if (open_buff_size >= 4) {
                int num_nodes = open_size_scan[open_buff_size - 1] + 
                                last_open_num_children;

                _append_to_queue(open_buff,
                                 open_buff_size,
                                 next_queue,
                                 next_queue_size,
                                 num_nodes);

                open_buff_size = 0;
                nchildren_offset = 0;
                next_queue_size += num_nodes;
            }

            unsigned int leaf_mask = __ballot_sync(0xffffffff, leaf_mask_bit);
            unsigned int src_lane = 0;
            // Loop over the set bits in leaf_mask
            while ((src_lane = __fns(leaf_mask, src_lane, 0)) < 32) {
                int leaf = __shfl_sync(0xffffffff, node, src_lane);
                _evaluate_leaf(bodies_pos,
                               bodies_acc,
                               bodies_begin,
                               bodies_end,
                               leaf,
                               x_buff,
                               y_buff,
                               z_buff,
                               group_first_body,
                               group_num_bodies);
            }

            // Evaluate cluster interaction list
            if (cluster_buff_size >= 32) {
                cluster_buff_size = _evaluate_approx(bodies_pos,
                                                     bodies_acc,
                                                     nodes_barycenter,
                                                     bodies_begin,
                                                     bodies_end,
                                                     cluster_buff,
                                                     x_buff,
                                                     y_buff,
                                                     z_buff,
                                                     cluster_buff_size,
                                                     group_first_body,
                                                     group_num_bodies);
            }

        }

        if (next_queue_size > 0) {
            int *tmp = queue;
            queue = next_queue;
            next_queue = tmp;

            queue_size = next_queue_size;
        }
    }

    // TODO: evaluate remaining nodes in buffers

}

template<typename T>
BarnesHut<T>::BarnesHut(SoAVec3<T> bodies_pos) : _pos(bodies_pos) {}

template<typename T>
BarnesHut<T>::compute_forces(const Octree<T> octree)
{
}

template<typename T>
BarnesHut<T>::update_bodies(T dt)
{
}

template<typename T>
BarnesHut<T>::~BarnesHut() {}
