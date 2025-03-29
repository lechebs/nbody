#include "cuda/barnes_hut.cuh"

#include "cuda/soa_vec3.cuh"
#include "cuda/soa_octree_nodes.cuh"
#include "cuda/octree.cuh"

#define WARP_SIZE 32

__device__ int _scan_warp(int value)
{
    int delta = 1;

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        if (threadIdx.x >= delta) {
            value += __shfl_up_sync(0xffffffff, value, delta);
        }
        delta = delta << 1;
    }

    return value;
}

template<typename T> __device__ __forceinline__
bool _opening_crit(T bx, T by, T bz, T dist, T theta)
{
    return true;
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
                  dist_z * dist_z) / mass;

    dst_x = dist_x / dist;
    dst_y = dist_y / dist;
    dst_z = dist_z / dist;

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
int _evaluate_cluster(const SoAVec3<T> bodies_pos,
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

            if (j + threadIdx.x < leaf_num_bodies) {
                int idx = leaf_first_body + j + threadIdx.x;
                x_buff[threadIdx.x] = bodies_pos.x(idx)
                y_buff[threadIdx.x] = bodies_pos.y(idx);
                z_buff[threadIdx.x] = bodies_pos.z(idx);
            }

            __syncwarp();

            if (j + threadIdx.x >= group_num_bodies) {
                continue;
            }

            #pragma unroll
            for (int b = 0; b < WARP_SIZE; ++b) {
                if (b + k >= leaf_num_bodies) {
                    break;
                }

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

template<typename T>
__global__ void _barnes_hut_traverse(const SoAVec3<T> bodies_pos,
                                     const SoAOctreeNodes nodes,
                                     const SoAVec3<T> nodes_barycenter,
                                     const int *bodies_begin,
                                     const int *bodies_end,
                                     const int *queue,
                                     int *next_queue,
                                     T *bodies_acc,

                                     int queue_size,
                                     T theta,
                                     int num_bodies)
{
    if (blockIdx.x >= *num_leaves) {
        return;
    } 

    __shared__ T buff_x[WARP_SIZE];
    __shared__ T buff_y[WARP_SIZE];
    __shared__ T buff_z[WARP_SIZE];

    // __shared__ int leaf_buff[64];
    __shared__ int cluster_buff[64];
    __shared__ int open_buff[36];
    __shared__ int open_size_buff[36];

    // int leaf_buff_size = 0;
    int cluster_buff_size = 0;
    int open_buff_size = 0;
    int open_size_last = 0;

    int first_code_idx = leaf_first_code_idx[blockIdx.x];
    int end_code_idx = leaf_first_code_idx[blockIdx.x + 1];
    int num_codes = end_code_idx - first_code_idx;

    int first_point_idx = codes_first_point_idx[first_code_idx];
    int end_point_idx = codes_first_point_idx[end_code_idx];
    int num_points = end_point_idx - first_point_idx;

    // Index of the first point covered by the current code
    int p_idx =
        codes_first_point_idx[first_code_idx + threadIdx.x % num_codes];

    T cx = bodies_pos.x(p_idx);
    T cy = bodies_pos.y(p_idx);
    T cz = bodies_pos.z(p_idx);

    while (queue_size > 0) {
        int next_queue_size = 0;
        // Process each node in the queue in round-robin fashion
        for (int i = 0; i < queue_size; i += WARP_SIZE) {

            buff_x[threadIdx.x] = cx;
            buff_y[threadIdx.x] = cy;
            buff_z[threadIdx.x] = cz;

            // Ensure memory ordering
            __syncwarp();

            int node;
            int first_child;
            int num_children = 0;

            int cluster_mask_bit = 0;
            int open_mask_bit = 0;

            if (i + threadIdx.x < queue_size) {
                node = queue[threadIdx.x + i];
                // Get barycenter of current node
                T bx = nodes_barycenter.x(node);
                T by = nodes_barycenter.y(node);
                T bz = nodes_barycenter.z(node);

                // Compute min distance of group to barycenter
                // WARNING: coordinates are within the unit cube
                T min_dist_sq = 3;
                #pragma unroll
                for (int j = 0; j < WARP_SIZE; ++j) {
                    // Shifting to avoid bank conflicts
                    int k = (j + threadIdx.x) % WARP_SIZE;
                    T dx = buff_x[k] - bx;
                    T dy = buff_y[k] - by;
                    T dz = buff_z[k] - bz;

                    T dist_sq = px * px + py * py + pz * pz;
                    if (dist_sq <= min_dist_sq) {
                        min_dist_sq = dist_sq;
                    }
                }

                // Synchronize active threads to allow coalescing
                __syncwarp(__activemask());

                num_children = nodes.num_children(node);
                first_child = nodes.first_child(node);
                leaf_mask_bit = num_children == 0;

                open_mask_bit = !leaf_mask_bit &&
                                _opening_crit(bx, by, bz, min_dist_sq, theta);

                cluster_mask_bit = !leaf_mask_bit && !open_mask_bit;
            }

            int cluster_mask_scan = cluster_mask_bit;
            int open_mask_scan = open_mask_bit;
            int open_size_scan = open_mask_bit * num_children;
            // In-warp exclusive scan of the bitmasks
            #pragma unroll
            for (int j = 0, delta = 1; j < 5; ++j, delta <<= 1) {

                int cluster_val = __shfl_up_sync(0xffffffff,
                                                 cluster_mask_scan,
                                                 delta);

                int open_val = __shfl_up_sync(0xffffffff,
                                              open_mask_scan,
                                              delta);

                int size_val = __shfl_up_sync(0xffffffff,
                                              open_size_scan,
                                              delta);


                if (threadIdx.x >= delta) {
                    cluster_mask_scan += cluster_val;
                    open_mask_scan += open_val;
                    open_size_scan += size_val;
                 }
            }

            // Compact interactions to be evaluated and nodes
            // to be opened into buffers
            int cluster_buff_off = cluster_buff_size + cluster_mask_scan;
            int open_buff_off = open_buff_size + open_mask_scan;

            if (cluster_mask_bit) {
                cluster_buff[cluster_buff_off] = node;
            } else if (open_mask_bit) {
                open_buff[open_buff_off] = first_child;
                open_size_buff[open_buff_off] = open_size_scan +
                                                open_size_last;
            }

            unsigned int open_mask = __ballot_sync(0xffffffff, open_mask_bit);
            int last_open_num_children = 0;
            int last_open_bit = 32 - __ffs(__brev(open_mask));
            if (last_open_bit >= 0) {
                last_num_children = __shfl_sync(0xffffffff,
                                                num_children,
                                                last_open_bit);
            }
            open_size_last += last_open_num_children;

            // Let warp 32 compute the length of the buffers
            cluster_buff_size =
                __shfl_sync(0xffffffff,
                            cluster_buff_off + cluster_mask_bit,
                            31);
            open_buff_size =
                __shfl_sync(0xffffffff,
                            open_buff_off + open_mask_bit,
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
                open_size_last = 0;
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
                               buff_x,
                               buff_y,
                               buff_z,
                               first_point_idx,
                               num_points);
            }

            // Evaluate cluster interaction list
            if (cluster_buff_size >= 32) {
                cluster_buff_size = _evaluate_cluster(bodies_pos,
                                                      bodies_acc,
                                                      nodes_barycenter,
                                                      bodies_begin,
                                                      bodies_end,
                                                      cluster_buff,
                                                      buff_x,
                                                      buff_y,
                                                      buff_z,
                                                      cluster_buff_size,
                                                      first_point_idx,
                                                      num_points);
            }

        }

        if (next_queue_size > 0) {
            int *tmp = queue;
            queue = next_queue;
            next_queue = tmp;

            queue_size = next_queue_size;
        }
    }

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


            // Evaluate leaf interaction list
            /*
            if (leaf_buff_size >= 32) {

                // Load leaves data into shared memory
                int start_buff_idx = leaf_buff_size - 32;
                int leaf = leaf_buff[start_buff_idx + threadIdx.x];

                int leaf_first_body_idx = bodies_begin[leaf];
                int leaf_end_body_idx = bodies_end[leaf];

                leaf_buff[start_buff_idx] = leaf_first_body_idx;
                buff_nbodies[threadIdx.x] = leaf_end_body_idx -
                                              leaf_first_body_idx;

                leaf_buff_size = start_buff_idx;

                __syncwarp();

                // Loop over cached leaves
                #pragma unroll
                for (int l = 0; l < 32; ++l) {

                    int first_body_idx = leaf_buff[start_buff_idx + l];
                    int num_bodies = buff_nbodies[l];

                    // Loop over chunks of points inside the current warp
                    for (int p = first_point_idx;
                         p < end_point_idx;
                         p += WARP_SIZE) {

                        T px, py pz;

                        T fx = 0.0;
                        T fy = 0.0;
                        T fz = 0.0;

                        if (p + threadIdx.x < end_point_idx) {
                            // Hopefully this will be often in cache
                            px = pos.x(j + threadIdx.x);
                            py = pos.y(j + threadIdx.x);
                            pz = pos.z(j + threadIdx.x);
                        }

                        for (int k = 0; k < num_bodies; k += WARP_SIZE) {
                            __syncwarp();

                            if (k + threadIdx.x < num_bodies) {
                                int idx = first_body_idx + k + threadIdx.x;
                                buff_x[threadIdx.x] = pos.x(idx)
                                buff_y[threadIdx.x] = pos.y(idx);
                                buff_z[threadIdx.x] = pos.z(idx);
                            }

                            __syncwarp();

                            if (p + threadIdx.x >= end_point_idx) {
                                continue;
                            }

                            #pragma unroll
                            for (int lp = 0; lp < 32; ++lp) {
                                if (k + lp >= num_bodies) {
                                    break;
                                }

                                _compute_pairwise_force(buff_x[lp],
                                                        buff_y[lp],
                                                        buff_z[lp],
                                                        px,
                                                        py,
                                                        pz,
                                                        1.0,
                                                        fx,
                                                        fy,
                                                        fz);
                            }
                        }

                        bodies_acc.x(j + threadIdx.x) += fx;
                        bodies_acc.y(j + threadIdx.x) += fy;
                        bodies_acc.z(j + threadIdx.x) += fz;
                    }
                }
            }
            */

