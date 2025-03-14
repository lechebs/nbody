#include "cuda/barnes_hut.cuh"

#include "cuda/soa_vec3.cuh"
#include "cuda/soa_octree_nodes.cuh"
#include "cuda/octree.cuh"

#define WARP_SIZE 32

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) {
        return;
    }

    __shared__ T buff_x[WARP_SIZE];
    __shared__ T buff_y[WARP_SIZE];
    __shared__ T buff_z[WARP_SIZE];

    __shared__ int interaction_list[32];

    // Should be kept in registers if indexed predictably
    int queue_alloc[WARP_SIZE];

    T px = pos.x(idx);
    T py = pos.y(idx);
    T pz = pos.z(idx);

    buff_x[threadIdx.x] = px;
    buff_y[threadIdx.x] = py;
    buff_z[threadIdx.x] = pz;

    while (queue_size > 0) {
       // Process each node in the queue in round-robin fashion
        for (int i = 0; i < queue_size; i += WARP_SIZE) {
            if (i >= queue_size) {
                continue;
            }

            unsigned int leaves_mask = 0;
            unsigned int open_mask = 0;

            // Get barycenter of current node
            int node = queue[threadIdx.x + i];
            T bx = nodes_barycenter.x(node);
            T by = nodes_barycenter.y(node);
            T bz = nodes_barycenter.z(node);

            // Compute min distance of group to barycenter
            T min_dist_sq = 3; // coordinates are within the unit cube
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

            bool open_node = _opening_crit(bx, by, bz, min_dist_sq, theta);
            int num_children = nodes.num_children(node);
            if (num_children == 0 || !open_node) {
                // Keep track of node to be evaluated
                interaction_list[threadIdx.x] = node;

                buff_x[threadIdx.x] = bx;
                buff_y[threadIdx.y] = by;
                buff_z[threadIdx.y] = bz;
            }

            leaves_mask = _ballot(num_children == 0);
            open_mask = _ballot(open_node && num_children > 0);

            // TODO: scan masks and compact into list
            // then wait for them to be > 32 before evaluating them
            // this should help with caching

            if (!(open_mask & 0xffffffff)) {
            unsigned int bit_mask = 1;
            #pragma unroll
            for (int j = 0; j < WARP_SIZE; ++j) {

                if (open_mask & bit_mask) {
                    continue;
                }

                int node = interaction_list[j];
                int first_body = bodies_begin[node]
                int last_body = bodies_end[node];

                T force_x = 0;
                T force_y = 0;
                T force_z = 0;

                // Divergence-free selection
                if (leaves_mask & bit_mask) {
                    // TODO: perhaps evaluate only with respect to one
                    // of the points covered by a code? otherwise we
                    // risk a O(n^2) here if the points collapse
                    for (int body = first_body; body <= last_body; ++body) {
                        _compute_pairwise_force(bodies_pos.x(body),
                                                bodies_pos.y(body),
                                                bodies_pos.z(body),
                                                px,
                                                py,
                                                pz,
                                                1.0,
                                                force_x,
                                                force_y,
                                                force_z);
                    }
                } else {
                    T mass = (T) (bodies_end - bodies_begin + 1);
                    _compute_pairwise_force(buff_x[threadIdx.x],
                                            buff_y[threadIdx.y],
                                            buff_z[threadIdx.z],
                                            px,
                                            py,
                                            pz,
                                            mass,
                                            force_x,
                                            force_y,
                                            force_z);
               }

                bodies_acc.x(threadIdx.x) += force_x;
                bodies_acc.x(threadIdx.x) += force_y;
                bodies_acc.x(threadIdx.x) += force_z;

                bit_mask = bit_mask << 1;
            }
            }

            queue_alloc[threadIdx.x] = num_children * open_node;

            // Scan queue_alloc, write next_queue
            // swap queue and next_queue pointers

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
