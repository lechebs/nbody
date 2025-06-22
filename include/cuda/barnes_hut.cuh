#ifndef BARNES_HUT_CUH
#define BARNES_HUT_CUH

#include "cuda/soa_vec3.cuh"
#include "cuda/octree.cuh"

template<typename T> class BarnesHut
{
public:
    BarnesHut(SoAVec3<T> &bodies_pos,
              int num_bodies,
              float theta,
              float dt,
              size_t mem_queues,
              int group_size);

    SoAVec3<T> &get_d_vel()
    {
        return vel_;
    }

    SoAVec3<T> &get_d_vel_half()
    {
        return vel_half_;
    }

    SoAVec3<T> &get_d_acc()
    {
        return acc_;
    }

    void sort_bodies(const int *sort_indices);

    // TODO: pass this parameters to constructor
    void solve_pos(const Octree<T> &octree,
                   const int *codes_first_point_idx,
                   const int *leaf_first_code_idx,
                   int num_leaves);

    void solve_vel(const Octree<T> &octree,
                   const T *bodies_mass,
                   const int *codes_first_point_idx,
                   const int *leaf_first_code_idx,
                   int num_leaves);

    ~BarnesHut();

private:
    void compute_forces(const Octree<T> &octree,
                        const T *bodies_mass,
                        const int *codes_first_point_idx,
                        const int *leaf_first_code_idx,
                        int num_leaves);

    int num_bodies_;
    float theta_;
    float dt_;
    size_t mem_queues_;
    int group_size_;

    SoAVec3<T> &pos_;

    SoAVec3<T> vel_;
    SoAVec3<T> vel_half_;
    SoAVec3<T> acc_;

    SoAVec3<T> tmp_vel_;
    SoAVec3<T> tmp_vel_half_;
    SoAVec3<T> tmp_acc_;

    float *mass_;
    float *tmp_mass_;

    int *queues_;
};

#endif
