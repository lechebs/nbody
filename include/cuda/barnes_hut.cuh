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
              int compute_energy_max_steps);

    SoAVec3<T> &get_d_vel()
    {
        return _vel;
    }

    SoAVec3<T> &get_d_acc()
    {
        return _acc;
    }

    // TODO: takes this parameters from constructor
    void solve_pos(const Octree<T> &octree,
                   const int *codes_first_point_idx,
                   const int *leaf_first_code_idx,
                   int num_leaves);

    void solve_vel(const Octree<T> &octree,
                   const int *codes_first_point_idx,
                   const int *leaf_first_code_idx,
                   int num_leaves);

    ~BarnesHut();

private:
    void _compute_forces(const Octree<T> &octree,
                         const int *codes_first_point_idx,
                         const int *leaf_first_code_idx,
                         int num_leaves);

    void _update_pos();
    void _update_vel();

    void _compute_energy();

    int _num_bodies;
    float _theta;
    float _dt;

    SoAVec3<T> &_pos;
    SoAVec3<T> _vel;
    SoAVec3<T> _acc;

    int _curr_step;
    int _compute_energy_max_steps;

    T *_tmp_energy;
    T *_tmp_reduce;
    size_t _tmp_reduce_size;

    T *_total_energy;

    int *_queues;
};

#endif
