#ifndef BARNES_HUT_CUH
#define BARNES_HUT_CUH

#include "cuda/soa_vec3.cuh"
#include "cuda/octree.cuh"

template<typename T> class BarnesHut
{
public:
    BarnesHut(SoAVec3<T> bodies_pos, int num_bodies);

    SoAVec3<T> &get_d_vel()
    {
        return _vel;
    }

    void compute_forces(const Octree<T> &octree,
                        const int *codes_first_point_idx,
                        const int *leaf_first_code_idx,
                        int num_leaves);

    void update_bodies();

    ~BarnesHut();

private:
    const T _dt = 0.0001;

    int _num_bodies;

    SoAVec3<T> _pos;
    SoAVec3<T> _vel;
    SoAVec3<T> _acc;

    int *_queues;
};

#endif
