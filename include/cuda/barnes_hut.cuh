#ifndef BARNES_HUT_CUH
#define BARNES_HUT_CUH

#include "cuda/soa_vec3.cuh"
#include "cuda/octree.cuh"

template<typename T> class BarnesHut
{
public:
    BarnesHut(SoAVec3<T> bodies_pos);

    void compute_forces(const Octree<T> octree);
    void update_bodies(T dt);

    ~BarnesHut();

private:
    SoAVec3<T> _pos;
    SoAVec3<T> _vel;
    SoAVec3<T> _acc;
};

#endif
