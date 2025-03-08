#include "CUDAWrappers.hpp"

#include "cuda/points_gpu.cuh"
#include "cuda/btree_gpu.cuh"
#include "cuda/octree_gpu.cuh"

namespace CUDAWrappers
{
    struct BarnesHut::Impl
    {
        Impl(int num_points) :
            points(num_points),
            btree(num_points),
            octree(num_points) {}

        Points<float> points;
        Btree btree;
        Octree<float> octree;
    };

    BarnesHut::BarnesHut(int num_points)
    {
        _impl = std::make_unique<BarnesHut::Impl>(num_points);
    }

    void BarnesHut::spawnPoints()
    {
        _impl->points.sample_uniform();
    }

    BarnesHut::~BarnesHut() {}
};
