#ifndef CUDA_WRAPPERS_HPP
#define CUDA_WRAPPERS_HPP

#include <memory>

namespace CUDAWrappers
{
    // Opaque class
    class BarnesHut
    {
    public:
        BarnesHut(int num_points);
        void spawnPoints();
        ~BarnesHut();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
};

#endif
