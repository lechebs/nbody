#ifndef CUDA_WRAPPERS_HPP
#define CUDA_WRAPPERS_HPP

#include <GL/glew.h>

#include <memory>

namespace CUDAWrappers
{
    // Opaque class
    class BarnesHut
    {
    public:
        struct Params
        {
            int num_points;
            int max_num_codes_per_leaf;
        };

        BarnesHut(Params &params);
        BarnesHut(Params &params, GLuint buffers[3]);
        void samplePoints();
        void update();
        ~BarnesHut();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;

        Params _params;
    };
};

#endif
