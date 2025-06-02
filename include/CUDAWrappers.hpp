#ifndef CUDA_WRAPPERS_HPP
#define CUDA_WRAPPERS_HPP

#include <GL/glew.h>

#include <memory>
#include <string>

namespace CUDAWrappers
{
    // Opaque class
    class Simulation
    {
    public:
        struct Params
        {
            int num_points;
            int max_num_codes_per_leaf;
            float theta;
            float dt;
        };

        Simulation(Params &params);
        Simulation(Params &params, GLuint buffers[5]);
        void samplePoints();
        void update();
        void writeHistory(const std::string &csv_file_path);
        ~Simulation();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;

        Params _params;
    };
};

#endif
