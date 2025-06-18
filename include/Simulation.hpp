#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <GL/glew.h>

#include <memory>
#include <string>

// Opaque class
template<typename T>
class Simulation
{
public:
    struct Params
    {
        int num_points;
        int max_num_codes_per_leaf;
        float theta;
        float dt;
        float gravity;
        float softening_factor;
        float domain_size;
        float velocity_dampening;
        int num_steps_validator;
    };

    Simulation(Params &params);
    Simulation(Params &params, GLuint buffers[7]);
    void spawnBodies();
    void update();
    int get_num_octree_nodes();
    void set_render_all_pairs(bool value);
    void writeHistory(const std::string &csv_file_path);
    ~Simulation();

private:
    template<typename U> struct Impl;
    std::unique_ptr<Impl<T>> _impl;

    Params _params;
    bool _render_all_pairs = false;
};

#endif
