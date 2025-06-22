#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <GL/glew.h>

#include <memory>
#include <vector>
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
        int traversal_group_size;
        size_t mem_traversal_queues;
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
    void spawn_points();
    void update();
    int get_num_octree_nodes();
    void set_render_all_pairs(bool value);
    void write_validation_history(const std::string &csv_file_path);
    ~Simulation();

private:
    template<typename U> struct Impl;
    std::unique_ptr<Impl<T>> _impl;

    Params params_;
    bool render_all_pairs_ = false;
};

#endif
