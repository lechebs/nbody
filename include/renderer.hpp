#ifndef RENDERER_H
#define RENDERER_H

#include <string>
#include <memory>
#include <cmath>

#include <SDL2/SDL.h>

#include "camera.hpp"
#include "shader_program.hpp"
#include "simulation.hpp"

#define SIM_PRECISION float

class Renderer
{
public:
    Renderer(unsigned int window_width, unsigned int window_height);

    bool init();
    void run();
    void quit();

    ~Renderer();

private:
    enum ShaderProgramId { PARTICLE_SHADER, CUBE_SHADER, OCTREE_SHADER };
    enum SimulationSpawner { SPAWN_UNIFORM, SPAWN_SPHERE, SPAWN_DISK };

    constexpr static int NUM_SHADER_PROGRAMS_ = 3;
    constexpr static int NUM_SSBOS_ = 7;

    bool init_();
    bool load_shaders();

    void alloc_buffers();
    void setup_scene();

    void start_simulation(SimulationSpawner spawner);

    void handle_events();
    void update_delta_time();
    void update_camera();
    void render_frame();

    bool running_ = true;
    bool initialized_ = false;

    bool paused_ = false;
    bool draw_octree_ = false;
    bool draw_domain_ = false;

    SDL_Window *window_;
    int window_width_;
    int window_height_;
    std::string window_title_;

    GLuint quad_vao_;
    GLuint cube_vao_;
    GLuint ssbos_[NUM_SSBOS_];

    ShaderProgram shader_programs_[NUM_SHADER_PROGRAMS_];

    Camera camera_;

    std::unique_ptr<Simulation<SIM_PRECISION>> simulation_;

    // Frametime
    float delta_time_;

    int num_points_;
    int num_octree_nodes_;
};

#endif
