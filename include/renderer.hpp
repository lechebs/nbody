#ifndef RENDERER_H
#define RENDERER_H

#include <string>
#include <memory>
#include <cmath>

#include <SDL2/SDL.h>

#include "camera.hpp"
#include "shader_program.hpp"

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

    constexpr static int _NUMshader_programs_ = 3;
    constexpr static int _NUMssbos_ = 7;

    bool init_();
    bool load_shaders();

    void alloc_buffers();
    void setup_scene();

    void handle_events();
    void update_delta_time();
    void updatecamera_();
    void render_frame();

    bool running_ = true;
    bool initialized_ = false;

    bool paused_ = false;
    bool draw_octree_ = false;
    bool draw_domain_ = false;

    // TODO: create separate Window class
    SDL_Window *window_;
    int window_width_;
    int window_height_;
    std::string window_title_;

    GLuint quad_vao_;
    GLuint cube_vao_;
    GLuint ssbos_[_NUMssbos_];

    ShaderProgram shader_programs_[_NUMshader_programs_];

    Camera camera_;

    // Frametime
    float delta_time_;
};

#endif
