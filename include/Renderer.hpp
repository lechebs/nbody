#ifndef RENDERER_H
#define RENDERER_H

#include <string>
#include <memory>
#include <cmath>

#include <SDL2/SDL.h>

#include "Camera.hpp"
#include "ShaderProgram.hpp"

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

    constexpr static int _NUM_SHADER_PROGRAMS = 3;
    constexpr static int _NUM_SSBOS = 7;

    bool _init();
    bool _loadShaders();

    void _allocBuffers();
    void _setupScene();

    void _handleEvents();
    void _updateDeltaTime();
    void _updateCamera();
    void _renderFrame();

    bool _running = true;
    bool _initialized = false;

    // TODO: create separate Window class
    SDL_Window *_window;
    int _window_width;
    int _window_height;
    std::string _window_title;

    GLuint _quad_vao;
    GLuint _cube_vao;
    GLuint _ssbos[_NUM_SSBOS];

    ShaderProgram _shader_programs[_NUM_SHADER_PROGRAMS];

    Camera _camera;

    // Frametime
    float _delta_time;
};

#endif
