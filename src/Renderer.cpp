#include "Renderer.hpp"

#include <iostream>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <memory>

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Vector.hpp"
#include "Camera.hpp"
#include "ShaderProgram.hpp"
#include "CUDAWrappers.hpp"

constexpr unsigned int N_POINTS = 2 << 15;

using vec3f = Vector<float, 3>;
using vec3d = Vector<double, 3>;

static bool paused = false;

Renderer::Renderer(unsigned int window_width,
                   unsigned int window_height) :
    _window_width(window_width),
    _window_height(window_height),
    _window_title("nbody"),
    _camera(M_PI / 3,
            static_cast<float>(window_width) / window_height,
            -0.01,
            -20.0) {}

bool Renderer::init()
{
    return _initialized = _init() && _loadShaders();
}

void Renderer::run()
{
    assert(_initialized);

    _allocBuffers();
    _setupScene();

    CUDAWrappers::Simulation::Params p = { N_POINTS, 32, 0.8, 0.0001 };
    CUDAWrappers::Simulation simulation(p, _particles_ssbo);
    simulation.samplePoints();

    while (_running) {
        _handleEvents();
        _updateDeltaTime();

        if (!paused) {
            simulation.update();
        }

        _updateCamera();
        _renderFrame();
    }

    simulation.writeHistory("output.csv");
}

void Renderer::quit()
{
    _running = false;
}

// Initializes the SDL window and the attached OpenGL context
bool Renderer::_init()
{
    // TODO: create utility to handle errors and logging

    // Initializing SDL2 video subsystem.
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cout << SDL_GetError() << std::endl;
        return false;
    }

    // Using OpenGL 4.3 core profile.
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    // Using double buffering.
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    // Creating SDL2 window with OpenGL support.
    if ((_window = SDL_CreateWindow(
        _window_title.c_str(),
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _window_width,
        _window_height,
        SDL_WINDOW_OPENGL)) == nullptr) {
        std::cout << SDL_GetError() << std::endl;
        return false;
    }

    // Creating OpenGL context.
    if (SDL_GL_CreateContext(_window) == nullptr) {
        std::cout << SDL_GetError() << std::endl;
        return false;
    }

    // Loading OpenGL function pointers.
    GLenum glew_err = glewInit();
    if (glew_err != GLEW_OK) {
        std::cout << glewGetErrorString(glew_err) << std::endl;
        return false;
    }

    // Enabling v-sync.
    SDL_GL_SetSwapInterval(1);

    return true;
}

// Loads, compiles and links OpenGL shaders
bool Renderer::_loadShaders()
{
    const std::array<std::string, 4> shaders_filename = {
        "shaders/particle.vert",
        "shaders/particle.frag",
        "shaders/cube.vert",
        "shaders/cube.frag"
    };

    bool loaded = true;

    for (int i = 0; i < _NUM_SHADER_PROGRAMS; ++i) {
        // Needs to be called after OpenGL context creation
        _shader_programs[i].create();

        loaded = loaded &
                 _shader_programs[i].loadShader(shaders_filename[2 * i],
                                                GL_VERTEX_SHADER) &
                 _shader_programs[i].loadShader(shaders_filename[2 * i + 1],
                                                GL_FRAGMENT_SHADER) &
                 _shader_programs[i].link();
    }

    return loaded;
}

// Allocates OpenGL buffers to draw a quadrilateral
// and to store particles positions
void Renderer::_allocBuffers()
{
    // Vertices and indices used to draw a quad.
    const std::array<float, 12> 
    quad_vertices = {
        //  x,     y,    z
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f
    };

    const std::array<unsigned int, 6>
    quad_indices = {
        0, 1, 2,
        0, 3, 2
    };

   // Vertices and indices used to draw a cube outline.
    const std::array<float, 24>
    cube_vertices = {
        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f
    };

    const std::array<unsigned int, 24>
    cube_indices = {
        0, 1,
        1, 2,
        2, 3,
        3, 0,

        4, 5,
        5, 6,
        6, 7,
        7, 4,

        0, 7,
        1, 6,
        2, 5,
        3, 4
    };

    // Creating a Vertex Array Object to handle quad vertices
    glGenVertexArrays(1, &_quad_vao);
    glBindVertexArray(_quad_vao);
    // Creating a Vertex Buffer Object to store vertices data.
    GLuint quad_vbo;
    glGenBuffers(1, &quad_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
    // Copying vertices to GPU memory.
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(quad_vertices),
                 quad_vertices.data(),
                 GL_STATIC_DRAW);
    // Telling OpenGL to interpret the data stream as vertices of 3 floats
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);
    glEnableVertexAttribArray(0);

    // Creating a Element Buffer Object to store quad indices.
    GLuint quad_ebo;
    glGenBuffers(1, &quad_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad_ebo);
    // Copying indices to GPU memory.
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(quad_indices),
                 quad_indices.data(),
                 GL_STATIC_DRAW);

    glGenVertexArrays(1, &_cube_vao);
    glBindVertexArray(_cube_vao);

    GLuint cube_vbo;
    glGenBuffers(1, &cube_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, cube_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(cube_vertices),
                 cube_vertices.data(),
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);
    glEnableVertexAttribArray(0);

    GLuint cube_ebo;
    glGenBuffers(1, &cube_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(cube_indices),
                 cube_indices.data(),
                 GL_STATIC_DRAW);

    // Sample particles data
    /*
    const std::array<float, 12>
    particles_data = {
         0.5f, -0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.0f, 1.0f,
         0.0f,  0.5f, 0.0f, 1.0f,
    };
    */

    std::vector<float> dummy;
    dummy.reserve(N_POINTS);
    // Creating Shader Storage Buffer Objects to store particles data.
    for (int i = 0; i < 5; ++i) {
        glGenBuffers(1, &_particles_ssbo[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, _particles_ssbo[i]);
        // Copying particles data to GPU memory to define size
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _particles_ssbo[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     N_POINTS * sizeof(float),
                     dummy.data(),
                     GL_DYNAMIC_DRAW);
    }
}

// Sets OpenGL rendering options, arranges the scene
void Renderer::_setupScene()
{
    // Defines the rendering target area with
    // respect to the window coordinates
    glViewport(0, 0, _window_width, _window_height);
    // Sets clear color
    glClearColor(0.05f, 0.05f, 0.05f, 0.0f);
    // Enabling transparency
    glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    // Enabling depth testing
    // glEnable(GL_DEPTH_TEST);

    //_camera.setPosition({ 0.0f, 0.0f, -1.0f });
    _camera.setSphericalPosition({ 1.0f, 0.0f, M_PI / 2 });
    _camera.lookAt({ 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

    _camera.setOrbitMode(true);

    for (int i = 0; i < _NUM_SHADER_PROGRAMS; ++i) {
        _shader_programs[i].loadUniformMat4(
            "perspective_projection", _camera.getPerspectiveProjection());
    }

    _updateCamera();
}

// Handles keyboard and mouse inputs
void Renderer::_handleEvents()
{
    static int curr_node = 0;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        // ESC key press
        if ((event.type == SDL_WINDOWEVENT &&
             event.window.event == SDL_WINDOWEVENT_CLOSE) ||
            (event.type == SDL_KEYUP &&
             event.key.keysym.sym == SDLK_ESCAPE)
        ) {
            quit();
        } else if (event.type == SDL_MOUSEWHEEL) {
            // Zooming camera
            vec3 zoom_delta({ 0, 0, 0.1f * event.wheel.preciseY });
            // _camera.move(zoom_delta);
            _camera.orbit({ 0.005f * event.wheel.preciseY, 0, 0 });
        } else if (event.type == SDL_KEYUP &&
                   event.key.keysym.sym == SDLK_SPACE) {
            paused = !paused;
            /*
            _shader_programs[PARTICLE_SHADER].loadUniformInt(
                "selected_octree_node", ++curr_node);
            */
        }
    }

    // Handling mouse related events
    static int prev_mouse_x, prev_mouse_y;

    int mouse_x, mouse_y;
    Uint32 mouse_btn_state = SDL_GetMouseState(&mouse_x, &mouse_y);

    // Left click
    if (mouse_btn_state & SDL_BUTTON(1)) {
        // Normalize to canonical cube
        vec3 normalized_mouse_delta({
            2.0f / _window_width * (prev_mouse_x - mouse_x),
            2.0f / _window_height * (prev_mouse_y - mouse_y),
            0
        });

        //normalized_mouse_delta[0] = -0.03;
        //normalized_mouse_delta[1] = 0.0;

        // Translating camera
        // _camera.move(normalized_mouse_delta);

        if (true) {//!paused) {
            _camera.orbit({
                0, normalized_mouse_delta[1], normalized_mouse_delta[0] });
            _camera.update(0.0);
            _camera.lookAt({ 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
        }
    }

    prev_mouse_x = mouse_x;
    prev_mouse_y = mouse_y;
}

// Computes the time between two consecutive frames
void Renderer::_updateDeltaTime()
{
    static Uint64 prev_ticks = 0;

    Uint64 curr_ticks = SDL_GetTicks64();
    _delta_time = static_cast<float>(curr_ticks - prev_ticks) / 1000;
    prev_ticks = curr_ticks;
}

void Renderer::_updateCamera()
{
    _camera.update(_delta_time);

    for (int i = 0; i < _NUM_SHADER_PROGRAMS; ++i) {
        // Updating the GLSL world to camera transformation matrix
        _shader_programs[i].loadUniformMat4(
            "world_to_camera", _camera.getWorldToCamera());
    }
}

// Renders a single frame to the window
void Renderer::_renderFrame()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _shader_programs[PARTICLE_SHADER].enable();
    glBindVertexArray(_quad_vao);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, N_POINTS);

    _shader_programs[CUBE_SHADER].enable();
    glBindVertexArray(_cube_vao);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);

    SDL_GL_SwapWindow(_window);
}

Renderer::~Renderer()
{
    SDL_DestroyWindow(_window);
    SDL_Quit();
}
