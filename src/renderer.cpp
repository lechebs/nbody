#include "renderer.hpp"

#include <iostream>
#include <fstream>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <memory>
#include <string>

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "vector.hpp"
#include "camera.hpp"
#include "shader_program.hpp"
#include "simulation.hpp"

#define PRECISION float
#define STR_HELPER(x) #x
#define TO_STR(x) STR_HELPER(x)

constexpr unsigned int N_POINTS = 1 << 17;

using vec3f = Vector<float, 3>;
using vec3d = Vector<double, 3>;

Renderer::Renderer(unsigned int window_width,
                   unsigned int window_height) :
    window_width_(window_width),
    window_height_(window_height),
    window_title_("nbody"),
    camera_(M_PI / 3,
            static_cast<float>(window_width) / window_height,
            -0.001,
            -20.0) {}

bool Renderer::init()
{
    return initialized_ = init_() && load_shaders();
}

void Renderer::run()
{
    assert(initialized_);

    alloc_buffers();
    setup_scene();

    Simulation<PRECISION>::Params p;
    p.num_points = N_POINTS;
    p.max_num_codes_per_leaf = 32;
    p.theta = 0.5;
    p.dt = 1e-4;
    p.gravity = 1.0;
    p.softening_factor = 0.001;
    p.velocity_dampening = 0.0;
    p.domain_size = 1.0;
    p.num_steps_validator = 0;
    p.mem_traversal_queues = (1 << 20) * 50;
    p.traversal_group_size = 32;

    num_points_ = p.num_points;

    shader_programs_[PARTICLE_SHADER].load_uniform_float("domain_size",
                                                         p.domain_size);
    shader_programs_[OCTREE_SHADER].load_uniform_float("domain_size",
                                                       p.domain_size);

    Simulation<PRECISION> simulation(p, ssbos_);
    simulation.spawn_points();


    while (running_) {
        handle_events();
        update_delta_time();

        if (!paused_) {
            simulation.update();
            num_octree_nodes_ = simulation.get_num_octree_nodes();
        }

        // Octree from previous iteration is drawn

        update_camera();
        render_frame();
    }

    if (p.num_steps_validator > 0) {
        simulation.write_validation_history(
            std::string() + "output-" + TO_STR(PRECISION) + "-0" +
            std::to_string((int) (p.theta * 10)) + ".csv");
        }
}

void Renderer::quit()
{
    running_ = false;
}

// Initializes the SDL window and the attached OpenGL context
bool Renderer::init_()
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
    if ((window_ = SDL_CreateWindow(
        window_title_.c_str(),
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        window_width_,
        window_height_,
        SDL_WINDOW_OPENGL)) == nullptr) {
        std::cout << SDL_GetError() << std::endl;
        return false;
    }

    // Creating OpenGL context.
    if (SDL_GL_CreateContext(window_) == nullptr) {
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
bool Renderer::load_shaders()
{
    const std::array<std::string, 6> shaders_filename = {
        "shaders/particle.vert",
        "shaders/particle.frag",
        "shaders/cube.vert",
        "shaders/cube.frag",
        "shaders/octree.vert",
        "shaders/octree.frag"
    };

    bool loaded = true;

    for (int i = 0; i < NUM_SHADER_PROGRAMS_; ++i) {
        // Needs to be called after OpenGL context creation
        shader_programs_[i].create();

        loaded = loaded &
                 shader_programs_[i].load_shader(shaders_filename[2 * i],
                                                 GL_VERTEX_SHADER,
                                                 TO_STR(PRECISION)) &
                 shader_programs_[i].load_shader(shaders_filename[2 * i + 1],
                                                 GL_FRAGMENT_SHADER,
                                                 TO_STR(PRECISION)) &
                 shader_programs_[i].link();
    }

    return loaded;
}

// Allocates OpenGL buffers to draw a quadrilateral
// and to store particles positions
void Renderer::alloc_buffers()
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
    glGenVertexArrays(1, &quad_vao_);
    glBindVertexArray(quad_vao_);
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

    glGenVertexArrays(1, &cube_vao_);
    glBindVertexArray(cube_vao_);

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

    std::vector<double> dummy;
    dummy.reserve(2 * N_POINTS); // Needed for octree nodes as well
    // Creating Shader Storage Buffer Objects to store simulation data.
    for (int i = 0; i < NUM_SSBOS_; ++i) {
        glGenBuffers(1, &ssbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, ssbos_[i]);
        // Copying particles data to GPU memory to define size
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos_[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     2 * N_POINTS * sizeof(double),
                     dummy.data(),
                     GL_DYNAMIC_DRAW);
    }
}

// Sets OpenGL rendering options, arranges the scene
void Renderer::setup_scene()
{
    // Defines the rendering target area with
    // respect to the window coordinates
    glViewport(0, 0, window_width_, window_height_);
    // Sets clear color
    glClearColor(0.05f, 0.05f, 0.05f, 0.0f);
    // Enabling additive blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    camera_.set_spherical_position({ 1.0f, 0.0f, M_PI / 2 });

    camera_.set_orbit_mode(true);
    camera_.set_orbit_mode_center({ 0.0f, 0.0f, 0.0f });

    for (int i = 0; i < NUM_SHADER_PROGRAMS_; ++i) {
        shader_programs_[i].load_uniform_mat4(
            "perspective_projection", camera_.get_perspective_projection());
    }

    update_camera();
}

// Handles keyboard and mouse inputs
void Renderer::handle_events()
{
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
            camera_.orbit({ 0.01f * event.wheel.preciseY, 0, 0 });

        } else if (event.type == SDL_KEYUP &&
                   event.key.keysym.sym == SDLK_SPACE) {
            paused_ = !paused_;

        } else if (event.type == SDL_KEYUP &&
                   event.key.keysym.sym == SDLK_d) {
            draw_domain_ = !draw_domain_;

        } else if (event.type == SDL_KEYUP &&
                   event.key.keysym.sym == SDLK_o) {
            draw_octree_ = !draw_octree_;

            if (!draw_octree_) {
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            } else {
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            }
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
            2.0f / window_width_ * (prev_mouse_x - mouse_x),
            2.0f / window_height_ * (prev_mouse_y - mouse_y),
            0
        });

        camera_.orbit({
            0, normalized_mouse_delta[1], normalized_mouse_delta[0] });
    }

    prev_mouse_x = mouse_x;
    prev_mouse_y = mouse_y;
}

// Computes the time between two consecutive frames
void Renderer::update_delta_time()
{
    static Uint64 prev_ticks = 0;

    Uint64 curr_ticks = SDL_GetTicks64();
    delta_time_ = static_cast<float>(curr_ticks - prev_ticks) / 1000;
    prev_ticks = curr_ticks;
}

void Renderer::update_camera()
{
    camera_.update(delta_time_);

    for (int i = 0; i < NUM_SHADER_PROGRAMS_; ++i) {
        // Updating the GLSL world to camera transformation matrix
        shader_programs_[i].load_uniform_mat4(
            "world_to_camera", camera_.get_world_to_camera());
    }
}

// Renders a single frame to the window
void Renderer::render_frame()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader_programs_[PARTICLE_SHADER].enable();
    glBindVertexArray(quad_vao_);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, num_points_);

    // Drawing octree
    if (draw_octree_) {
       shader_programs_[OCTREE_SHADER].enable();
        glBindVertexArray(cube_vao_);
        glDrawElementsInstanced(
            GL_LINES, 24, GL_UNSIGNED_INT, 0, num_octree_nodes_);
    }

    if (draw_domain_) {
        shader_programs_[CUBE_SHADER].enable();
        glBindVertexArray(cube_vao_);
        glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
    }

    SDL_GL_SwapWindow(window_);
}

Renderer::~Renderer()
{
    SDL_DestroyWindow(window_);
    SDL_Quit();
}
