#version 430 core

struct Particle {
    vec3 position;
    float mass;
};

layout (location = 0) in vec3 vert_pos;
// SSBOs used to gather particles data
layout (std430, binding = 0) readonly buffer BufferX { float particles_x[]; };
layout (std430, binding = 1) readonly buffer BufferY { float particles_y[]; };
layout (std430, binding = 2) readonly buffer BufferZ { float particles_z[]; };
layout (std430, binding = 3) readonly buffer OctreeBegin
{
    int octree_begin[];
};
layout (std430, binding = 4) readonly buffer OctreeEnd
{
    int octree_end[];
};

out vec2 frag_uv;
out vec4 particle_eye_pos;
flat out int is_selected;
flat out int particle_id;
flat out float particle_radius;

uniform int selected_octree_node;
uniform mat4 world_to_camera;
uniform mat4 perspective_projection;

void main()
{
    particle_id = gl_InstanceID;
    is_selected = int(particle_id >= octree_begin[selected_octree_node] &&
                      particle_id <= octree_end[selected_octree_node]);
    // Forwards xy position to allow interpolation
    frag_uv = vert_pos.xy;

    vec3 particle_world_pos = vec3(particles_x[gl_InstanceID],
                                   particles_y[gl_InstanceID],
                                   particles_z[gl_InstanceID]);

    particle_world_pos -= vec3(0.5f, 0.5f, 0.5f);

    // Determines the size of the particle
    particle_radius = 0.005;
    // Position in camera coordinates
    particle_eye_pos = world_to_camera * vec4(particle_world_pos, 1.0f);

    gl_Position = perspective_projection *
                  (vec4(particle_radius * vert_pos, 1.0f) + particle_eye_pos);
}
