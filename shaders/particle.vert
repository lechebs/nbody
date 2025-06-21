#version 430 core

layout (location = 0) in vec3 vert_pos;
// SSBOs used to gather particles data
layout (std430, binding = 0) readonly buffer PartX { FTYPE_ particles_x[]; };
layout (std430, binding = 1) readonly buffer PartY { FTYPE_ particles_y[]; };
layout (std430, binding = 2) readonly buffer PartZ { FTYPE_ particles_z[]; };

out vec2 frag_uv;
out vec4 particle_eye_pos;
flat out int is_selected;
flat out int particle_id;
flat out float particle_radius;

uniform float domain_size;
uniform int selected_octree_node;
uniform mat4 world_to_camera;
uniform mat4 perspective_projection;

void main()
{
    particle_id = gl_InstanceID;
    // Forwards xy position to allow interpolation
    frag_uv = vert_pos.xy;

    // Rescale to unit cube and translate to center
    vec3 particle_world_pos = vec3(float(particles_x[gl_InstanceID]),
                                   float(particles_y[gl_InstanceID]),
                                   float(particles_z[gl_InstanceID])) / domain_size;

    particle_world_pos -= vec3(0.5f, 0.5f, 0.5f);

    // Determines the size of the particle
    particle_radius = 0.003;
    // Position in camera coordinates
    particle_eye_pos = world_to_camera * vec4(particle_world_pos, 1.0f);

    gl_Position = perspective_projection *
                  (vec4(particle_radius * vert_pos, 1.0f) + particle_eye_pos);
}
