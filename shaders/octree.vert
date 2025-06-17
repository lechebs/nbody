#version 430 core

layout (location = 0) in vec3 vert_pos;
layout (std430, binding = 3) readonly buffer BuffX { FTYPE_ coms_x[]; };
layout (std430, binding = 4) readonly buffer BuffY { FTYPE_ coms_y[]; };
layout (std430, binding = 5) readonly buffer BuffZ { FTYPE_ coms_z[]; };
layout (std430, binding = 6) readonly buffer BuffSize { FTYPE_ size[]; };

uniform float domain_size;
uniform mat4 world_to_camera;
uniform mat4 perspective_projection;
uniform int max_node_id;

void main()
{
    int node_id = gl_InstanceID;
    FTYPE_ node_size = size[node_id];

    float scale = float(node_size * 0.5);

    float i = floor(float(coms_x[node_id] / node_size));
    float j = floor(float(coms_y[node_id] / node_size));
    float k = floor(float(coms_z[node_id] / node_size));

    vec3 delta = vec3(i, j, k) * float(node_size / domain_size) +
        vec3(scale / domain_size - 0.5f,
             scale / domain_size - 0.5f,
             scale / domain_size - 0.5f);

    vec4 eye_pos = world_to_camera * vec4(vec3(vert_pos * scale / domain_size + delta), 1.0f);

    gl_Position = perspective_projection * eye_pos;
}

