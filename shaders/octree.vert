#version 430 core

layout (location = 0) in vec3 vert_pos;
layout (std430, binding = 3) readonly buffer BuffX { double barycenters_x[]; };
layout (std430, binding = 4) readonly buffer BuffY { double barycenters_y[]; };
layout (std430, binding = 5) readonly buffer BuffZ { double barycenters_z[]; };
layout (std430, binding = 6) readonly buffer BuffSize { double nodes_size[]; };

uniform mat4 world_to_camera;
uniform mat4 perspective_projection;
uniform int max_node_id;

void main()
{
    int node_id = gl_InstanceID;
    double node_size = nodes_size[node_id];
    /*
    if (node_size >= 0.25f) {
        return;
    }
    */

    double scale = node_size * 0.5;

    double i = floor(double(barycenters_x[node_id]) / node_size);
    double j = floor(double(barycenters_y[node_id]) / node_size);
    double k = floor(double(barycenters_z[node_id]) / node_size);

    dvec3 delta = dvec3(i, j, k) * node_size + 
        dvec3(scale, scale, scale) - dvec3(0.5, 0.5, 0.5);

    vec4 eye_pos = world_to_camera * vec4(vec3(vert_pos * scale + delta), 1.0f);

    gl_Position = perspective_projection * eye_pos;
}

