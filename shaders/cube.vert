#version 430 core

layout (location = 0) in vec3 vert_pos;

uniform mat4 world_to_camera;
uniform mat4 perspective_projection;

void main()
{
    float scale = 0.5;
    vec4 eye_pos = world_to_camera * vec4(vert_pos * scale, 1.0f);

    gl_Position = perspective_projection * eye_pos;
}
