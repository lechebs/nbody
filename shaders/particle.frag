#version 430 core

in vec2 frag_uv;
in vec4 particle_eye_pos;
flat in int is_selected;
flat in int particle_id;
flat in float particle_radius;

out vec4 frag_color;

uniform mat4 perspective_projection;

void main()
{
    float r = dot(frag_uv, frag_uv);
    if (r > 1.0f) {
        discard;
    }

    vec4 color = vec4(0.9f, 0.3f, 0.2f, 1.0f);

    // Simple sphere shading
    float falloff = 1.0f - r;
    frag_color = color * vec4(1.0f, 1.0f, 1.0f, falloff * 0.08f);
}
