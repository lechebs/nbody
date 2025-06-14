#version 430 core

// layout (depth_less) out float gl_FragDepth;

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

    /*
    vec3 normal = vec3(frag_uv, sqrt(1.0f - r));
    vec4 frag_eye_pos = particle_eye_pos +
                        vec4(normal * particle_radius, 0.0f);

    vec4 frag_clip_pos = perspective_projection * frag_eye_pos;

    gl_FragDepth = frag_clip_pos.z / frag_clip_pos.w;
    */

    vec4 color = vec4(0.9f, 0.3f, 0.2f, 1.0f);

    // Simple sphere shading
    float falloff = (1.0f - r) * (1.0f - r) * (1.0f - r);
    frag_color = color * vec4(1.0f, 1.0f, 1.0f, falloff * 1f);
}
