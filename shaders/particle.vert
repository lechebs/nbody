#version 430 core

#define RADIUS 0.01f

struct Particle {
	vec3 position;
	float mass;
};

layout (location = 0) in vec3 vert_pos;
// SSBO used to gather particles data
layout (std430, binding = 0) buffer Particles {
	Particle particles[];
};

out vec2 frag_uv;

uniform mat4 world_to_camera;
uniform mat4 perspective_projection;

void main()
{
	// Forwards xy position to allow interpolation
	frag_uv = vert_pos.xy;

	// Determines the size of the particle
	float scale = min(RADIUS * particles[gl_InstanceID].mass, 0.1f);
	// Position in world coordinates
	vec3 world_translate = particles[gl_InstanceID].position;

	gl_Position = perspective_projection *
				  world_to_camera *
				  vec4(scale * vert_pos + world_translate, 1.0f);
}
