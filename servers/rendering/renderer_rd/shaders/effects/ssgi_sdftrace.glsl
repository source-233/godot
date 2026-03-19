#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define MAX_CASCADES 8

layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 projection[2];
	mat4 inv_projection[2];
	mat4 reprojection[2];
	mat4 inv_view_matrix;
	mat4 view_matrix;
	vec4 eye_offset[2];
}
scene_data;

layout(set = 0, binding = 1) uniform texture3D sdf_cascades[MAX_CASCADES];
layout(set = 0, binding = 2) uniform texture3D light_cascades[MAX_CASCADES];
layout(set = 0, binding = 3) uniform texture3D aniso0_cascades[MAX_CASCADES];
layout(set = 0, binding = 4) uniform texture3D aniso1_cascades[MAX_CASCADES];
layout(set = 0, binding = 5) uniform texture3D occlusion_texture;

layout(set = 0, binding = 7) uniform sampler2D source_hiz;
layout(set = 0, binding = 8) uniform sampler linear_sampler;

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 9, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

layout(rgba16f, set = 0, binding = 10) uniform restrict image2D output_color;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	ivec2 compute_size;
	float intensity;
	int view_index;
	uint frame_count;

	float y_mult;
	vec3 grid_size;
	uint max_cascades;
}
params;

#include "restir_inc.glsl"

vec3 screen_to_view_pos(vec3 screen_pos) {
	vec4 pos;
	pos.xy = screen_pos.xy * 2.0 - 1.0;
	pos.z = screen_pos.z;
	pos.w = 1.0;
	pos = scene_data.inv_projection[params.view_index] * pos;
	return pos.xyz / pos.w;
}

vec3 view_to_world_pos(vec3 pos) {
	return (scene_data.inv_view_matrix * vec4(pos, 1.0)).xyz;
}

vec3 screen_to_world_pos(vec3 screen_pos) {
	return view_to_world_pos(screen_to_view_pos(screen_pos));
}

vec3 linear_to_srgb(vec3 color) {
	//if going to srgb, clamp from 0 to 1.
	color = clamp(color, vec3(0.0), vec3(1.0));
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint hash(uint value) {
	uint state = value * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

uint random_seed(ivec3 seed) {
	return hash(seed.x ^ hash(seed.y ^ hash(seed.z)));
}

// generates a random value in range [0.0, 1.0)
float random_float(inout uint value) {
	value = hash(value);
	return float(value / 4294967296.0);
}

float luminance(vec3 color) {
	return dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
}

const int kNumSamples = 4;
const ivec2 kOffsets2x2[4] = {
	ivec2( 0, 0),
	ivec2( 0, 1),
	ivec2( 1, 1),
	ivec2( 1, 0),
};
ivec2 get_jitter_offset(uint idx) {
	if (any(notEqual(params.screen_size, params.compute_size))) {
		return kOffsets2x2[idx % kNumSamples];
	}
	return kOffsets2x2[0];
}

void main() {
	ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);
	vec4 ray = imageLoad(output_color, pixel_pos).xyzw;
	imageStore(output_color, pixel_pos, vec4(0.0));

	if (any(greaterThanEqual(pixel_pos, params.compute_size))) { //too large, do nothing
		return;
	}

	Reservoir pre_reservoir = reservoirs.data[reservoir_index(pixel_pos, reservoirs_setting.reservoir_size)];
	if (pre_reservoir.hsample.pdf > 0.2) {
		return;
	}

	ivec2 screen_coord = pixel_pos * (params.screen_size / params.compute_size);
	screen_coord += get_jitter_offset(params.frame_count);
	vec3 screen_pos;
	screen_pos.xy = vec2(screen_coord + 0.5) / params.screen_size;
	screen_pos.z = texelFetch(source_hiz, screen_coord, 0).x;

	if (screen_pos.z <= 0.0) {
		return;
	}

	vec3 world_pos = screen_to_world_pos(screen_pos);
	vec3 ray_pos = world_pos;

	vec3 ray_dir = ray.xyz;
	vec3 inv_dir = 1.0 / ray_dir;

	//this is how to properly bias outgoing rays
	float cell_size = 1.0 / cascades.data[0].to_cell;
	ray_pos += sign(ray_dir) * cell_size; // go almost to the box edge but remain inside
	ray_pos += ray_dir * 0.4 * cell_size; //apply a small bias from there

	vec3 hit_normal = vec3(0.0);
	vec4 light = vec4(0.0);

	bool hit = false;
	uint hit_cascade = 0;
	vec3 uvw;
	vec3 hit_position = ray_pos;
	vec3 pos_to_uvw = 1.0 / params.grid_size;
	vec3 val = vec3(0.0);

	ray_pos.y *= params.y_mult;
	ray_dir.y *= params.y_mult;
	ray_dir = normalize(ray_dir);

	// Generate noise seed
	// uint noise_seed = random_seed(ivec3(screen_coord, params.frame_count));
	uint noise_seed = pre_reservoir.noise_seed;

	for (uint j = 0; j < params.max_cascades; j++) {
		//convert to local bounds
		vec3 pos = ray_pos - cascades.data[j].offset;
		pos *= cascades.data[j].to_cell;

		if (any(lessThan(pos, vec3(0.0))) || any(greaterThanEqual(pos, params.grid_size))) {
			//this is how to properly bias outgoing rays
			float cell_size = 1.0 / cascades.data[j].to_cell;
			ray_pos += sign(ray_dir) * cell_size; // go almost to the box edge but remain inside
			ray_pos += ray_dir * 0.4 * cell_size; //apply a small bias from there
			continue; //already past bounds for this cascade, goto next
		}

		//find maximum advance distance (until reaching bounds)
		vec3 t0 = -pos * inv_dir;
		vec3 t1 = (params.grid_size - pos) * inv_dir;
		vec3 tmax = max(t0, t1);
		float max_advance = min(tmax.x, min(tmax.y, tmax.z));

		float advance = 0.0;

		while (advance < max_advance) {
			//read how much to advance from SDF
			uvw = (pos + ray_dir * advance) * pos_to_uvw;

			float distance = texture(sampler3D(sdf_cascades[j], linear_sampler), uvw).r * 255.0 - 1.0;
			if (distance < 0.05) {
				//consider hit
				hit = true;
				hit_cascade = j;
				hit_position = ray_pos + ray_dir * advance;
				break;
			}

			advance += distance;
		}

		if (hit) {
			//avoid reading different texture from different threads
			const float EPSILON = 0.001;
			hit_normal = normalize(vec3(
					texture(sampler3D(sdf_cascades[hit_cascade], linear_sampler), uvw + vec3(EPSILON, 0.0, 0.0)).r - texture(sampler3D(sdf_cascades[hit_cascade], linear_sampler), uvw - vec3(EPSILON, 0.0, 0.0)).r,
					texture(sampler3D(sdf_cascades[hit_cascade], linear_sampler), uvw + vec3(0.0, EPSILON, 0.0)).r - texture(sampler3D(sdf_cascades[hit_cascade], linear_sampler), uvw - vec3(0.0, EPSILON, 0.0)).r,
					texture(sampler3D(sdf_cascades[hit_cascade], linear_sampler), uvw + vec3(0.0, 0.0, EPSILON)).r - texture(sampler3D(sdf_cascades[hit_cascade], linear_sampler), uvw - vec3(0.0, 0.0, EPSILON)).r));

			vec3 hit_light = texture(sampler3D(light_cascades[hit_cascade], linear_sampler), uvw).rgb;
			vec4 aniso0 = texture(sampler3D(aniso0_cascades[hit_cascade], linear_sampler), uvw);
			vec3 hit_aniso0 = aniso0.rgb;
			vec3 hit_aniso1 = vec3(aniso0.a, texture(sampler3D(aniso1_cascades[hit_cascade], linear_sampler), uvw).rg);

			//one liner magic
			light.rgb = hit_light * (dot(max(vec3(0.0), (hit_normal * hit_aniso0)), vec3(1.0)) + dot(max(vec3(0.0), (-hit_normal * hit_aniso1)), vec3(1.0)));
			light.a = 1.0;
			break;
		}

		//change ray origin to collision with bounds
		pos += ray_dir * max_advance;
		pos /= cascades.data[j].to_cell;
		pos += cascades.data[j].offset;
		ray_pos = pos;
	}

	// save reservoirs
	HitSample hit_sample;
	hit_sample.ray_direction = ray_dir;
	hit_sample.distance = length(hit_position - world_pos);
	hit_sample.hit_normal = hit_normal;
	hit_sample.out_radiance = light.rgb * params.intensity;
	hit_sample.pdf = luminance(light.rgb);
	hit_sample.proposal_pdf = ray.w;

	Reservoir reservoir = new_reservoir();
	add_sample_to_reservoir(reservoir, hit_sample, ray.w, random_float(noise_seed));
	reservoir.noise_seed = noise_seed;

	reservoirs.data[reservoir_index(pixel_pos, reservoirs_setting.reservoir_size)] = reservoir;

	imageStore(output_color, pixel_pos, vec4(light.rgb, 1.0));
}