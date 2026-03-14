#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 projection[2];
	mat4 inv_projection[2];
	mat4 reprojection[2];
	vec4 eye_offset[2];
	mat4 inv_view_matrix;
	mat4 view_matrix;
}
scene_data;

layout(set = 0, binding = 1) uniform sampler2D source_last_frame;
layout(set = 0, binding = 2) uniform sampler2D source_hiz;
layout(set = 0, binding = 3) uniform sampler2D source_normal_roughness;

layout(rgba16f, set = 0, binding = 4) uniform restrict writeonly image2D output_color;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int mipmaps;
	int num_steps;
	float depth_tolerance;
	float intensity;
	int view_index;
	float frame_count;
}
params;

#include "restir_inc.glsl"

vec2 compute_cell_count(int level) {
	int cell_count_x = max(1, params.screen_size.x >> level);
	int cell_count_y = max(1, params.screen_size.y >> level);
	return vec2(cell_count_x, cell_count_y);
}

float linearize_depth(float depth) {
	vec4 pos = vec4(0.0, 0.0, depth, 1.0);
	pos = scene_data.inv_projection[params.view_index] * pos;
	return pos.z / pos.w;
}

vec3 screen_to_view_pos(vec3 screen_pos) {
	vec4 pos;
	pos.xy = screen_pos.xy * 2.0 - 1.0;
	pos.z = screen_pos.z;
	pos.w = 1.0;
	pos = scene_data.inv_projection[params.view_index] * pos;
	return pos.xyz / pos.w;
}

vec3 view_to_screen_pos(vec3 pos) {
	vec4 screen_pos = scene_data.projection[params.view_index] * vec4(pos, 1.0);
	screen_pos.xyz /= screen_pos.w;
	screen_pos.xy = screen_pos.xy * 0.5 + 0.5;
	return screen_pos.xyz;
}

// https://habr.com/ru/articles/744336/
vec3 compute_geometric_normal(ivec2 pixel_pos, float depth_c, vec3 view_c, float pixel_offset) {
	vec4 H = vec4(
			texelFetch(source_hiz, pixel_pos + ivec2(-1, 0), 0).x,
			texelFetch(source_hiz, pixel_pos + ivec2(-2, 0), 0).x,
			texelFetch(source_hiz, pixel_pos + ivec2(1, 0), 0).x,
			texelFetch(source_hiz, pixel_pos + ivec2(2, 0), 0).x);

	vec4 V = vec4(
			texelFetch(source_hiz, pixel_pos + ivec2(0, -1), 0).x,
			texelFetch(source_hiz, pixel_pos + ivec2(0, -2), 0).x,
			texelFetch(source_hiz, pixel_pos + ivec2(0, 1), 0).x,
			texelFetch(source_hiz, pixel_pos + ivec2(0, 2), 0).x);

	vec2 he = abs((2.0 * H.xz - H.yw) - depth_c);
	vec2 ve = abs((2.0 * V.xz - V.yw) - depth_c);

	int h_sign = he.x < he.y ? -1 : 1;
	int v_sign = ve.x < ve.y ? -1 : 1;

	vec3 view_h = screen_to_view_pos(vec3((pixel_pos + vec2(h_sign, 0) + pixel_offset) / params.screen_size, H[1 + int(h_sign)]));
	vec3 view_v = screen_to_view_pos(vec3((pixel_pos + vec2(0, v_sign) + pixel_offset) / params.screen_size, V[1 + int(v_sign)]));

	vec3 h_der = h_sign * (view_h - view_c);
	vec3 v_der = v_sign * (view_v - view_c);

	return cross(v_der, h_der);
}

vec3 reprojection(vec2 uv, float depth) {
	vec4 previous_pos_ndc = scene_data.reprojection[params.view_index] * vec4(uv * 2.0f - 1.0f, depth, 1.0f);
	return vec3((previous_pos_ndc.xy / previous_pos_ndc.w) * 0.5f + 0.5f, previous_pos_ndc.z / previous_pos_ndc.w);
}

vec3 view_to_world_pos(vec3 pos) {
	return (scene_data.inv_view_matrix * vec4(pos, 1.0)).xyz;
}

vec3 view_to_world_normal(vec3 normal) {
    mat3 inv_view_matrix_basis = mat3(scene_data.inv_view_matrix);
    vec3 world_normal = normalize(inv_view_matrix_basis * normal);
    return world_normal;
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

const float PI = 3.14159265f;

// http://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.4.pdf (chapter 15)
vec4 generate_hemisphere_cosine_weighted_direction(inout uint noise) {
	float noise1 = random_float(noise);
	float noise2 = random_float(noise) * 2.0 * PI;

	vec3 h;
	h.x = sqrt(noise1) * cos(noise2);
	h.y = sqrt(noise1) * sin(noise2);
	h.z = sqrt(1.0 - noise1);

	float pdf = h.z * (1.0 / PI);

	return vec4(h, pdf);
}

vec4 generate_ray_dir_from_normal(vec3 normal, inout uint noise) {
	vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
	vec3 tangent = normalize(cross(v0, normal));
	vec3 bitangent = normalize(cross(tangent, normal));
	mat3 normal_mat = mat3(tangent, bitangent, normal);
	vec4 dir = generate_hemisphere_cosine_weighted_direction(noise);
	return vec4(normal_mat * dir.xyz, dir.w);
}

float luminance(vec3 color) {
	return dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
	ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(pixel_pos, params.screen_size))) {
		return;
	}

	vec4 color = vec4(0.0);
	float mip_level = 0.0;

	vec3 screen_pos;
	screen_pos.xy = vec2(pixel_pos + 0.5) / params.screen_size;
	screen_pos.z = texelFetch(source_hiz, pixel_pos, 0).x;

	bool should_trace = screen_pos.z != 0.0;
	if (should_trace) {
		vec3 pos = screen_to_view_pos(screen_pos);

		vec4 normal_roughness = texelFetch(source_normal_roughness, pixel_pos, 0);
		vec3 normal = normalize(normal_roughness.xyz * 2.0 - 1.0);
		float roughness = normal_roughness.w;
		if (roughness > 0.5) {
			roughness = 1.0 - roughness;
		}
		roughness /= (127.0 / 255.0);

		vec3 geom_normal = normalize(compute_geometric_normal(pixel_pos, screen_pos.z, pos, 0.5));

		// Add a small bias towards the geometry normal to prevent self intersections.
		pos += geom_normal * (1.0 - pow(clamp(dot(normal, geom_normal), 0.0, 1.0), 8.0));
		screen_pos = view_to_screen_pos(pos);

		uint noise_seed = random_seed(ivec3(pixel_pos, params.frame_count));
		vec4 ray_dir = generate_ray_dir_from_normal(normal, noise_seed);

		// Check if the ray is immediately intersecting with itself. If so, bounce!
		if (dot(ray_dir.xyz, geom_normal) < 0.0) {
			ray_dir.xyz = normalize(reflect(ray_dir.xyz, geom_normal));
		}

		vec3 end_pos = pos + ray_dir.xyz;

		// Clip to near plane. Add a small bias so we don't go to infinity.
		if (end_pos.z > 0.0) {
			end_pos -= ray_dir.xyz / ray_dir.z * (end_pos.z + 0.00001);
		}

		vec3 screen_end_pos = view_to_screen_pos(end_pos);

		// Normalize Z to -1.0 or +1.0 and do parametric T tracing as suggested here:
		// https://hacksoflife.blogspot.com/2020/10/a-tip-for-hiz-ssr-parametric-t-tracing.html
		vec3 screen_ray_dir = screen_end_pos - screen_pos;
		screen_ray_dir /= abs(screen_ray_dir.z);

		bool facing_camera = screen_ray_dir.z >= 0.0;

		// Find the screen edge point where we will stop tracing.
		vec2 t0 = (vec2(0.0) - screen_pos.xy) / screen_ray_dir.xy;
		vec2 t1 = (vec2(1.0) - screen_pos.xy) / screen_ray_dir.xy;
		vec2 t2 = max(t0, t1);
		float t_max = min(t2.x, t2.y);

		vec2 cell_step = vec2(screen_ray_dir.x < 0.0 ? -1.0 : 1.0, screen_ray_dir.y < 0.0 ? -1.0 : 1.0);

		int cur_level = 0;
		int cur_iteration = params.num_steps;

		// Advance the start point to the closest next cell to prevent immediate self intersection.
		float t;
		{
			vec2 cell_index = floor(screen_pos.xy * params.screen_size);
			vec2 new_cell_index = cell_index + clamp(cell_step, vec2(0.0), vec2(1.0));
			vec2 new_cell_pos = (new_cell_index / params.screen_size) + cell_step * 0.000001;
			vec2 pos_t = (new_cell_pos - screen_pos.xy) / screen_ray_dir.xy;
			float edge_t = min(pos_t.x, pos_t.y);

			t = edge_t;
		}

		while (cur_level >= 0 && cur_iteration > 0 && t < t_max) {
			vec3 cur_screen_pos = screen_pos + screen_ray_dir * t;

			vec2 cell_count = compute_cell_count(cur_level);
			vec2 cell_index = floor(cur_screen_pos.xy * cell_count);
			float cell_depth = texelFetch(source_hiz, ivec2(cell_index), cur_level).x;
			float depth_t = (cell_depth - screen_pos.z) * screen_ray_dir.z; // Z is either -1.0 or 1.0 so we don't need to do a divide.

			vec2 new_cell_index = cell_index + clamp(cell_step, vec2(0.0), vec2(1.0));
			vec2 new_cell_pos = (new_cell_index / cell_count) + cell_step * 0.000001;
			vec2 pos_t = (new_cell_pos - screen_pos.xy) / screen_ray_dir.xy;
			float edge_t = min(pos_t.x, pos_t.y);

			bool hit = facing_camera ? (t <= depth_t) : (depth_t <= edge_t);
			int mip_offset = hit ? -1 : +1;

			if (cur_level == 0) {
				float z0 = linearize_depth(cell_depth);
				float z1 = linearize_depth(cur_screen_pos.z);

				if ((z0 - z1) > params.depth_tolerance) {
					hit = false;
					mip_offset = 0; // Keep the mip index the same to prevent it from decreasing and increasing in repeat.
				}
			}

			if (hit) {
				if (!facing_camera) {
					t = max(t, depth_t);
				}
			} else {
				t = edge_t;
			}

			cur_level = min(cur_level + mip_offset, params.mipmaps - 1);
			--cur_iteration;
		}

		vec3 cur_screen_pos = screen_pos + screen_ray_dir * t;

		vec3 reprojected_pos = reprojection(cur_screen_pos.xy, cur_screen_pos.z);

		// Instead of hard rejecting samples, write sample validity to the alpha channel.
		// This allows invalid samples to write mip levels to let valid samples have smoother roughness transitions.
		float validity = 1.0;

		// Hit validation logic is referenced from here:
		// https://github.com/GPUOpen-Effects/FidelityFX-SSSR/blob/master/ffx-sssr/ffx_sssr.h

		ivec2 cur_pixel_pos = ivec2(cur_screen_pos.xy * params.screen_size);

		float hit_depth = texelFetch(source_hiz, cur_pixel_pos, 0).x;
		if (t >= t_max || hit_depth == 0.0) {
			validity = 0.0;
		}

		vec3 hit_normal = texelFetch(source_normal_roughness, cur_pixel_pos, 0).xyz * 2.0 - 1.0;

		if (all(lessThan(abs(screen_ray_dir.xy * t), 2.0 / params.screen_size))) { //自相交
			if (dot(ray_dir.xyz, hit_normal) >= 0.0) {
				validity = 0.0;
			}
		}

		vec3 cur_pos = screen_to_view_pos(cur_screen_pos); //这里需要处理超出屏幕的情况
		vec3 hit_pos = screen_to_view_pos(vec3(cur_screen_pos.xy, hit_depth));

		float delta = length(cur_pos - hit_pos);
		float confidence = 1.0 - smoothstep(0.0, params.depth_tolerance, delta);
		validity *= clamp(confidence * confidence, 0.0, 1.0);

		// save validity to alpha channel
		color = vec4(textureLod(source_last_frame, reprojected_pos.xy, 0).xyz, validity);

		// save reservoirs
		HitSample hit_sample;
		hit_sample.ray_direction = view_to_world_normal(ray_dir.xyz);
		hit_sample.distance = length(cur_pos - pos);
		hit_sample.hit_normal = view_to_world_normal(hit_normal);
		hit_sample.out_radiance = color.rgb *= params.intensity;
		hit_sample.pdf = luminance(color.rgb) * validity;
		hit_sample.validity = validity;

		Reservoir reservoir = new_reservoir();
		reservoir.pad = 0.0f;
		bool is_add = add_sample_to_reservoir(reservoir, hit_sample, ray_dir.w, random_float(noise_seed));
		if (is_add) {
			reservoir.pad = 1.0f;
		}

		reservoirs.data[reservoir_index(pixel_pos.xy, params.screen_size)] = reservoir;

		color = vec4(view_to_world_normal(ray_dir.xyz), 1.0);
	} else {
		Reservoir reservoir = new_reservoir();
		reservoirs.data[reservoir_index(pixel_pos.xy, params.screen_size)] = reservoir;
	}
	color *= params.intensity;

	imageStore(output_color, pixel_pos, color);
}