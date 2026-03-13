#[compute]

#version 450
// #extension GL_ARB_shading_language_include : enable
#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "restir_inc.glsl"

#ifdef RESTIR_PIPELINE_TEMPORAL_CLEAR
void temporal_clear(const ivec2 pixel_pos)
{
	ivec2 reservoir_coord = pixel_pos;

	if (any(greaterThanEqual(reservoir_coord, restir_setting.reservoir_size))) {
		return;
	}
	Reservoir reservoir = new_reservoir();
	temporal_reservoirs.data[reservoir_index(pixel_pos.xy, restir_setting.reservoir_size)] = reservoir;
}
#else

layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 projection;
	mat4 inv_projection;
	mat4 reprojection;
	vec4 eye_offset;
	mat4 inv_view_matrix;
	mat4 view_matrix;
}
scene_data;

layout(r32f, set = 0, binding = 1) uniform restrict readonly image2D source_depth;
layout(rgba8, set = 0, binding = 2) uniform restrict readonly image2D source_normal_roughness;
#ifdef RESTIR_PIPELINE_TEMPORAL_REUSE
layout(r32f, set = 0, binding = 3) uniform restrict readonly image2D source_history_depth;
#elif defined(RESTIR_PIPELINE_INTEGRATE_AND_UPSAMPLE)
layout(rgba16f, set = 0, binding = 3) uniform restrict writeonly image2D out_diffuse;
#endif

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	uint frame_count;
}
params;

vec3 reprojection(vec2 uv, float depth) {
	vec4 previous_pos_ndc = scene_data.reprojection * vec4(uv * 2.0f - 1.0f, depth, 1.0f);
	return vec3((previous_pos_ndc.xy / previous_pos_ndc.w) * 0.5f + 0.5f, previous_pos_ndc.z / previous_pos_ndc.w);
}

vec3 load_normal(ivec2 pixel_pos) {
	return normalize(imageLoad(source_normal_roughness, pixel_pos).xyz * 2.0 - 1.0);
}

vec3 view_to_world_pos(vec3 pos) {
	return (scene_data.inv_view_matrix * vec4(pos, 1.0)).xyz;
}

vec3 view_to_world_normal(vec3 normal) {
    return normalize(mat3(scene_data.inv_view_matrix) * normal);;
}

vec3 world_to_view_pos(vec3 pos) {
	return (scene_data.view_matrix * vec4(pos, 1.0)).xyz;
}

float linearize_depth(float depth) {
	vec4 pos = vec4(0.0, 0.0, depth, 1.0);
	pos = scene_data.inv_projection * pos;
	return pos.z / pos.w;
}

vec3 screen_to_view_pos(vec3 screen_pos) {
	vec4 pos;
	pos.xy = screen_pos.xy * 2.0 - 1.0;
	pos.z = screen_pos.z;
	pos.w = 1.0;
	pos = scene_data.inv_projection * pos;
	return pos.xyz / pos.w;
}

vec3 view_to_screen_pos(vec3 pos) {
	vec4 screen_pos = scene_data.projection * vec4(pos, 1.0);
	screen_pos.xyz /= screen_pos.w;
	screen_pos.xy = screen_pos.xy * 0.5 + 0.5;
	return screen_pos.xyz;
}

vec3 screen_to_world_pos(vec3 screen_pos) {
	return view_to_world_pos(screen_to_view_pos(screen_pos));
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
const float Diffuse_Lambert = 1.0f / PI;

float calculate_jacobian(vec3 receiver_position, vec3 neighbor_receiver_position, const HitSample neighbor_sample) {
	// 从ray_direction和distance计算hit_pos
	vec3 neighbor_hit_pos = neighbor_receiver_position + neighbor_sample.ray_direction * neighbor_sample.distance;

	vec3 sample_to_receiver_neighbor = neighbor_receiver_position - neighbor_hit_pos;
	float original_distance = length(sample_to_receiver_neighbor);
	float original_cos_angle = clamp(dot(neighbor_sample.hit_normal, vec3(sample_to_receiver_neighbor / original_distance)), 0.0001f, 1.0f);

	vec3 sample_to_receiver = receiver_position - neighbor_hit_pos;
	float new_distance = length(sample_to_receiver);
	float new_cos_angle = clamp(dot(neighbor_sample.hit_normal, vec3(sample_to_receiver / new_distance)), 0.0f, 1.0f);

	float jacobian = (new_cos_angle * original_distance * original_distance) / (original_cos_angle * new_distance * new_distance);

	if (isinf(jacobian) || isnan(jacobian)) {
		jacobian = 0;
	}

	// hit_normal is 0 when the ray misses the scene
	if (abs(dot(neighbor_sample.hit_normal, vec3(1.0f))) < .01f) {
		jacobian = 1.0f;
	}

	// Discard extreme re-weights that show up as fireflies
	if (jacobian > 10.0f || jacobian < 1 / 10.0f) {
		jacobian = 0;
	}

	return jacobian;
}

#ifdef RESTIR_PIPELINE_TEMPORAL_REUSE
void temporal_resampling(const ivec2 pixel_pos) {
	ivec2 reservoir_coord = pixel_pos;
	vec2 screen_uv = (vec2(pixel_pos) + 0.5f) / vec2(restir_setting.reservoir_size);
	const float screen_depth = imageLoad(source_depth, reservoir_coord).x;

	if (any(greaterThanEqual(reservoir_coord, restir_setting.reservoir_size)) || screen_depth <= 0.0f) {
		return;
	}

	Reservoir reservoir = reservoirs.data[reservoir_index(reservoir_coord, ivec2(restir_setting.reservoir_size))];

	vec3 world_position = screen_to_world_pos(vec3(screen_uv, screen_depth));
	uint noise_seed = random_seed(ivec3(pixel_pos, params.frame_count));

	const vec3 uv_history = reprojection(screen_uv, screen_depth);
	const bool b_history_was_on_screen = all(lessThanEqual(uv_history, vec3(1.0f))) && all(greaterThanEqual(uv_history, vec3(0.0f)));

	if (b_history_was_on_screen) {
		vec3 view_normal = load_normal(reservoir_coord);

		ivec2 reservoir_coord_history = ivec2(uv_history.xy * restir_setting.reservoir_size);
		ivec2 screen_coord_history = reservoir_coord_history.xy * (params.screen_size / restir_setting.reservoir_size);

		// Similarity detection
		float prev_scene_depth = imageLoad(source_history_depth, screen_coord_history).x;
		float depth_error = abs(max(0.3f, view_normal.z) * (prev_scene_depth / screen_depth - 1.0));
		bool b_history_from_nearby = depth_error < restir_setting.temporal_pos_threshold;

		if (b_history_from_nearby && uv_history.z > 0.0f) {
			Reservoir prev_reservoir = temporal_reservoirs.data[reservoir_index(reservoir_coord_history, ivec2(restir_setting.reservoir_size))];

			prev_reservoir.sample_count = min(prev_reservoir.sample_count, 20u);

			vec3 history_sample_world_position = screen_to_world_pos(vec3(uv_history.xy, prev_scene_depth));

			// @todo - causes fireflies
			float jacobian = calculate_jacobian(world_position, history_sample_world_position, prev_reservoir.hsample);

			if (jacobian > 0) {
				merge_reservoirs(reservoir, prev_reservoir, jacobian, random_float(noise_seed));
			}
		}
	}

	temporal_reservoirs.data[reservoir_index(reservoir_coord, ivec2(restir_setting.reservoir_size))] = reservoir;
}
#endif

#ifdef RESTIR_PIPELINE_SPATIAL_REUSE
void spatial_resampling(const ivec2 pixel_pos) {
	ivec2 reservoir_coord = pixel_pos;
	vec2 screen_uv = (vec2(pixel_pos) + 0.5f) / params.screen_size;
	const float screen_depth = imageLoad(source_depth, reservoir_coord).x;

	if (any(greaterThanEqual(reservoir_coord, restir_setting.reservoir_size)) || screen_depth <= 0.0f) {
		return;
	}
	Reservoir reservoir = temporal_reservoirs.data[reservoir_index(reservoir_coord, ivec2(restir_setting.reservoir_size))];

	vec3 world_position = screen_to_world_pos(vec3(screen_uv, screen_depth));
	uint noise_seed = random_seed(ivec3(pixel_pos, params.frame_count));

	vec3 view_normal = load_normal(reservoir_coord);
	vec3 world_normal = view_to_world_normal(view_normal);

	float noise = random_float(noise_seed);
	float spatial_kernel_scale = restir_setting.spatial_resampling_kernel_radius * restir_setting.reservoir_size.x;

	const float golden_angle = 2.3999632f;
	for (uint sample_index = 0; sample_index < restir_setting.spatial_num_samples; sample_index++) {
		const float angle = (sample_index + noise + .3f * restir_setting.spatial_resampling_pass_index) * golden_angle;
		const float radius = pow(float(sample_index + 1), 0.666f) * spatial_kernel_scale / float(restir_setting.spatial_num_samples);
		const vec2 reservoir_offset_float = vec2(cos(angle), sin(angle)) * radius;
		const ivec2 reservoir_pixel_offset = ivec2(floor(reservoir_offset_float + .5f));

		ivec2 neighbor_reservoir_coord = clamp(ivec2(reservoir_coord) + reservoir_pixel_offset, ivec2(0), ivec2(restir_setting.reservoir_size) - 1);
		float neighbor_scene_depth = imageLoad(source_depth, neighbor_reservoir_coord).x;

		if (neighbor_scene_depth > 0.0f) {
			Reservoir neighbor_reservoir = temporal_reservoirs.data[reservoir_index(neighbor_reservoir_coord, ivec2(restir_setting.reservoir_size))];

			uvec2 sample_screen_coord = neighbor_reservoir_coord;
			vec2 sample_screen_uv = (sample_screen_coord + 0.5f) / params.screen_size;
			float sample_scene_depth = neighbor_scene_depth;
			vec3 sample_world_position = screen_to_world_pos(vec3(sample_screen_uv, sample_scene_depth));
			vec3 neighbor_world_normal = view_to_world_normal(load_normal(neighbor_reservoir_coord));

			float depth_error = abs(max(0.3f, view_normal.z) * (screen_depth / neighbor_scene_depth - 1.0));
			float normal_dot = dot(world_normal, neighbor_world_normal);

			if (depth_error < restir_setting.resampling_depth_error_threshold && normal_dot > restir_setting.resampling_normal_dot_threshold) {
				vec3 neighbor_world_position = sample_world_position;
				float jacobian = calculate_jacobian(world_position, neighbor_world_position, neighbor_reservoir.hsample);

				bool b_neighbor_hit_visible = true;
				float swap_noise = random_float(noise_seed);

				// TODO: screen raycast, visibility check
				// if (restir_setting.spatial_resampling_occlusion_screen_trace_distance > 0.0f && jacobian > 0.0f && will_change_sample_on_merge(reservoir, neighbor_reservoir, neighbor_reservoir.pdf * jacobian, swap_noise)) {
				// 	vec3 neighbor_hit_position = neighbor_reservoir.hsample.hit_pos;
				// 	vec3 shadow_ray_direction = normalize(neighbor_hit_position - world_position);
				// 	const float contact_shadow_length_screen_scale = get_screen_ray_length_multiplier_for_projection_type(scene_depth).y;
				// 	float ray_length = restir_setting.spatial_resampling_occlusion_screen_trace_distance * contact_shadow_length_screen_scale;
				// 	float step_offset = noise - 0.5;
				// 	b_neighbor_hit_visible = screen_shadow_ray_cast(world_position, shadow_ray_direction, ray_length, 8, step_offset) < 0;

				// 	b_neighbor_hit_visible = true;
				// }

				if (b_neighbor_hit_visible) {
					merge_reservoirs(reservoir, neighbor_reservoir, jacobian, swap_noise);
				}
			}
		}
	}

	reservoirs.data[reservoir_index(reservoir_coord, ivec2(restir_setting.reservoir_size))] = reservoir;
}
#endif

#ifdef RESTIR_PIPELINE_INTEGRATE_AND_UPSAMPLE
void integrate_and_upsample(const ivec2 pixel_pos) {
	ivec2 screen_coord = pixel_pos;
	vec2 screen_uv = (vec2(pixel_pos) + 0.5f) / params.screen_size;

	if (any(greaterThanEqual(screen_coord, params.screen_size))) {
		return;
	}

	vec3 world_position = screen_to_world_pos(vec3(screen_uv, imageLoad(source_depth, screen_coord).x));
	vec3 view_normal = load_normal(screen_coord);
	vec3 world_normal = view_to_world_normal(view_normal);
	vec4 scene_plane = vec4(world_normal, dot(world_position, world_normal));
	const vec3 V = normalize(-world_position);
	const float NoV = clamp(dot(world_normal, V), 0.0f, 1.0f);

#ifdef JITTERED_BILINEAR_UPSAMPLE

	vec2 noise_offset = vec2(0.0f);

	// Jitter the bilinear sample position, but only accept the jittered position if it lies in the same plane as the original pixel
	if (upsample_kernel_size > 0) {
		vec2 screen_tile_jitter_e = blue_noise_vec2(screen_coord, scene_data.frame_count);
		vec2 jitter_noise_offset = (screen_tile_jitter_e * 2.0f - 1.0f) * reservoir_downsample_factor * upsample_kernel_size;
		vec2 jittered_screen_uv = (clamp(vec2(screen_coord) + jitter_noise_offset, vec2(scene_data.view_rect_min.xy), vec2(scene_data.view_rect_min.xy) + vec2(scene_data.view_size_inv.xy) - 1.0f)) * scene_data.buffer_size_inv.zw;
		float jittered_scene_depth = calc_scene_depth(jittered_screen_uv);

		float depth_weight;

		{
			vec3 translated_jittered_world_position = get_translated_world_position_from_screen_uv(jittered_screen_uv, jittered_scene_depth);
			float plane_distance = abs(dot(vec4(translated_jittered_world_position, -1.0f), scene_plane));
			float relative_depth_difference = plane_distance / material.scene_depth;
			depth_weight = exp2(-1000000.0f * (relative_depth_difference * relative_depth_difference));
		}

		if (depth_weight > 0.01f) {
			noise_offset = jitter_noise_offset;
		}
	}
	ScreenProbeSample screen_probe_sample = (ScreenProbeSample)0;
	calculate_uniform_upsample_interpolation_weights(screen_coord, noise_offset, translated_world_position, material.scene_depth, material.world_normal, screen_probe_sample);

	float epsilon = 0.01f;
	screen_probe_sample.weights /= max(dot(screen_probe_sample.weights, vec4(1.0f)), epsilon);

	// Fallback to unjittered position if there isn't at least one valid reservoir to sample from
	if (dot(screen_probe_sample.weights, vec4(1.0f)) <= 1.0f - epsilon) {
		calculate_uniform_upsample_interpolation_weights(screen_coord, vec2(0.0f), translated_world_position, material.scene_depth, material.world_normal, screen_probe_sample);
		screen_probe_sample.weights /= max(dot(screen_probe_sample.weights, vec4(1.0f)), epsilon);
	}

	vec3 diffuse_lighting = vec3(0.0f);
	vec3 specular_lighting = vec3(0.0f);

	float total_weight = 0.0f;
	float mean = 0.0f;
	float s = 0.0f;

	for (uint sample_index = 0; sample_index < 4; sample_index++) {
		ivec2 sample_reservoir_coord = ivec2(screen_probe_sample.atlas_coord[sample_index]);
		float sample_scene_depth = downsampled_scene_depth[sample_reservoir_coord];
		float weight = screen_probe_sample.weights[sample_index];

		if (sample_scene_depth > 0.0f && weight > 0.0f) {
			Reservoir sample_reservoir = reservoirs.data[reservoir_index(sample_reservoir_coord, ivec2(restir_setting.reservoir_size))];

			vec3 sample_lighting = sample_reservoir.hsample.out_radiance * sample_reservoir.weight;
			diffuse_lighting += sample_lighting * weight * max(dot(material.world_normal, sample_reservoir.hsample.ray_direction), 0.0f);

			vec3 H = normalize(V + sample_reservoir.hsample.ray_direction);
			float NoH = clamp(dot(material.world_normal, H), 0.0f, 1.0f);
			float D = d_ggx(pow4(material.roughness), NoH);
			float Vis = vis_implicit();

			specular_lighting += tonemap_lighting_for_rough_specular(sample_lighting * (D * Vis)) * weight;

#ifdef USE_BILATERAL_FILTER
			// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
			float weight = screen_probe_sample.weights[sample_index];
			total_weight += weight;
			float luma_sample_radiance = luminance(sample_lighting);
			float old_mean = mean;
			mean += weight / total_weight * (luma_sample_radiance - old_mean);
			s += weight * (luma_sample_radiance - old_mean) * (luma_sample_radiance - mean);
#endif
		}
	}

	specular_lighting = inverse_tonemap_lighting_for_rough_specular(specular_lighting);
	float resolve_variance = total_weight > 0.0f ? s / total_weight : 0.0f;

#else
	ivec2 reservoir_coord = ivec2(screen_coord) * (restir_setting.reservoir_size / params.screen_size);
	reservoir_coord = min(reservoir_coord, restir_setting.reservoir_size - 1);
	Reservoir reservoir = reservoirs.data[reservoir_index(reservoir_coord, ivec2(restir_setting.reservoir_size))];

	vec3 diffuse_lighting = vec3(0.0f);
	float resolve_variance = 0.0f;

	if (reservoir.hsample.pdf > 0.0f) {
		vec3 reservoir_radiance = reservoir.hsample.out_radiance;
		float reservoir_weight = reservoir.weight;
		diffuse_lighting = reservoir_radiance * reservoir_weight * max(dot(world_normal, reservoir.hsample.ray_direction), 0.0f);
	}
#endif
	diffuse_lighting *= Diffuse_Lambert;
	imageStore(out_diffuse, screen_coord, vec4(diffuse_lighting, 1.0f));

#if USE_BILATERAL_FILTER
	// Clamp the variance range to not overlap with DisocclusionVariance, so the bilateral filter can detect disocclusion
	resolve_variance = min(resolve_variance, disocclusion_variance - 0.1f);
	rw_resolve_variance[screen_coord] = resolve_variance;
#endif
}
#endif
#endif

void main() {
	const ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

#ifdef RESTIR_PIPELINE_TEMPORAL_CLEAR
	temporal_clear(pixel_pos);
#endif
#ifdef RESTIR_PIPELINE_TEMPORAL_REUSE
	temporal_resampling(pixel_pos);
#endif
#ifdef RESTIR_PIPELINE_SPATIAL_REUSE
	spatial_resampling(pixel_pos);
#endif
#ifdef RESTIR_PIPELINE_INTEGRATE_AND_UPSAMPLE
	integrate_and_upsample(pixel_pos);
#endif
}