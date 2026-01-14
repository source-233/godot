#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 projection;
	mat4 inv_projection;
	mat4 reprojection;
	mat4 inv_view_matrix;
	mat4 view_matrix;
	vec4 eye_offset;
} scene_data;

layout(r32f, set = 0, binding = 1) uniform restrict readonly image2D source_depth_texture;
layout(r32f, set = 0, binding = 2) uniform restrict readonly image2D history_depth_texture;
layout(rgba8, set = 0, binding = 3) uniform restrict readonly image2D source_normal_roughness_texture;

layout(r32f, set = 0, binding = 4) uniform restrict readonly image2D history_num_frames_accumulated_texture;
layout(r32f, set = 0, binding = 5) uniform restrict writeonly image2D out_num_frames_accumulated_texture;

layout(rgba16f, set = 0, binding = 6) uniform restrict image2D diffuse_texture;
layout(rgba16f, set = 0, binding = 7) uniform restrict readonly image2D history_diffuse_texture;

#ifdef DENOISE_SPECULAR
layout(rgba16f, set = 0, binding = 9) uniform restrict image2D rough_specular_texture;
layout(rgba16f, set = 0, binding = 10) uniform restrict readonly image2D history_rough_specular_texture;
#endif

#ifdef DENOISE_VARIANCE
layout(r32f, set = 0, binding = 12) uniform restrict readonly image2D source_resolve_variance_texture;
layout(r32f, set = 0, binding = 13) uniform restrict readonly image2D history_resolve_variance_texture;
layout(r32f, set = 0, binding = 14) uniform restrict writeonly image2D out_resolve_variance_texture;
#endif


layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	uint frame_count;
	float max_frames_accumulated;
	float history_distance_threshold;
	float bilateral_filter_spatial_kernel_radius;
	uint bilateral_filter_num_samples;
	float bilateral_filter_depth_weight_scale;
	float bilateral_filter_normal_angle_threshold_scale;
	float bilateral_filter_strong_blur_variance_threshold;
	float disocclusion_variance;
} params;

const ivec2 kOffsets3x3[8] = {
	ivec2(-1, -1),
	ivec2( 0, -1),
	ivec2( 1, -1),
	ivec2(-1,  0),
	ivec2( 1,  0),
	ivec2(-1,  1),
	ivec2( 0,  1),
	ivec2( 1,  1),
};

struct Bilinear {
	vec2 origin;
	vec2 weights;
};

Bilinear get_bilinear_filter(vec2 uv, vec2 texture_size) {
	Bilinear result;
	result.origin = floor(uv * texture_size - .5f);
	result.weights = fract(uv * texture_size - .5f);
	return result;
}

vec4 get_bilinear_custom_weights(Bilinear f, vec4 custom_weights) {
	vec4 weights;
	weights.x = (1.0f - f.weights.x) * (1.0f - f.weights.y);
	weights.y = f.weights.x * (1.0f - f.weights.y);
	weights.z = (1.0f - f.weights.x) * f.weights.y;
	weights.w = f.weights.x * f.weights.y;
	return weights * custom_weights;
}

vec3 weighted_average(vec3 v00, vec3 v10, vec3 v01, vec3 v11, vec4 weights) {
	vec3 result = v00 * weights.x + v10 * weights.y + v01 * weights.z + v11 * weights.w;
	return result / max(dot(weights, vec4(1.0f)), .00001f);
}

float weighted_average(vec4 v, vec4 weights) {
	return dot(v, weights) / max(dot(weights, vec4(1.0f)), .00001f);
}

struct FGatherUV {
	vec2 uv00;
	vec2 uv10;
	vec2 uv11;
	vec2 uv01;
};

FGatherUV get_gather_uv(Bilinear in_filter, vec2 in_texel_size) {
	FGatherUV out_uv;
	out_uv.uv00 = (in_filter.origin + .5f) * in_texel_size;
	out_uv.uv10 = out_uv.uv00 + vec2(in_texel_size.x, 0);
	out_uv.uv01 = out_uv.uv00 + vec2(0, in_texel_size.y);
	out_uv.uv11 = out_uv.uv00 + in_texel_size;
	return out_uv;
}

vec3 reprojection(vec2 uv, float depth) {
	vec4 previous_pos_ndc = scene_data.reprojection * vec4(uv * 2.0f - 1.0f, depth, 1.0f);
	return vec3((previous_pos_ndc.xy / previous_pos_ndc.w) * 0.5f + 0.5f, previous_pos_ndc.z / previous_pos_ndc.w);
}

vec3 load_normal(ivec2 pixel_pos) {
	return normalize(imageLoad(source_normal_roughness_texture, pixel_pos).xyz * 2.0 - 1.0);
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

vec3 tonemap_lighting_for_bilateral(vec3 lighting) {
	return lighting / (1.0f + dot(lighting, vec3(0.2126, 0.7152, 0.0722)));
}

vec3 inverse_tonemap_lighting_for_bilateral(vec3 tonemapped_lighting) {
	float luminance = dot(tonemapped_lighting, vec3(0.2126, 0.7152, 0.0722));
	return tonemapped_lighting / (1.0f - luminance);
}

#ifdef DENOISER_PIPELINE_TEMPORAL_ACCUMULATION
void temporal_accumulation(const ivec2 pixel_pos) {
	ivec2 screen_coord = pixel_pos;
	vec2 screen_uv = vec2(pixel_pos + 0.5f) / vec2(params.screen_size);

	if (any(greaterThanEqual(screen_coord, params.screen_size))) {
		return;
	}

	const float screen_depth = imageLoad(source_depth_texture, screen_coord).x;
	if (screen_depth <= 0.0f) {
		return;
	}

	const vec3 history_screen_position = reprojection(screen_uv, screen_depth);
	vec2 history_screen_uv = history_screen_position.xy;
	const bool b_history_was_onscreen = all(lessThan(history_screen_uv, vec2(1.0f))) && all(greaterThan(history_screen_uv, vec2(0.0f)));
	history_screen_uv = clamp(history_screen_uv, vec2(0.0f), vec2(1.0f));

	const Bilinear bilinear_filter_at_history_screen_uv = get_bilinear_filter(history_screen_uv, vec2(params.screen_size));
	vec2 history_gather_uv = (bilinear_filter_at_history_screen_uv.origin + 1.0f) / vec2(params.screen_size);

	// History depth sampling
	ivec2 history_depth_coord = ivec2(bilinear_filter_at_history_screen_uv.origin);
	ivec2 history_depth_coord10 = history_depth_coord + ivec2(1, 0);
	ivec2 history_depth_coord01 = history_depth_coord + ivec2(0, 1);
	ivec2 history_depth_coord11 = history_depth_coord + ivec2(1, 1);

	history_depth_coord = clamp(history_depth_coord, ivec2(0), params.screen_size - 1);
	history_depth_coord10 = clamp(history_depth_coord10, ivec2(0), params.screen_size - 1);
	history_depth_coord01 = clamp(history_depth_coord01, ivec2(0), params.screen_size - 1);
	history_depth_coord11 = clamp(history_depth_coord11, ivec2(0), params.screen_size - 1);

	float history_depth00 = imageLoad(history_depth_texture, history_depth_coord).x;
	float history_depth10 = imageLoad(history_depth_texture, history_depth_coord10).x;
	float history_depth01 = imageLoad(history_depth_texture, history_depth_coord01).x;
	float history_depth11 = imageLoad(history_depth_texture, history_depth_coord11).x;

	vec4 history_scene_depth = vec4(
		linearize_depth(history_depth00),
		linearize_depth(history_depth10),
		linearize_depth(history_depth01),
		linearize_depth(history_depth11)
	);

	vec3 view_normal = load_normal(screen_coord);
	vec3 world_normal = view_to_world_normal(view_normal);

	// Noise for threshold variation
	uint noise_seed = random_seed(ivec3(pixel_pos, params.frame_count));
	float noise = random_float(noise_seed);
	float disocclusion_distance_threshold = params.history_distance_threshold * mix(0.5f, 1.5f, noise);

	FGatherUV history_gather = get_gather_uv(bilinear_filter_at_history_screen_uv, 1.0f / vec2(params.screen_size));

	// Adjust threshold for grazing angles
	vec3 view_pos = screen_to_view_pos(vec3(screen_uv, screen_depth));
	vec3 V = -normalize(view_pos + scene_data.eye_offset.xyz);
	disocclusion_distance_threshold /= clamp(dot(V, view_normal), 0.1f, 1.0f);

	float prev_scene_depth = linearize_depth(history_screen_position.z);
	vec4 distance_to_history_value = abs(history_scene_depth - prev_scene_depth);
	vec4 occlusion_weights = vec4(lessThan(distance_to_history_value, vec4(- prev_scene_depth * disocclusion_distance_threshold)));

	vec4 visibility_weights = clamp(vec4(b_history_was_onscreen ? 1.0f : 0.0f) * occlusion_weights, 0.0f, 1.0f);
	vec4 final_weights = get_bilinear_custom_weights(bilinear_filter_at_history_screen_uv, visibility_weights);

	vec3 new_diffuse_lighting = imageLoad(diffuse_texture, screen_coord).xyz;

	// Sample history
	vec3 history_diffuse_indirect00 = imageLoad(history_diffuse_texture, ivec2(history_gather.uv00 * params.screen_size)).xyz;
	vec3 history_diffuse_indirect10 = imageLoad(history_diffuse_texture, ivec2(history_gather.uv10 * params.screen_size)).xyz;
	vec3 history_diffuse_indirect01 = imageLoad(history_diffuse_texture, ivec2(history_gather.uv01 * params.screen_size)).xyz;
	vec3 history_diffuse_indirect11 = imageLoad(history_diffuse_texture, ivec2(history_gather.uv11 * params.screen_size)).xyz;

	vec3 history_diffuse_indirect = weighted_average(
		history_diffuse_indirect00, history_diffuse_indirect10, 
		history_diffuse_indirect01, history_diffuse_indirect11, 
		final_weights
	);

#ifdef DENOISE_SPECULAR
	vec3 new_rough_specular_lighting = imageLoad(source_rough_specular_texture, screen_coord).xyz;

	vec3 history_rough_specular_indirect00 = imageLoad(history_rough_specular_texture, ivec2(history_gather.uv00 * params.screen_size)).xyz;
	vec3 history_rough_specular_indirect10 = imageLoad(history_rough_specular_texture, ivec2(history_gather.uv10 * params.screen_size)).xyz;
	vec3 history_rough_specular_indirect01 = imageLoad(history_rough_specular_texture, ivec2(history_gather.uv01 * params.screen_size)).xyz;
	vec3 history_rough_specular_indirect11 = imageLoad(history_rough_specular_texture, ivec2(history_gather.uv11 * params.screen_size)).xyz;

	vec3 history_rough_specular_indirect = weighted_average(
		history_rough_specular_indirect00, history_rough_specular_indirect10, 
		history_rough_specular_indirect01, history_rough_specular_indirect11, 
		final_weights
	);
#endif

	// Sample num frames accumulated
	float num_frames_accumulated00 = imageLoad(history_num_frames_accumulated_texture, ivec2(history_gather.uv00 * params.screen_size)).x;
	float num_frames_accumulated10 = imageLoad(history_num_frames_accumulated_texture, ivec2(history_gather.uv10 * params.screen_size)).x;
	float num_frames_accumulated01 = imageLoad(history_num_frames_accumulated_texture, ivec2(history_gather.uv01 * params.screen_size)).x;
	float num_frames_accumulated11 = imageLoad(history_num_frames_accumulated_texture, ivec2(history_gather.uv11 * params.screen_size)).x;

	vec4 num_frames_accumulated_neighborhood = vec4(
		num_frames_accumulated00, num_frames_accumulated10, 
		num_frames_accumulated01, num_frames_accumulated11
	) * params.max_frames_accumulated;

	num_frames_accumulated_neighborhood = min(num_frames_accumulated_neighborhood + 1.0f, params.max_frames_accumulated);
	float num_frames_accumulated = weighted_average(num_frames_accumulated_neighborhood, final_weights);

	float new_num_frames_accumulated = num_frames_accumulated;
	new_num_frames_accumulated = min(new_num_frames_accumulated, params.max_frames_accumulated);
	new_num_frames_accumulated = b_history_was_onscreen ? new_num_frames_accumulated : 0.0f;

	if (dot(visibility_weights, vec4(1.0f)) < 0.01f) {
		new_num_frames_accumulated = 0.0f;
	}

	// Neighborhood clamp
	const ivec2 min_screen_coord = ivec2(0);
	const ivec2 max_screen_coord = params.screen_size - 1;

	{
		vec3 neighbor_min = new_diffuse_lighting;
		vec3 neighbor_max = new_diffuse_lighting;

		for (uint neighbor_id = 0; neighbor_id < 8; neighbor_id++) {
			const ivec2 sample_offset = kOffsets3x3[neighbor_id];

			ivec2 neighbor_screen_coord = screen_coord + sample_offset;
			neighbor_screen_coord = clamp(neighbor_screen_coord, min_screen_coord, max_screen_coord);

			const vec3 lighting = imageLoad(diffuse_texture, neighbor_screen_coord).xyz;
			neighbor_min = min(neighbor_min, lighting.xyz);
			neighbor_max = max(neighbor_max, lighting.xyz);
		}

		history_diffuse_indirect = clamp(history_diffuse_indirect, neighbor_min, neighbor_max);
	}

#ifdef DENOISE_SPECULAR
	{
		vec3 neighbor_min = new_rough_specular_lighting;
		vec3 neighbor_max = new_rough_specular_lighting;

		for (uint neighbor_id = 0; neighbor_id < 8; neighbor_id++) {
			const ivec2 sample_offset = kOffsets3x3[neighbor_id];

			ivec2 neighbor_screen_coord = screen_coord + sample_offset;
			neighbor_screen_coord = clamp(neighbor_screen_coord, min_screen_coord, max_screen_coord);

			const vec3 lighting = imageLoad(source_rough_specular_texture, neighbor_screen_coord).xyz;
			neighbor_min = min(neighbor_min, lighting.xyz);
			neighbor_max = max(neighbor_max, lighting.xyz);
		}

		history_rough_specular_indirect = clamp(history_rough_specular_indirect, neighbor_min, neighbor_max);
	}
#endif

	// Temporal accumulation
	float alpha = 1.0f / (1.0f + new_num_frames_accumulated);
	vec3 out_diffuse_indirect = mix(history_diffuse_indirect, new_diffuse_lighting, alpha);
	// Ensure non-negative values
	out_diffuse_indirect = max(out_diffuse_indirect, 0.0f);

#ifdef DENOISE_SPECULAR
	vec3 out_rough_specular_indirect = mix(history_rough_specular_indirect, new_rough_specular_lighting, alpha);
	// Ensure non-negative values
	out_rough_specular_indirect = max(out_rough_specular_indirect, 0.0f);
#endif

	// Write outputs
	imageStore(diffuse_texture, screen_coord, vec4(out_diffuse_indirect, 1.0f));
#ifdef DENOISE_SPECULAR
	imageStore(out_rough_specular_texture, screen_coord, vec4(out_rough_specular_indirect, 1.0f));
#endif
	imageStore(out_num_frames_accumulated_texture, screen_coord, vec4(new_num_frames_accumulated / params.max_frames_accumulated, 0.0f, 0.0f, 0.0f));

	// Handle variance
#ifdef DENOISE_VARIANCE
	float variance_history_weight = b_history_was_onscreen ? 0.9f : 0.0f;
	float new_resolve_variance = imageLoad(source_resolve_variance_texture, screen_coord).x;

	if (dot(visibility_weights, vec4(1.0f)) < 1.0f) {
		variance_history_weight = 0.0f;
		new_resolve_variance = params.disocclusion_variance;
	}

	float resolve_variance_history_value = 0.0f;
	if (variance_history_weight > 0.0f) {
		resolve_variance_history_value = imageLoad(history_resolve_variance_texture, ivec2(history_screen_uv * params.screen_size)).x;
	}

	float accumulated_resolve_variance = max(mix(new_resolve_variance, resolve_variance_history_value, variance_history_weight), 0.0f);
	imageStore(out_resolve_variance_texture, screen_coord, vec4(accumulated_resolve_variance, 0.0f, 0.0f, 0.0f));
#endif
}
#endif

#ifdef DENOISER_PIPELINE_BILATERAL_FILTER
void bilateral_filter(const ivec2 pixel_pos) {
	ivec2 screen_coord = pixel_pos;
	vec2 screen_uv = (vec2(pixel_pos) + 0.5f) / vec2(params.screen_size);

	if (any(greaterThanEqual(screen_coord, params.screen_size))) {
		return;
	}

	const float screen_depth = imageLoad(source_depth_texture, screen_coord).x;
	if (screen_depth <= 0.0f) {
		return;
	}

	vec3 out_diffuse_indirect = imageLoad(diffuse_texture, screen_coord).xyz;
	// Apply tonemapping for bilateral filter
	out_diffuse_indirect = tonemap_lighting_for_bilateral(out_diffuse_indirect);

#ifdef DENOISE_SPECULAR
	vec3 out_rough_specular_indirect = imageLoad(source_rough_specular_texture, screen_coord).xyz;
	// Apply tonemapping for bilateral filter
	out_rough_specular_indirect = tonemap_lighting_for_bilateral(out_rough_specular_indirect);
#endif


// #ifdef DENOISE_VARIANCE
// 	float variance_from_spatial_resolve = imageLoad(source_resolve_variance_texture, screen_coord).x;
// 	float strong_blur = variance_from_spatial_resolve > params.bilateral_filter_strong_blur_variance_threshold ? 1.0f : 0.0f;
// 	float disocclusion_blur = variance_from_spatial_resolve > params.disocclusion_variance - 0.1f ? 1.0f : 0.0f;
// 	float min_kernel_radius = 0.0f;
// 	float max_kernel_radius = params.bilateral_filter_spatial_kernel_radius * params.screen_size.x * mix(1.0f, 2.0f, strong_blur);
// 	float kernel_radius = mix(min_kernel_radius, max_kernel_radius, 1.0f);

// 	if (kernel_radius >= 0.5f && variance_from_spatial_resolve > 0.04f) {
// #else
	float strong_blur = 0.0f;
	float disocclusion_blur = 0.0f;
	float min_kernel_radius = 0.0f;
	float max_kernel_radius = params.bilateral_filter_spatial_kernel_radius * params.screen_size.x;
	float kernel_radius = mix(min_kernel_radius, max_kernel_radius, 1.0f);
	if (kernel_radius >= 0.5f) {
// #endif
		float total_weight = 1.0f;
		float guassian_normalize = 2.0f / (kernel_radius * kernel_radius);
		uint noise_seed = random_seed(ivec3(pixel_pos, params.frame_count));

		vec3 view_position = screen_to_view_pos(vec3(screen_uv, screen_depth));
		vec3 view_normal = load_normal(screen_coord);
		vec4 scene_plane = vec4(view_normal, dot(view_position, view_normal));
		float linear_depth = linearize_depth(screen_depth);

		uint num_bilateral_filter_samples = min(params.bilateral_filter_num_samples * uint(mix(1.0f, 2.0f, strong_blur)), 16u);

		for (uint sample_index = 0; sample_index < num_bilateral_filter_samples; sample_index++) {
			// Generate sample offset using simple random
			float rand1 = random_float(noise_seed);
			float rand2 = random_float(noise_seed);
			float angle = 2.0f * PI * rand1;
			float radius = kernel_radius * sqrt(rand2);
			vec2 offset = vec2(cos(angle), sin(angle)) * radius;
			ivec2 neighbor_screen_coord = screen_coord + ivec2(offset);

			if (all(greaterThanEqual(neighbor_screen_coord, ivec2(0))) && all(lessThan(neighbor_screen_coord, params.screen_size))) {
				float neighbor_screen_depth = imageLoad(source_depth_texture, neighbor_screen_coord).x;
				if (neighbor_screen_depth > 0.0f) {
					vec2 neighbor_screen_uv = (vec2(neighbor_screen_coord) + 0.5f) / vec2(params.screen_size);
					vec3 neighbor_view_position = screen_to_view_pos(vec3(neighbor_screen_uv, neighbor_screen_depth));
					vec3 neighbor_view_normal = load_normal(neighbor_screen_coord);

					float plane_distance = abs(dot(vec4(neighbor_view_position, -1.0f), scene_plane));
					float relative_depth_difference = plane_distance / linear_depth;
					float depth_weight = exp2(-params.bilateral_filter_depth_weight_scale * (relative_depth_difference * relative_depth_difference));
					float spatial_weight = exp2(-guassian_normalize * radius * radius);
					float normal_dot = clamp(dot(scene_plane.xyz, neighbor_view_normal), -1.0f, 1.0f);
					float normal_weight = clamp((normal_dot - params.bilateral_filter_normal_angle_threshold_scale) / 
						(1.0f - params.bilateral_filter_normal_angle_threshold_scale), 0.0f, 1.0f);

					float sample_weight = spatial_weight * depth_weight * mix(normal_weight, 1.0f, disocclusion_blur);
					vec3 neighbor_diffuse = tonemap_lighting_for_bilateral(imageLoad(diffuse_texture, neighbor_screen_coord).xyz);
					out_diffuse_indirect += neighbor_diffuse * sample_weight;
#ifdef DENOISE_SPECULAR
					vec3 neighbor_specular = tonemap_lighting_for_bilateral(imageLoad(source_rough_specular_texture, neighbor_screen_coord).xyz);
					out_rough_specular_indirect += neighbor_specular * sample_weight;
#endif

					total_weight += sample_weight;
				}
			}
		}

		out_diffuse_indirect = out_diffuse_indirect / total_weight;
#ifdef DENOISE_SPECULAR
		out_rough_specular_indirect = out_rough_specular_indirect / total_weight;
#endif
	}

	out_diffuse_indirect = max(inverse_tonemap_lighting_for_bilateral(out_diffuse_indirect), 0.0f);
	imageStore(diffuse_texture, screen_coord, vec4(out_diffuse_indirect, 1.0f));

#ifdef DENOISE_SPECULAR
	out_rough_specular_indirect = max(inverse_tonemap_lighting_for_bilateral(out_rough_specular_indirect), 0.0f);
	imageStore(out_rough_specular_texture, screen_coord, vec4(out_rough_specular_indirect, 1.0f));
#endif
}
#endif

void main() {
	const ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

#ifdef DENOISER_PIPELINE_TEMPORAL_ACCUMULATION
	temporal_accumulation(pixel_pos);
#endif
#ifdef DENOISER_PIPELINE_BILATERAL_FILTER
	bilateral_filter(pixel_pos);
#endif
}