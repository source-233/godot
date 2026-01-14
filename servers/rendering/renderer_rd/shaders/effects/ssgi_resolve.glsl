#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 projection[2];
	mat4 inv_projection[2];
	mat4 reprojection[2];
	vec4 eye_offset[2];
}
scene_data;
layout(rgba16f, set = 0, binding = 1) uniform restrict readonly image2D source_ssgi;
layout(rgba16f, set = 0, binding = 2) uniform restrict readonly image2D source_history;
layout(set = 0, binding = 3) uniform sampler2D source_hiz;

layout(rgba16f, set = 0, binding = 4) uniform restrict writeonly image2D output_color;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	uint view_index;
	uint weight;
}
params;

vec3 reprojection(vec2 uv, float depth) {
	vec4 previous_pos_ndc = scene_data.reprojection[params.view_index] * vec4(uv * 2.0f - 1.0f, depth, 1.0f);
	return vec3((previous_pos_ndc.xy / previous_pos_ndc.w) * 0.5f + 0.5f, previous_pos_ndc.z / previous_pos_ndc.w);
}

vec4 ReSTIR(ivec2 pixel_pos, ivec2 pos_group, vec2 uv) {
	vec4 color = vec4(0.0);

	vec3 uv_reprojected = reprojection(uv, texelFetch(source_hiz, pixel_pos, 0).x);

	ivec2 pixel_pos_reprojected = ivec2(uv_reprojected.xy * params.screen_size);

	color = imageLoad(source_ssgi, ivec2(pixel_pos));

	// Get color from reprojected pixel
	vec4 history_color = imageLoad(source_history, ivec2(pixel_pos_reprojected));
	float history_color_luminance = dot(history_color.rgb, vec3(0.2126, 0.7152, 0.0722));

	float color_luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));

	float delta_luminance = clamp(abs(history_color_luminance - color_luminance), 1.0 / params.weight, 1.0 - 1.0 / params.weight);

	history_color *= 1.0 - delta_luminance;
	color *= delta_luminance;

	return clamp(color + history_color, vec4(0.0), vec4(1.0));
}

void main() {
	if (any(greaterThanEqual(vec2(gl_GlobalInvocationID.xy), params.screen_size))) {
		return;
	}

	const ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);
	const ivec2 pos_group = ivec2(gl_LocalInvocationID.xy);
	const vec2 uv = (gl_GlobalInvocationID.xy + 0.5f) / params.screen_size;

	vec4 color = ReSTIR(pixel_pos, pos_group, uv);

	imageStore(output_color, pixel_pos, color);
}
