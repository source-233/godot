/**************************************************************************/
/*  restir.h                                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/math/vector2i.h"
#include "core/templates/rid.h"
#include "servers/rendering/renderer_rd/pipeline_deferred_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/restir.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/restir_denoise.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"

class ReSTIR {
public:
	ReSTIR();
	~ReSTIR();

	void allocate_buffers(Size2i size);
	RID init_restir_uniform_set(RID shader, uint32_t set_num);
	void free_buffers();

	struct SceneData {
		float projection[16];
		float inv_projection[16];
		float reprojection[16];
		float eye_offset[4];
		float inv_view_matrix[16];
		float view_matrix[16];
	};

	struct ReSTIRSetting {
		float temporal_pos_threshold = 0.05f;
		float spatial_resampling_kernel_radius = 1.0f;
		uint32_t spatial_num_samples = 4;
		uint32_t spatial_resampling_pass_index = 0;
		float spatial_resampling_occlusion_screen_trace_distance = 10.0f;

		float resampling_depth_error_threshold = 0.01f;
		float resampling_normal_dot_threshold = 0.5f;
	};
	void set_setting(ReSTIRSetting &setting);

	struct ReSTIRResource {
		RID normal_roughness_texture;
		RID depth_texture;
		RID history_depth_texture;
		RID diffuse_texture;
		RID history_diffuse_texture;
	};
	void process(Ref<RenderSceneBuffersRD> p_render_buffers, const ReSTIRResource &p_restir_resource, const SceneData &p_scene_data);

	struct ReSTIRDenoiserSetting {
		float max_frames_accumulated = 16.0f;
		float history_distance_threshold = 0.5f;
		float bilateral_filter_spatial_kernel_radius = 0.01f;
		uint32_t bilateral_filter_num_samples = 16u;
		float bilateral_filter_depth_weight_scale = 10.0f;
		float bilateral_filter_normal_angle_threshold_scale = 0.5f;
		float bilateral_filter_strong_blur_variance_threshold = 0.05f;
		float disocclusion_variance = 0.1f;
	};
	void set_denoiser_setting(ReSTIRDenoiserSetting &setting);

	struct ReSTIRDenoiserResource {
		RID normal_roughness_texture;
		RID depth_texture;
		RID history_depth_texture;
		RID history_num_frames_accumulated_texture;
		RID out_num_frames_accumulated_texture;
		RID diffuse_texture;
		RID history_diffuse_texture;
		RID out_diffuse_texture;
	};
	void process_denoise(Ref<RenderSceneBuffersRD> p_render_buffers, const ReSTIRDenoiserResource &p_denoiser_resource, const SceneData &p_scene_data);

private:
	struct HitSample {
		float ray_direction[3];
		float distance;
		float hit_normal[3];
		float pdf;
		float out_radiance[3];
		float validate;
	};

	struct Reservoir {
		HitSample hsample;

		float weight_sum;
		float weight;
		uint32_t sample_count;
		float pad;
	};

	enum ReSTIRUniformSet {
		RESTIR_UNIFORM_SET = 1,
	};

	enum ReSTIRPipeline {
		RESTIR_PIPELINE_TEMPORAL_CLEAR = 0,
		RESTIR_PIPELINE_TEMPORAL_REUSE = 1,
		RESTIR_PIPELINE_SPATIAL_REUSE = 2,
		RESTIR_PIPELINE_INTEGRATE_AND_UPSAMPLE = 3,
		RESTIR_PIPELINE_MAX = 4,
	};

	ReSTIRSetting restir_setting;

	struct ReSTIRPushConstant {
		int32_t screen_size[2];
		uint32_t frame_count;

		float temporal_pos_threshold;
		float spatial_resampling_kernel_radius;
		uint32_t spatial_num_samples;
		uint32_t spatial_resampling_pass_index;
		float spatial_resampling_occlusion_screen_trace_distance;

		float resampling_depth_error_threshold;
		float resampling_normal_dot_threshold;
	};

	RestirShaderRD restir_shader;
	RID restir_shader_version;
	PipelineDeferredRD restir_pipelines[RESTIR_PIPELINE_MAX];

	RID scene_data_ubo;

	struct ReservoirsSetting {
		int32_t reservoir_size[2] = { 0, 0 };
		int32_t pad[2] = { 0, 0 };
	} reservoirs_setting;
	RID reservoirs_setting_ubo;

	RID reservoirs;
	RID temporal_reservoirs;

	enum DenoiserPipeline {
		DENOISER_PIPELINE_TEMPORAL_ACCUMULATION = 0,
		DENOISER_PIPELINE_BILATERAL_FILTER = 1,
		DENOISER_PIPELINE_MAX = 2,
	};

	ReSTIRDenoiserSetting denoiser_setting;

	struct DenoiserPushConstant {
		int32_t screen_size[2];
		uint32_t frame_count;

		float max_frames_accumulated;
		float history_distance_threshold;
		float bilateral_filter_spatial_kernel_radius;
		uint32_t bilateral_filter_num_samples;
		float bilateral_filter_depth_weight_scale;
		float bilateral_filter_normal_angle_threshold_scale;
		float bilateral_filter_strong_blur_variance_threshold;
		float disocclusion_variance;
	};

	RestirDenoiseShaderRD denoiser_shader;
	RID denoiser_shader_version;
	PipelineDeferredRD denoiser_pipelines[DENOISER_PIPELINE_MAX];
};