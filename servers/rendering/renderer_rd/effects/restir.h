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
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"

class ReSTIR {
public:
	ReSTIR();
	~ReSTIR();

	void allocate_buffers(Size2i size);
	void free_buffers();

	struct ReSTIRSetting {
		int32_t reservoir_size[2] = { 0, 0 };

		float temporal_pos_threshold = 0.05f;
		float spatial_resampling_kernel_radius = 1.0f;
		uint32_t spatial_num_samples = 4;
		uint32_t spatial_resampling_pass_index = 0;
		float spatial_resampling_occlusion_screen_trace_distance = 10.0f;

		float resampling_depth_error_threshold = 0.01f;
		float resampling_normal_dot_threshold = 0.5f;

		int32_t pad[3];
	};
	void set_setting(ReSTIRSetting &setting);
	RID init_uniform_set(RID shader, uint32_t set_num);

	struct SceneData {
		float projection[16];
		float inv_projection[16];
		float reprojection[16];
		float eye_offset[4];
		float inv_view_matrix[16];
		float view_matrix[16];
	};
	struct ReSTIRResource {
		RID normal_roughness_texture;
		RID depth_texture;
		RID history_depth_texture;
		RID result_texture;
		RID history_result_texture;
	};
	void process(Ref<RenderSceneBuffersRD> p_render_buffers, const ReSTIRResource &p_restir_resource, const SceneData &p_scene_data);

	void debug(const RID &debug_radiance, const RID &debug_hit_normal, const RID &debug_weight);

private:
	struct HitSample {
		float ray_direction[3];
		float distance;
		float hit_normal[3];
		float pdf;
		float out_radiance[3];
		float pad;
	};

	struct Reservoir {
		HitSample hsample;

		float weight_sum; // Processed weight sum
		float weight; // Weight of the current sample, which is the reciprocal of the PDF of the importance sampling
		uint32_t sample_count; // Processed sample count
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

	struct ReSTIRPushConstant {
		int32_t screen_size[2];
		uint32_t frame_count;
	};

	RestirShaderRD restir_shader;
	RID restir_shader_version;
	PipelineDeferredRD restir_pipelines[RESTIR_PIPELINE_MAX];

	RID scene_data_ubo;
	RID restir_setting_ubo;

	size_t current_reservoirs_size = 0;
	RID reservoirs;
	RID temporal_reservoirs;

	enum DenoisePipeline {
		DENOISE_TEMPORAL_ACCUMULATION = 0,
		DENOISE_BILATERAL_FILTER = 1,
		DENOISE_MAX = 2,
	};
};