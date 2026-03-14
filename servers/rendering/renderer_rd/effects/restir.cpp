/**************************************************************************/
/*  restir.cpp                                                            */
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

#include "restir.h"

#include "core/templates/vector.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

ReSTIR::ReSTIR() {
	Vector<String> restir_modes;
	restir_modes.push_back("\n#define RESTIR_PIPELINE_TEMPORAL_CLEAR\n"); // RESTIR_PIPELINE_TEMPORAL_CLEAR
	restir_modes.push_back("\n#define RESTIR_PIPELINE_TEMPORAL_REUSE\n"); // RESTIR_PIPELINE_TEMPORAL_REUSE
	restir_modes.push_back("\n#define RESTIR_PIPELINE_SPATIAL_REUSE\n"); // RESTIR_PIPELINE_SPATIAL_REUSE
	restir_modes.push_back("\n#define RESTIR_PIPELINE_INTEGRATE_AND_UPSAMPLE\n"); // RESTIR_PIPELINE_INTEGRATE_UPSAMPLE

	String defines = "\n#define RESTIR_UNIFORM_SET " + itos(RESTIR_UNIFORM_SET) + "\n";
	restir_shader.initialize(restir_modes, defines);
	restir_shader_version = restir_shader.version_create();

	for (int i = 0; i < RESTIR_PIPELINE_MAX; i++) {
		restir_pipelines[i].create_compute_pipeline(restir_shader.version_get_shader(restir_shader_version, i));
	}

	Vector<String> denoiser_modes;
	denoiser_modes.push_back("\n#define DENOISER_PIPELINE_TEMPORAL_ACCUMULATION\n");
	denoiser_modes.push_back("\n#define DENOISER_PIPELINE_BILATERAL_FILTER\n");
	denoiser_shader.initialize(denoiser_modes, defines);
	denoiser_shader_version = denoiser_shader.version_create();

	for (int i = 0; i < DENOISER_PIPELINE_MAX; i++) {
		denoiser_pipelines[i].create_compute_pipeline(denoiser_shader.version_get_shader(denoiser_shader_version, i));
	}
}

ReSTIR::~ReSTIR() {
	free_buffers();

	for (int i = 0; i < RESTIR_PIPELINE_MAX; i++) {
		restir_pipelines[i].free();
	}
	restir_shader.version_free(restir_shader_version);

	if (scene_data_ubo.is_valid()) {
		RD::get_singleton()->free_rid(scene_data_ubo);
	}

	for (int i = 0; i < DENOISER_PIPELINE_MAX; i++) {
		denoiser_pipelines[i].free();
	}
	denoiser_shader.version_free(denoiser_shader_version);
}

void ReSTIR::allocate_buffers(Size2i size) {
	if (reservoirs_setting.reservoir_size[0] == size.width && reservoirs_setting.reservoir_size[1] == size.height) {
		return;
	}
	free_buffers();
	size_t reservoirs_size = size.width * size.height;

	reservoirs = RD::get_singleton()->storage_buffer_create(sizeof(Reservoir) * reservoirs_size);
	RD::get_singleton()->set_resource_name(reservoirs, "Reservoirs");

	temporal_reservoirs = RD::get_singleton()->storage_buffer_create(sizeof(Reservoir) * reservoirs_size);
	RD::get_singleton()->set_resource_name(temporal_reservoirs, "Temporal_Reservoirs");

	reservoirs_setting.reservoir_size[0] = size.width;
	reservoirs_setting.reservoir_size[1] = size.height;

	reservoirs_setting_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(ReservoirsSetting));
	RD::get_singleton()->buffer_update(reservoirs_setting_ubo, 0, sizeof(ReservoirsSetting), &reservoirs_setting);

	// ReSTIR Temporal Clear
	RD::get_singleton()->draw_command_begin_label("ReSTIR Temporal Clear");

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	int32_t mode = RESTIR_PIPELINE_TEMPORAL_CLEAR;
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, restir_pipelines[mode].get_rid());

	RID uniform_set = init_restir_uniform_set(restir_shader.version_get_shader(restir_shader_version, mode), RESTIR_UNIFORM_SET);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, RESTIR_UNIFORM_SET);
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, size.width, size.height, 1);
	RD::get_singleton()->compute_list_end();

	RD::get_singleton()->draw_command_end_label();
}

void ReSTIR::free_buffers() {
	if (reservoirs.is_valid()) {
		RD::get_singleton()->free_rid(reservoirs);
	}
	if (temporal_reservoirs.is_valid()) {
		RD::get_singleton()->free_rid(temporal_reservoirs);
	}
	if (reservoirs_setting_ubo.is_valid()) {
		RD::get_singleton()->free_rid(reservoirs_setting_ubo);
	}
	reservoirs_setting = {{0, 0}, {0, 0}};
}

RID ReSTIR::init_restir_uniform_set(RID shader, uint32_t set_num) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL_V(uniform_set_cache, RID());
	
	RD::Uniform u_setting(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, reservoirs_setting_ubo);
	RD::Uniform u_buffer(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, reservoirs);
	RD::Uniform u_temp_buffer(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, temporal_reservoirs);
	return uniform_set_cache->get_cache(shader, set_num, u_setting, u_buffer, u_temp_buffer);
}

void ReSTIR::set_setting(ReSTIRSetting &setting) {
	restir_setting = setting;
}

void ReSTIR::process(Ref<RenderSceneBuffersRD> p_render_buffers, const ReSTIRResource &p_restir_resource, const SceneData &p_scene_data) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	Vector2i internal_size = p_render_buffers->get_internal_size();

	{
		if (scene_data_ubo.is_null()) {
			scene_data_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(SceneData));
		} 

		RD::get_singleton()->buffer_update(scene_data_ubo, 0, sizeof(SceneData), &p_scene_data);
	}

	{
		RD::get_singleton()->draw_command_begin_label("ReSTIR Temporal Reuse");

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		int32_t mode = RESTIR_PIPELINE_TEMPORAL_REUSE;
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, restir_pipelines[mode].get_rid());

		ReSTIRPushConstant push_constant{};
		push_constant.screen_size[0] = MAX(1, internal_size.width);
		push_constant.screen_size[1] = MAX(1, internal_size.height);
		push_constant.frame_count = RSG::rasterizer->get_frame_number();
		push_constant.temporal_pos_threshold = restir_setting.temporal_pos_threshold;
		push_constant.spatial_resampling_kernel_radius = restir_setting.spatial_resampling_kernel_radius;
		push_constant.spatial_num_samples = restir_setting.spatial_num_samples;
		push_constant.spatial_resampling_pass_index = restir_setting.spatial_resampling_pass_index;
		push_constant.spatial_resampling_occlusion_screen_trace_distance = restir_setting.spatial_resampling_occlusion_screen_trace_distance;
		push_constant.resampling_depth_error_threshold = restir_setting.resampling_depth_error_threshold;
		push_constant.resampling_normal_dot_threshold = restir_setting.resampling_normal_dot_threshold;

		RID shader = restir_shader.version_get_shader(restir_shader_version, mode);

		RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, scene_data_ubo);
		RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 1, p_restir_resource.depth_texture);
		RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_IMAGE, 2, p_restir_resource.normal_roughness_texture);
		RD::Uniform u_history_depth(RD::UNIFORM_TYPE_IMAGE, 3, p_restir_resource.history_depth_texture);

		RID uniform_set = init_restir_uniform_set(restir_shader.version_get_shader(restir_shader_version, mode), RESTIR_UNIFORM_SET);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_scene_data, u_depth, u_normal_roughness, u_history_depth), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, RESTIR_UNIFORM_SET);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, reservoirs_setting.reservoir_size[0], reservoirs_setting.reservoir_size[1], 1);

		RD::get_singleton()->compute_list_end();

		RD::get_singleton()->draw_command_end_label();
	}

	{
		RD::get_singleton()->draw_command_begin_label("ReSTIR Spatial Reuse");

		ReSTIRPushConstant push_constant{};
		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		int32_t mode = RESTIR_PIPELINE_SPATIAL_REUSE;
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, restir_pipelines[mode].get_rid());

		push_constant.screen_size[0] = MAX(1, internal_size.width);
		push_constant.screen_size[1] = MAX(1, internal_size.height);
		push_constant.frame_count = RSG::rasterizer->get_frame_number();
		push_constant.temporal_pos_threshold = restir_setting.temporal_pos_threshold;
		push_constant.spatial_resampling_kernel_radius = restir_setting.spatial_resampling_kernel_radius;
		push_constant.spatial_num_samples = restir_setting.spatial_num_samples;
		push_constant.spatial_resampling_pass_index = restir_setting.spatial_resampling_pass_index;
		push_constant.spatial_resampling_occlusion_screen_trace_distance = restir_setting.spatial_resampling_occlusion_screen_trace_distance;
		push_constant.resampling_depth_error_threshold = restir_setting.resampling_depth_error_threshold;
		push_constant.resampling_normal_dot_threshold = restir_setting.resampling_normal_dot_threshold;

		RID shader = restir_shader.version_get_shader(restir_shader_version, mode);

		RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, scene_data_ubo);
		RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 1, p_restir_resource.depth_texture);
		RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_IMAGE, 2, p_restir_resource.normal_roughness_texture);

		RID uniform_set = init_restir_uniform_set(restir_shader.version_get_shader(restir_shader_version, mode), RESTIR_UNIFORM_SET);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_scene_data, u_depth, u_normal_roughness), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, RESTIR_UNIFORM_SET);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, reservoirs_setting.reservoir_size[0], reservoirs_setting.reservoir_size[1], 1);

		RD::get_singleton()->compute_list_end();

		RD::get_singleton()->draw_command_end_label();
	}

	{
		RD::get_singleton()->draw_command_begin_label("ReSTIR Integrate and Upsample");

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		ReSTIRPushConstant push_constant{};
		push_constant.screen_size[0] = MAX(1, internal_size.width);
		push_constant.screen_size[1] = MAX(1, internal_size.height);
		push_constant.frame_count = RSG::rasterizer->get_frame_number();
		push_constant.temporal_pos_threshold = restir_setting.temporal_pos_threshold;
		push_constant.spatial_resampling_kernel_radius = restir_setting.spatial_resampling_kernel_radius;
		push_constant.spatial_num_samples = restir_setting.spatial_num_samples;
		push_constant.spatial_resampling_pass_index = restir_setting.spatial_resampling_pass_index;
		push_constant.spatial_resampling_occlusion_screen_trace_distance = restir_setting.spatial_resampling_occlusion_screen_trace_distance;
		push_constant.resampling_depth_error_threshold = restir_setting.resampling_depth_error_threshold;
		push_constant.resampling_normal_dot_threshold = restir_setting.resampling_normal_dot_threshold;

		int32_t mode = RESTIR_PIPELINE_INTEGRATE_AND_UPSAMPLE;

		RID shader = restir_shader.version_get_shader(restir_shader_version, mode);

		RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, scene_data_ubo);
		RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 1, p_restir_resource.depth_texture);
		RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_IMAGE, 2, p_restir_resource.normal_roughness_texture);
		RD::Uniform u_out_diffuse(RD::UNIFORM_TYPE_IMAGE, 3, p_restir_resource.diffuse_texture);

		RID uniform_set = init_restir_uniform_set(restir_shader.version_get_shader(restir_shader_version, mode), RESTIR_UNIFORM_SET);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, restir_pipelines[mode].get_rid());
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_scene_data, u_depth, u_normal_roughness, u_out_diffuse), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, RESTIR_UNIFORM_SET);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.screen_size[0], push_constant.screen_size[1], 1);

		RD::get_singleton()->compute_list_end();

		RD::get_singleton()->draw_command_end_label();
	}
}

void ReSTIR::set_denoiser_setting(ReSTIRDenoiserSetting &setting) {
	denoiser_setting = setting;
}

void ReSTIR::process_denoise(Ref<RenderSceneBuffersRD> p_render_buffers, const ReSTIRDenoiserResource &p_denoiser_resource, const SceneData &p_scene_data) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	Vector2i internal_size = p_render_buffers->get_internal_size();

	{
		if (scene_data_ubo.is_null()) {
			scene_data_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(SceneData));
		} 

		RD::get_singleton()->buffer_update(scene_data_ubo, 0, sizeof(SceneData), &p_scene_data);
	}

	{
		RD::get_singleton()->draw_command_begin_label("Denoise Temporal Accumulation");

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		int32_t mode = DENOISER_PIPELINE_TEMPORAL_ACCUMULATION;
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, denoiser_pipelines[mode].get_rid());

		DenoiserPushConstant push_constant{};
		push_constant.screen_size[0] = MAX(1, internal_size.width);
		push_constant.screen_size[1] = MAX(1, internal_size.height);
		push_constant.frame_count = RSG::rasterizer->get_frame_number();
		push_constant.max_frames_accumulated = denoiser_setting.max_frames_accumulated;
		push_constant.history_distance_threshold = denoiser_setting.history_distance_threshold;
		push_constant.bilateral_filter_spatial_kernel_radius = denoiser_setting.bilateral_filter_spatial_kernel_radius;
		push_constant.bilateral_filter_num_samples = denoiser_setting.bilateral_filter_num_samples;
		push_constant.bilateral_filter_depth_weight_scale = denoiser_setting.bilateral_filter_depth_weight_scale;
		push_constant.bilateral_filter_normal_angle_threshold_scale = denoiser_setting.bilateral_filter_normal_angle_threshold_scale;
		push_constant.bilateral_filter_strong_blur_variance_threshold = denoiser_setting.bilateral_filter_strong_blur_variance_threshold;
		push_constant.disocclusion_variance = denoiser_setting.disocclusion_variance;

		RID shader = denoiser_shader.version_get_shader(denoiser_shader_version, mode);

		RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, scene_data_ubo);
		RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 1, p_denoiser_resource.depth_texture);
		RD::Uniform u_history_depth(RD::UNIFORM_TYPE_IMAGE, 2, p_denoiser_resource.history_depth_texture);
		RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_IMAGE, 3, p_denoiser_resource.normal_roughness_texture);
		RD::Uniform u_history_num_frames_accumulated(RD::UNIFORM_TYPE_IMAGE, 4, p_denoiser_resource.history_num_frames_accumulated_texture);
		RD::Uniform u_out_num_frames_accumulated(RD::UNIFORM_TYPE_IMAGE, 5, p_denoiser_resource.out_num_frames_accumulated_texture);
		RD::Uniform u_diffuse_indirect(RD::UNIFORM_TYPE_IMAGE, 6, p_denoiser_resource.out_diffuse_texture);
		RD::Uniform u_history_diffuse_indirect(RD::UNIFORM_TYPE_IMAGE, 7, p_denoiser_resource.history_diffuse_texture);
		RD::Uniform u_out_diffuse_indirect(RD::UNIFORM_TYPE_IMAGE, 8, p_denoiser_resource.diffuse_texture);

		RID uniform_set = uniform_set_cache->get_cache(shader, 0, u_scene_data, u_depth, u_history_depth, u_normal_roughness, u_history_num_frames_accumulated, u_out_num_frames_accumulated, u_diffuse_indirect, u_history_diffuse_indirect, u_out_diffuse_indirect);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.screen_size[0], push_constant.screen_size[1], 1);

		RD::get_singleton()->compute_list_end();

		RD::get_singleton()->draw_command_end_label();
	}

	{
		RD::get_singleton()->draw_command_begin_label("Denoise Bilateral Filter");

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		int32_t mode = DENOISER_PIPELINE_BILATERAL_FILTER;
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, denoiser_pipelines[mode].get_rid());

		DenoiserPushConstant push_constant{};
		push_constant.screen_size[0] = MAX(1, internal_size.width);
		push_constant.screen_size[1] = MAX(1, internal_size.height);
		push_constant.frame_count = RSG::rasterizer->get_frame_number();
		push_constant.max_frames_accumulated = denoiser_setting.max_frames_accumulated;
		push_constant.history_distance_threshold = denoiser_setting.history_distance_threshold;
		push_constant.bilateral_filter_spatial_kernel_radius = denoiser_setting.bilateral_filter_spatial_kernel_radius;
		push_constant.bilateral_filter_num_samples = denoiser_setting.bilateral_filter_num_samples;
		push_constant.bilateral_filter_depth_weight_scale = denoiser_setting.bilateral_filter_depth_weight_scale;
		push_constant.bilateral_filter_normal_angle_threshold_scale = denoiser_setting.bilateral_filter_normal_angle_threshold_scale;
		push_constant.bilateral_filter_strong_blur_variance_threshold = denoiser_setting.bilateral_filter_strong_blur_variance_threshold;
		push_constant.disocclusion_variance = denoiser_setting.disocclusion_variance;

		RID shader = denoiser_shader.version_get_shader(denoiser_shader_version, mode);

		RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, scene_data_ubo);
		RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 1, p_denoiser_resource.depth_texture);
		RD::Uniform u_history_depth(RD::UNIFORM_TYPE_IMAGE, 2, p_denoiser_resource.history_depth_texture);
		RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_IMAGE, 3, p_denoiser_resource.normal_roughness_texture);
		RD::Uniform u_history_num_frames_accumulated(RD::UNIFORM_TYPE_IMAGE, 4, p_denoiser_resource.history_num_frames_accumulated_texture);
		RD::Uniform u_out_num_frames_accumulated(RD::UNIFORM_TYPE_IMAGE, 5, p_denoiser_resource.out_num_frames_accumulated_texture);
		RD::Uniform u_diffuse_indirect(RD::UNIFORM_TYPE_IMAGE, 6, p_denoiser_resource.diffuse_texture);
		RD::Uniform u_history_diffuse_indirect(RD::UNIFORM_TYPE_IMAGE, 7, p_denoiser_resource.history_diffuse_texture);
		RD::Uniform u_out_diffuse_indirect(RD::UNIFORM_TYPE_IMAGE, 8, p_denoiser_resource.out_diffuse_texture);

		RID uniform_set = uniform_set_cache->get_cache(shader, 0, u_scene_data, u_depth, u_history_depth, u_normal_roughness, u_history_num_frames_accumulated, u_out_num_frames_accumulated, u_diffuse_indirect, u_history_diffuse_indirect, u_out_diffuse_indirect);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.screen_size[0], push_constant.screen_size[1], 1);

		RD::get_singleton()->compute_list_end();

		RD::get_singleton()->draw_command_end_label();
	}
}