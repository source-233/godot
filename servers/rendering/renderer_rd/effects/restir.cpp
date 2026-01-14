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
}

ReSTIR::~ReSTIR() {
	free_buffers();

	for (int i = 0; i < RESTIR_PIPELINE_MAX; i++) {
		restir_pipelines[i].free();
	}
	restir_shader.version_free(restir_shader_version);

	if (restir_setting_ubo.is_valid()) {
		RD::get_singleton()->free_rid(restir_setting_ubo);
	}
	if (scene_data_ubo.is_valid()) {
		RD::get_singleton()->free_rid(scene_data_ubo);
	}
}

void ReSTIR::allocate_buffers(Size2i size) {
	size_t reservoirs_size = size.width * size.height;
	if (current_reservoirs_size == reservoirs_size) {
		return;
	}
	free_buffers();

	reservoirs = RD::get_singleton()->storage_buffer_create(sizeof(Reservoir) * reservoirs_size);
	RD::get_singleton()->set_resource_name(reservoirs, "Reservoirs");

	temporal_reservoirs = RD::get_singleton()->storage_buffer_create(sizeof(Reservoir) * reservoirs_size);
	RD::get_singleton()->set_resource_name(temporal_reservoirs, "Temporal_Reservoirs");

	current_reservoirs_size = reservoirs_size;

	RD::get_singleton()->draw_command_begin_label("ReSTIR Temporal Clear");

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	int32_t mode = RESTIR_PIPELINE_TEMPORAL_CLEAR;
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, restir_pipelines[mode].get_rid());

	RID uniform_set = init_uniform_set(restir_shader.version_get_shader(restir_shader_version, mode), RESTIR_UNIFORM_SET);
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
	current_reservoirs_size = 0;
}

void ReSTIR::set_setting(ReSTIRSetting &setting) {
	if (restir_setting_ubo.is_null()) {
		restir_setting_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(ReSTIRSetting));
		RD::get_singleton()->set_resource_name(restir_setting_ubo, "ReSTIRSetting");
	}
	RD::get_singleton()->buffer_update(restir_setting_ubo, 0, sizeof(ReSTIRSetting), &setting);
}

RID ReSTIR::init_uniform_set(RID shader, uint32_t set_num) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL_V(uniform_set_cache, RID());
	
	RD::Uniform u_restir_setting(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, restir_setting_ubo);
	RD::Uniform u_buffer(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, reservoirs);
	RD::Uniform u_temp_buffer(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, temporal_reservoirs);
	return uniform_set_cache->get_cache(shader, set_num, u_restir_setting, u_buffer, u_temp_buffer);
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

		RID shader = restir_shader.version_get_shader(restir_shader_version, mode);

		RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, scene_data_ubo);
		RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 1, p_restir_resource.depth_texture);
		RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_IMAGE, 2, p_restir_resource.normal_roughness_texture);
		RD::Uniform u_history_depth(RD::UNIFORM_TYPE_IMAGE, 3, p_restir_resource.history_depth_texture);

		RID uniform_set = init_uniform_set(restir_shader.version_get_shader(restir_shader_version, mode), RESTIR_UNIFORM_SET);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_scene_data, u_depth, u_normal_roughness, u_history_depth), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, RESTIR_UNIFORM_SET);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.screen_size[0], push_constant.screen_size[1], 1);

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

		RID shader = restir_shader.version_get_shader(restir_shader_version, mode);

		RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, scene_data_ubo);
		RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 1, p_restir_resource.depth_texture);
		RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_IMAGE, 2, p_restir_resource.normal_roughness_texture);

		RID uniform_set = init_uniform_set(restir_shader.version_get_shader(restir_shader_version, mode), RESTIR_UNIFORM_SET);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_scene_data, u_depth, u_normal_roughness), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, RESTIR_UNIFORM_SET);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.screen_size[0], push_constant.screen_size[1], 1);

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

		int32_t mode = RESTIR_PIPELINE_INTEGRATE_AND_UPSAMPLE;

		RID shader = restir_shader.version_get_shader(restir_shader_version, mode);

		RD::Uniform u_scene_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, scene_data_ubo);
		RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 1, p_restir_resource.depth_texture);
		RD::Uniform u_normal_roughness(RD::UNIFORM_TYPE_IMAGE, 2, p_restir_resource.normal_roughness_texture);
		RD::Uniform u_out_diffuse(RD::UNIFORM_TYPE_IMAGE, 3, p_restir_resource.result_texture);

		RID uniform_set = init_uniform_set(restir_shader.version_get_shader(restir_shader_version, mode), RESTIR_UNIFORM_SET);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, restir_pipelines[mode].get_rid());
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_scene_data, u_depth, u_normal_roughness, u_out_diffuse), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, RESTIR_UNIFORM_SET);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.screen_size[0], push_constant.screen_size[1], 1);

		RD::get_singleton()->compute_list_end();

		RD::get_singleton()->draw_command_end_label();
	}
}

