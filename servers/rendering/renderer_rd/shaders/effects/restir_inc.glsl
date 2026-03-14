struct HitSample {
	vec3 ray_direction;
	float distance;
	vec3 hit_normal;
	float pdf;
	vec3 out_radiance;
	float validity; // 样本有效性
};

struct Reservoir {
	HitSample hsample;

	float weight_sum; // 已处理的权重和
	float weight; // 被积函数在当前采样点对应的权重，同时也就是重采样重要性采样的realPdf(SIR PDF)的倒数
	uint sample_count; // 已处理的采样总数，M
	float pad; // 填充字段，保持结构体大小一致
};

uint reservoir_index(ivec2 pos, ivec2 reservoir_size) // 到reservoir的索引
{
	return pos.x + pos.y * reservoir_size.x;
}

void clean_reservoir(inout Reservoir self) {
	self.hsample.ray_direction = vec3(0.0f);
	self.hsample.distance = 0.0f;
	self.hsample.hit_normal = vec3(0.0f);
	self.hsample.out_radiance = vec3(0.0f);
	self.hsample.pdf = 0.0f;
	self.hsample.validity = 0.0f;

	self.weight_sum = 0.0f;
	self.weight = 0.0f;
}

Reservoir new_reservoir() {
	Reservoir result;
	clean_reservoir(result);
	result.sample_count = 0u;

	return result;
}

bool update_reservoir(
		inout Reservoir self,
		in HitSample new_sample,
		float weight_new,
		float noise) {
	bool b_changed_sample = false;
	self.weight_sum += weight_new; // 更新总计权重

	if (noise < (weight_new / (self.weight_sum + 0.00001))) // 按Reservoir更新原则更新样本
	{
		self.hsample = new_sample;
		b_changed_sample = true;
	}
	return b_changed_sample;
}

bool add_sample_to_reservoir(
		inout Reservoir self,
		in HitSample hit_sample,
		float proposal_pdf,
		float noise) {
	self.sample_count = max(1u, self.sample_count + 1u); // 更新已处理采样数

	float weight_new = hit_sample.pdf / proposal_pdf; // 重要性重采样的权重，pdf/新分布下采样该点的概率
	bool b_changed_sample = update_reservoir(self, hit_sample, weight_new, noise);

	if (self.hsample.pdf <= 0) {
		clean_reservoir(self);
	} else {
		self.weight = self.weight_sum / max(self.sample_count * self.hsample.pdf, .00001f);
	}
	return b_changed_sample;
}

bool merge_reservoirs(
		inout Reservoir self,
		Reservoir other,
		float jacobian,
		float noise) {
	// 更新已处理采样数
	self.sample_count += other.sample_count;
	other.hsample.pdf *= jacobian; // 更新其他样本的pdf
	float weight = other.hsample.pdf * other.weight * other.sample_count;
	bool b_changed_sample = update_reservoir(self, other.hsample, weight, noise);

	if (self.hsample.pdf <= 0) {
		clean_reservoir(self);
	} else {
		self.weight = self.weight_sum / max(self.sample_count * self.hsample.pdf, .00001f);
	}

	return b_changed_sample;
}

bool will_change_sample_on_merge(
		Reservoir self,
		Reservoir other,
		float jacobian,
		float noise) {
	float weight_new = other.hsample.pdf * jacobian * other.weight * other.sample_count;
	float future_weight_sum = self.weight_sum + weight_new;
	return noise < weight_new / future_weight_sum;
}

layout(set = RESTIR_UNIFORM_SET, binding = 0) restrict uniform Setting {
	ivec2 reservoir_size;
}
reservoirs_setting;

layout(set = RESTIR_UNIFORM_SET, binding = 1, std430) restrict buffer Reservoirs {
	Reservoir data[];
}
reservoirs;

layout(set = RESTIR_UNIFORM_SET, binding = 2, std430) restrict buffer ReservoirsOld {
	Reservoir data[];
}
temporal_reservoirs;