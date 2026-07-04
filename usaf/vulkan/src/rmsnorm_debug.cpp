#include "vulkan_core.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

using namespace usaf::vkcore;

int main() {
    auto ctx = init_compute("rmsnorm_debug");

    // Tiny test: 1 row, 8 cols
    uint32_t rows = 1, cols = 8;
    float eps = 1e-6f;

    // Input: [0.5, 1.0, -0.5, -1.0, 2.0, 0.0, -2.0, 0.25]
    float x_in[8] = {0.5f, 1.0f, -0.5f, -1.0f, 2.0f, 0.0f, -2.0f, 0.25f};
    float w_in[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Compute expected manually in fp32
    float sum_sq = 0;
    for (int i = 0; i < 8; i++) sum_sq += x_in[i] * x_in[i];
    float mean_sq = sum_sq / 8.0f;
    float rms = 1.0f / sqrtf(mean_sq + eps);
    printf("Expected: sum_sq=%.6f rms=%.6f\n", sum_sq, rms);
    for (int i = 0; i < 8; i++) printf("  y[%d] = %.6f\n", i, x_in[i] * rms * w_in[i]);

    // Convert to fp16 for GPU
    auto to_fp16 = [](float v) -> uint16_t {
        uint32_t f; memcpy(&f, &v, sizeof(f));
        uint32_t s = (f >> 16) & 0x8000;
        int e = (int)((f >> 23) & 0xFF) - 127 + 15;
        uint32_t m = (f >> 13) & 0x3FF;
        if (e >= 31) { e = 31; m = 0; }
        else if (e <= 0) { e = 0; m = 0; }
        return (uint16_t)(s | ((uint32_t)e << 10) | m);
    };

    std::vector<uint8_t> x_data(rows * cols * 2);
    std::vector<uint8_t> w_data(cols * 2);
    for (int i = 0; i < 8; i++) {
        uint16_t v = to_fp16(x_in[i]);
        memcpy(x_data.data() + i * 2, &v, 2);
        v = to_fp16(w_in[i]);
        memcpy(w_data.data() + i * 2, &v, 2);
    }

    auto buf_x = create_buffer(ctx, x_data.size(),
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_w = create_buffer(ctx, w_data.size(),
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_y = create_buffer(ctx, x_data.size(),
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    upload_buffer(ctx, buf_x, x_data.data(), x_data.size());
    upload_buffer(ctx, buf_w, w_data.data(), w_data.size());
    // Zero initialize output to avoid stale values
    std::vector<uint8_t> zeros(x_data.size(), 0);
    upload_buffer(ctx, buf_y, zeros.data(), zeros.size());

    auto shader = load_shader(ctx, "spirv/rmsnorm_fp16.spv", "main");
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    std::vector<vk::PushConstantRange> push = {{vk::ShaderStageFlagBits::eCompute, 0, 12}};
    auto pipeline = create_compute_pipeline(ctx, shader, bindings, push);

    vk::DescriptorBufferInfo dbi_x(buf_x.buffer, 0, x_data.size());
    vk::DescriptorBufferInfo dbi_w(buf_w.buffer, 0, w_data.size());
    vk::DescriptorBufferInfo dbi_y(buf_y.buffer, 0, x_data.size());
    std::vector<vk::WriteDescriptorSet> writes(3);
    writes[0].setDstSet(pipeline.desc_set).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_x);
    writes[1].setDstSet(pipeline.desc_set).setDstBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_w);
    writes[2].setDstSet(pipeline.desc_set).setDstBinding(2).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_y);
    update_descriptor_set(ctx, pipeline, writes);

    struct { uint32_t r, c; float e; } pc = {rows, cols, eps};
    dispatch(ctx, pipeline, rows, 1, 1, &pc, sizeof(pc));

    std::vector<uint8_t> result(x_data.size());
    download_buffer(ctx, buf_y, result.data(), result.size());

    printf("Vulkan output:\n");
    for (int i = 0; i < 8; i++) {
        uint16_t v;
        memcpy(&v, result.data() + i * 2, 2);
        uint32_t s = (v >> 15) & 1, e = (v >> 10) & 0x1F, m = v & 0x3FF;
        float fv;
        if (e == 0) fv = ldexpf((float)m, -24);
        else if (e < 31) fv = ldexpf((float)(m | 0x400), e - 25);
        else fv = NAN;
        if (s) fv = -fv;
        printf("  y[%d] = %.6f\n", i, fv);
    }

    destroy_pipeline(ctx, pipeline);
    destroy_buffer(ctx, buf_x);
    destroy_buffer(ctx, buf_w);
    destroy_buffer(ctx, buf_y);
    destroy_compute(ctx);
    return 0;
}
