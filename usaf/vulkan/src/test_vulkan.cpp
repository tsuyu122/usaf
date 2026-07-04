#include "vulkan_core.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace usaf::vkcore;

int main() {
    try {
        auto ctx = init_compute("test_vulkan");

        uint32_t N = 1024 * 1024;
        std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);

        auto buf_a = create_buffer(ctx, N * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        auto buf_b = create_buffer(ctx, N * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        auto buf_c = create_buffer(ctx, N * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        upload_buffer(ctx, buf_a, a.data(), N * sizeof(float));
        upload_buffer(ctx, buf_b, b.data(), N * sizeof(float));

        auto shader = load_shader(ctx, "spirv/test_add.spv", "main");

        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };

        std::vector<vk::PushConstantRange> push = {
            {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)}
        };

        auto pipeline = create_compute_pipeline(ctx, shader, bindings, push);

        vk::DescriptorBufferInfo dbi_a(buf_a.buffer, 0, N * sizeof(float));
        vk::DescriptorBufferInfo dbi_b(buf_b.buffer, 0, N * sizeof(float));
        vk::DescriptorBufferInfo dbi_c(buf_c.buffer, 0, N * sizeof(float));

        std::vector<vk::WriteDescriptorSet> writes(3);
        writes[0].setDstSet(pipeline.desc_set).setDstBinding(0).setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_a);
        writes[1].setDstSet(pipeline.desc_set).setDstBinding(1).setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_b);
        writes[2].setDstSet(pipeline.desc_set).setDstBinding(2).setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_c);
        update_descriptor_set(ctx, pipeline, writes);

        // Benchmark
        const int WARMUP = 10, BENCH = 100;
        uint32_t gx = (N + 255) / 256;
        for (int i = 0; i < WARMUP; ++i)
            dispatch(ctx, pipeline, gx, 1, 1, &N, sizeof(N));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH; ++i)
            dispatch(ctx, pipeline, gx, 1, 1, &N, sizeof(N));
        auto t1 = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / BENCH;
        double bw = (N * 3 * sizeof(float)) / (us * 1e-6) / 1e9;
        std::cout << "[TEST] add " << N << " floats: " << us << " us | " << bw << " GB/s" << std::endl;

        download_buffer(ctx, buf_c, c.data(), N * sizeof(float));

        float max_err = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            max_err = std::max(max_err, std::abs(c[i] - 3.0f));
        }
        std::cout << "[TEST] max error: " << max_err << (max_err < 1e-5 ? " PASS" : " FAIL") << std::endl;

        destroy_pipeline(ctx, pipeline);
        destroy_buffer(ctx, buf_a);
        destroy_buffer(ctx, buf_b);
        destroy_buffer(ctx, buf_c);
        destroy_compute(ctx);

        return max_err < 1e-5 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return 1;
    }
}
