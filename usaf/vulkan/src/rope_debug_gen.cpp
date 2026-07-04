#include "vulkan_core.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <cmath>
using namespace usaf::vkcore;

float h2f(uint16_t v) {
    int s = (v >> 15) & 1, e = (v >> 10) & 0x1F, m = v & 0x3FF;
    float f;
    if (e == 0) f = ldexpf((float)m, -24);
    else if (e < 31) f = ldexpf((float)(m | 0x400), e - 25);
    else f = (m == 0) ? INFINITY : NAN;
    return s ? -f : f;
}

int main() {
    auto ctx = init_compute("rope_debug");
    uint32_t B = 1, nH = 1, nKV = 1, S = 4, hd = 4;
    size_t q_bytes = B * nH * S * hd * 2;
    size_t k_bytes = B * nKV * S * hd * 2;
    size_t cs_bytes = S * hd * 2;

    auto read_bin = [](const char* p, std::vector<uint8_t>& v) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        size_t sz = f.tellg(); f.seekg(0); v.resize(sz);
        f.read(reinterpret_cast<char*>(v.data()), sz);
    };
    std::vector<uint8_t> inp, exp_data;
    read_bin("test_data/rope_debug.bin", inp);
    read_bin("test_data/rope_debug_exp.bin", exp_data);

    auto buf_q = create_buffer(ctx, q_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_k = create_buffer(ctx, k_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_c = create_buffer(ctx, cs_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_s = create_buffer(ctx, cs_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_qo = create_buffer(ctx, q_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_ko = create_buffer(ctx, k_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    size_t off = 0;
    upload_buffer(ctx, buf_q, inp.data() + off, q_bytes); off += q_bytes;
    upload_buffer(ctx, buf_k, inp.data() + off, k_bytes); off += k_bytes;
    upload_buffer(ctx, buf_c, inp.data() + off, cs_bytes); off += cs_bytes;
    upload_buffer(ctx, buf_s, inp.data() + off, cs_bytes);

    auto shader = load_shader(ctx, "spirv/rope_fp16.spv", "main");
    std::vector<vk::DescriptorSetLayoutBinding> bnd = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    auto pip = create_compute_pipeline(ctx, shader, bnd, {{vk::ShaderStageFlagBits::eCompute, 0, 20}});

    std::vector<vk::DescriptorBufferInfo> dbi = {
        {buf_q.buffer, 0, q_bytes}, {buf_k.buffer, 0, k_bytes},
        {buf_c.buffer, 0, cs_bytes}, {buf_s.buffer, 0, cs_bytes},
        {buf_qo.buffer, 0, q_bytes}, {buf_ko.buffer, 0, k_bytes},
    };
    std::vector<vk::WriteDescriptorSet> writes(6);
    for (int i = 0; i < 6; i++)
        writes[i].setDstSet(pip.desc_set).setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi[i]);
    update_descriptor_set(ctx, pip, writes);

    struct { uint32_t B, nH, nKV, S, hd; } pc = {B, nH, nKV, S, hd};
    uint32_t g = (std::max(B * nH * S, B * nKV * S) + 255) / 256;
    dispatch(ctx, pip, g, 1, 1, &pc, sizeof(pc));

    std::vector<uint8_t> r_q(q_bytes), r_k(k_bytes);
    download_buffer(ctx, buf_qo, r_q.data(), q_bytes);
    download_buffer(ctx, buf_ko, r_k.data(), k_bytes);

    printf("Expected Q:\n");
    auto* e16 = reinterpret_cast<uint16_t*>(exp_data.data());
    for (uint32_t i = 0; i < B * nH * S * hd; i++) {
        if (i % hd == 0) printf("  tok%d: ", i / hd);
        printf("%.4f ", h2f(e16[i]));
        if (i % hd == hd - 1) printf("\n");
    }
    printf("Vulkan Q:\n");
    auto* r16 = reinterpret_cast<uint16_t*>(r_q.data());
    for (uint32_t i = 0; i < B * nH * S * hd; i++) {
        if (i % hd == 0) printf("  tok%d: ", i / hd);
        printf("%.4f ", h2f(r16[i]));
        if (i % hd == hd - 1) printf("\n");
    }
    printf("Vulkan K:\n");
    r16 = reinterpret_cast<uint16_t*>(r_k.data());
    for (uint32_t i = 0; i < B * nKV * S * hd; i++) {
        if (i % hd == 0) printf("  tok%d: ", i / hd);
        printf("%.4f ", h2f(r16[i]));
        if (i % hd == hd - 1) printf("\n");
    }

    for (auto* b : {&buf_q, &buf_k, &buf_c, &buf_s, &buf_qo, &buf_ko}) destroy_buffer(ctx, *b);
    destroy_pipeline(ctx, pip);
    destroy_compute(ctx);
    return 0;
}
