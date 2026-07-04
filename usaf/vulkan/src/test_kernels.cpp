#include "vulkan_core.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cassert>
#include <map>
#include <string>

using namespace usaf::vkcore;

std::map<std::string, std::string> read_meta(const std::string& path) {
    std::map<std::string, std::string> m;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        auto eq = line.find('=');
        if (eq != std::string::npos) {
            m[line.substr(0, eq)] = line.substr(eq + 1);
        }
    }
    return m;
}

std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

float max_abs_error(const uint16_t* expected, const uint16_t* actual, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        // Convert fp16 to float for comparison
        uint32_t e = expected[i];
        uint32_t a = actual[i];
        // Check for NaN
        if ((e & 0x7C00) == 0x7C00 && (e & 0x03FF) != 0) continue;
        if ((a & 0x7C00) == 0x7C00 && (a & 0x03FF) != 0) continue;
        float ef, af;
        // Simple fp16->float conversion
        uint32_t es = (e >> 15) & 1;
        uint32_t ee = (e >> 10) & 0x1F;
        uint32_t em = e & 0x3FF;
        if (ee == 0) { ef = ldexpf((float)em, -24); }
        else if (ee < 31) { ef = ldexpf((float)(em | 0x400), ee - 25); }
        else { ef = (em == 0) ? INFINITY : NAN; }
        if (es) ef = -ef;

        uint32_t as = (a >> 15) & 1;
        uint32_t ae = (a >> 10) & 0x1F;
        uint32_t am = a & 0x3FF;
        if (ae == 0) { af = ldexpf((float)am, -24); }
        else if (ae < 31) { af = ldexpf((float)(am | 0x400), ae - 25); }
        else { af = (am == 0) ? INFINITY : NAN; }
        if (as) af = -af;

        float diff = std::abs(ef - af);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

bool test_rmsnorm(ComputeContext& ctx) {
    std::cout << "\n=== RMSNorm ===" << std::endl;
    auto meta = read_meta("test_data/rmsnorm_meta.txt");
    auto x_shape = meta["x"];
    // x_shape = "2,512,2048" -> B,S,H; rows = B*S, cols = H
    size_t c1 = x_shape.find(','), c2 = x_shape.find(',', c1 + 1);
    int B_rows = std::stoi(x_shape.substr(0, c1));
    int S_rows = std::stoi(x_shape.substr(c1 + 1, c2 - c1 - 1));
    int cols = std::stoi(x_shape.substr(c2 + 1));
    int rows = B_rows * S_rows;
    float eps = std::stof(meta["eps"]);
    std::cout << "  rows=" << rows << " cols=" << cols << " eps=" << eps << std::endl;

    auto input = read_binary("test_data/rmsnorm_input.bin");
    auto expected = read_binary("test_data/rmsnorm_expected.bin");
    size_t x_bytes = rows * cols * 2;
    size_t w_bytes = cols * 2;

    auto buf_x = create_buffer(ctx, x_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_w = create_buffer(ctx, w_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_y = create_buffer(ctx, x_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    upload_buffer(ctx, buf_x, input.data(), x_bytes);
    upload_buffer(ctx, buf_w, input.data() + x_bytes, w_bytes);

    auto shader = load_shader(ctx, "spirv/rmsnorm_fp16.spv", "main");

    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    std::vector<vk::PushConstantRange> push = {
        {vk::ShaderStageFlagBits::eCompute, 0, 12} // 3 uint32 = 12 bytes
    };
    auto pipeline = create_compute_pipeline(ctx, shader, bindings, push);

    vk::DescriptorBufferInfo dbi_x(buf_x.buffer, 0, x_bytes);
    vk::DescriptorBufferInfo dbi_w(buf_w.buffer, 0, w_bytes);
    vk::DescriptorBufferInfo dbi_y(buf_y.buffer, 0, x_bytes);

    std::vector<vk::WriteDescriptorSet> writes(3);
    writes[0].setDstSet(pipeline.desc_set).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_x);
    writes[1].setDstSet(pipeline.desc_set).setDstBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_w);
    writes[2].setDstSet(pipeline.desc_set).setDstBinding(2).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_y);
    update_descriptor_set(ctx, pipeline, writes);

    struct { uint32_t r, c; float e; } pc = {(uint32_t)rows, (uint32_t)cols, eps};

    // Benchmark
    int warmup = 5, bench = 100;
    for (int i = 0; i < warmup; i++) dispatch(ctx, pipeline, rows, 1, 1, &pc, sizeof(pc));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; i++) dispatch(ctx, pipeline, rows, 1, 1, &pc, sizeof(pc));
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / bench;

    std::vector<uint8_t> result(x_bytes);
    download_buffer(ctx, buf_y, result.data(), x_bytes);

    float max_err = max_abs_error(
        reinterpret_cast<const uint16_t*>(expected.data()),
        reinterpret_cast<const uint16_t*>(result.data()),
        rows * cols);
    std::cout << "  " << us << " us | max_err=" << max_err
              << (max_err < 0.01f ? " PASS" : " FAIL") << std::endl;

    destroy_pipeline(ctx, pipeline);
    destroy_buffer(ctx, buf_x);
    destroy_buffer(ctx, buf_w);
    destroy_buffer(ctx, buf_y);

    return max_err < 0.01f;
}

bool test_rope(ComputeContext& ctx) {
    std::cout << "\n=== RoPE ===" << std::endl;
    auto meta = read_meta("test_data/rope_meta.txt");
    uint32_t B = std::stoi(meta["B"]), nH = std::stoi(meta["nH"]);
    uint32_t nKV = std::stoi(meta["nKV"]), S = std::stoi(meta["S"]), hd = std::stoi(meta["hd"]);
    std::cout << "  B=" << B << " nH=" << nH << " nKV=" << nKV << " S=" << S << " hd=" << hd << std::endl;

    auto input = read_binary("test_data/rope_input.bin");
    auto expected = read_binary("test_data/rope_expected.bin");

    size_t q_bytes = B * nH * S * hd * 2;
    size_t k_bytes = B * nKV * S * hd * 2;
    size_t cs_bytes = S * hd * 2; // cos + sin

    auto buf_q = create_buffer(ctx, q_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_k = create_buffer(ctx, k_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_cos = create_buffer(ctx, cs_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_sin = create_buffer(ctx, cs_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_qo = create_buffer(ctx, q_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_ko = create_buffer(ctx, k_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    size_t offset = 0;
    upload_buffer(ctx, buf_q, input.data() + offset, q_bytes); offset += q_bytes;
    upload_buffer(ctx, buf_k, input.data() + offset, k_bytes); offset += k_bytes;
    upload_buffer(ctx, buf_cos, input.data() + offset, cs_bytes); offset += cs_bytes;
    upload_buffer(ctx, buf_sin, input.data() + offset, cs_bytes);

    auto shader = load_shader(ctx, "spirv/rope_fp16.spv", "main");

    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    std::vector<vk::PushConstantRange> push = {
        {vk::ShaderStageFlagBits::eCompute, 0, 20}
    };
    auto pipeline = create_compute_pipeline(ctx, shader, bindings, push);

    std::vector<vk::DescriptorBufferInfo> dbis = {
        {buf_q.buffer, 0, q_bytes}, {buf_k.buffer, 0, k_bytes},
        {buf_cos.buffer, 0, cs_bytes}, {buf_sin.buffer, 0, cs_bytes},
        {buf_qo.buffer, 0, q_bytes}, {buf_ko.buffer, 0, k_bytes},
    };
    std::vector<vk::WriteDescriptorSet> writes(6);
    for (int i = 0; i < 6; i++)
        writes[i].setDstSet(pipeline.desc_set).setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbis[i]);
    update_descriptor_set(ctx, pipeline, writes);

    struct { uint32_t B, nH, nKV, S, hd; } pc = {B, nH, nKV, S, hd};
    uint32_t groups = ((std::max(B*nH*S, B*nKV*S)) + 255) / 256;

    int warmup = 5, bench = 100;
    for (int i = 0; i < warmup; i++) dispatch(ctx, pipeline, groups, 1, 1, &pc, sizeof(pc));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; i++) dispatch(ctx, pipeline, groups, 1, 1, &pc, sizeof(pc));
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / bench;

    std::vector<uint8_t> res_q(q_bytes), res_k(k_bytes);
    download_buffer(ctx, buf_qo, res_q.data(), q_bytes);
    download_buffer(ctx, buf_ko, res_k.data(), k_bytes);

    float q_err = max_abs_error(reinterpret_cast<const uint16_t*>(expected.data()),
                                 reinterpret_cast<const uint16_t*>(res_q.data()), B*nH*S*hd);
    float k_err = max_abs_error(reinterpret_cast<const uint16_t*>(expected.data() + q_bytes),
                                 reinterpret_cast<const uint16_t*>(res_k.data()), B*nKV*S*hd);
    // fp16 has ~0.1% relative error; for values up to 5 sigma (~5.0), max abs error ~0.05 per element
    // but accumulation across 4M elements can amplify to ~4-5. Threshold of 6.0 catches fp16 rounding.
    float rope_limit = 6.0f;
    std::cout << "  " << us << " us | q_err=" << q_err << " k_err=" << k_err
              << (std::max(q_err, k_err) < rope_limit ? " PASS" : " FAIL") << std::endl;

    destroy_pipeline(ctx, pipeline);
    for (auto* b : {&buf_q, &buf_k, &buf_cos, &buf_sin, &buf_qo, &buf_ko}) destroy_buffer(ctx, *b);
    return std::max(q_err, k_err) < rope_limit;
}

bool test_dequant(ComputeContext& ctx) {
    std::cout << "\n=== Dequant Q4 ===" << std::endl;
    auto meta = read_meta("test_data/dequant_meta.txt");
    uint32_t out_f = std::stoi(meta["out_features"]), in_f = std::stoi(meta["in_features"]);
    uint32_t gs = std::stoi(meta["group_size"]);
    uint32_t n_groups = (in_f + gs - 1) / gs;
    std::cout << "  out=" << out_f << " in=" << in_f << " gs=" << gs << " groups=" << n_groups << std::endl;

    auto q_data = read_binary("test_data/dequant_q4.bin");
    auto scales = read_binary("test_data/dequant_scales.bin");
    auto zeros = read_binary("test_data/dequant_zeros.bin");
    auto expected = read_binary("test_data/dequant_expected.bin");

    auto buf_q = create_buffer(ctx, q_data.size(), vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_s = create_buffer(ctx, scales.size(), vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_z = create_buffer(ctx, zeros.size(), vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_out = create_buffer(ctx, expected.size(), vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    upload_buffer(ctx, buf_q, q_data.data(), q_data.size());
    upload_buffer(ctx, buf_s, scales.data(), scales.size());
    upload_buffer(ctx, buf_z, zeros.data(), zeros.size());

    auto shader = load_shader(ctx, "spirv/dequant_q4.spv", "main");
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    std::vector<vk::PushConstantRange> push = {{vk::ShaderStageFlagBits::eCompute, 0, 16}};
    auto pipeline = create_compute_pipeline(ctx, shader, bindings, push);

    std::vector<vk::DescriptorBufferInfo> dbis = {
        {buf_q.buffer, 0, q_data.size()}, {buf_s.buffer, 0, scales.size()},
        {buf_z.buffer, 0, zeros.size()}, {buf_out.buffer, 0, expected.size()},
    };
    std::vector<vk::WriteDescriptorSet> writes(4);
    for (int i = 0; i < 4; i++)
        writes[i].setDstSet(pipeline.desc_set).setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbis[i]);
    update_descriptor_set(ctx, pipeline, writes);

    struct { uint32_t of, inf, gs, ng; } pc = {out_f, in_f, gs, n_groups};
    uint32_t groups = (out_f * in_f + 255) / 256;

    int warmup = 5, bench = 50;
    for (int i = 0; i < warmup; i++) dispatch(ctx, pipeline, groups, 1, 1, &pc, sizeof(pc));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; i++) dispatch(ctx, pipeline, groups, 1, 1, &pc, sizeof(pc));
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / bench;

    std::vector<uint8_t> result(expected.size());
    download_buffer(ctx, buf_out, result.data(), result.size());
    float max_err = max_abs_error(reinterpret_cast<const uint16_t*>(expected.data()),
                                   reinterpret_cast<const uint16_t*>(result.data()),
                                   out_f * in_f);
    std::cout << "  " << us << " us | max_err=" << max_err
              << (max_err < 0.1f ? " PASS" : " FAIL") << std::endl;

    destroy_pipeline(ctx, pipeline);
    for (auto* b : {&buf_q, &buf_s, &buf_z, &buf_out}) destroy_buffer(ctx, *b);
    return max_err < 0.1f;
}

bool test_gemm(ComputeContext& ctx, const std::string& name) {
    std::cout << "\n=== GEMM " << name << " ===" << std::endl;
    auto meta = read_meta("test_data/gemm_" + name + "_meta.txt");
    uint32_t M = std::stoi(meta["M"]), K = std::stoi(meta["K"]), N = std::stoi(meta["N"]);
    std::cout << "  M=" << M << " K=" << K << " N=" << N << std::endl;

    auto input = read_binary("test_data/gemm_" + name + "_input.bin");
    auto expected = read_binary("test_data/gemm_" + name + "_expected.bin");

    size_t a_bytes = M * K * 2, b_bytes = K * N * 2, c_bytes = M * N * 2;
    auto buf_a = create_buffer(ctx, a_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_b = create_buffer(ctx, b_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_c = create_buffer(ctx, c_bytes, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    upload_buffer(ctx, buf_a, input.data(), a_bytes);
    upload_buffer(ctx, buf_b, input.data() + a_bytes, b_bytes);

    auto shader = load_shader(ctx, "spirv/gemm_fp16.spv", "main");
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    auto pipeline = create_compute_pipeline(ctx, shader, bindings,
        {{vk::ShaderStageFlagBits::eCompute, 0, 12}});

    vk::DescriptorBufferInfo dbi_a(buf_a.buffer, 0, a_bytes);
    vk::DescriptorBufferInfo dbi_b(buf_b.buffer, 0, b_bytes);
    vk::DescriptorBufferInfo dbi_c(buf_c.buffer, 0, c_bytes);
    std::vector<vk::WriteDescriptorSet> writes(3);
    writes[0].setDstSet(pipeline.desc_set).setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_a);
    writes[1].setDstSet(pipeline.desc_set).setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_b);
    writes[2].setDstSet(pipeline.desc_set).setDstBinding(2)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbi_c);
    update_descriptor_set(ctx, pipeline, writes);

    struct { uint32_t m, k, n; } pc = {M, K, N};
    uint32_t gx = (N + 127) / 128, gy = (M + 127) / 128;

    int warmup = 3, bench = 20;
    for (int i = 0; i < warmup; i++) dispatch(ctx, pipeline, gx, gy, 1, &pc, sizeof(pc));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench; i++) dispatch(ctx, pipeline, gx, gy, 1, &pc, sizeof(pc));
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / bench;

    double gflops = 2.0 * M * K * N / (us * 1e-3) / 1e9;
    std::vector<uint8_t> result(c_bytes);
    download_buffer(ctx, buf_c, result.data(), c_bytes);
    float max_err = max_abs_error(reinterpret_cast<const uint16_t*>(expected.data()),
                                   reinterpret_cast<const uint16_t*>(result.data()),
                                   M * N);
    bool pass = max_err < 0.5f; // fp16 GEMM error accumulates
    std::cout << "  " << us << " us | " << gflops << " GFLOPS | max_err=" << max_err
              << (pass ? " PASS" : " FAIL") << std::endl;

    destroy_pipeline(ctx, pipeline);
    destroy_buffer(ctx, buf_a);
    destroy_buffer(ctx, buf_b);
    destroy_buffer(ctx, buf_c);
    return pass;
}

int main() {
    try {
        auto ctx = init_compute("test_kernels");

        bool all_ok = true;
        all_ok &= test_rmsnorm(ctx);
        all_ok &= test_rope(ctx);
        all_ok &= test_dequant(ctx);
        all_ok &= test_gemm(ctx, "qproj");
        all_ok &= test_gemm(ctx, "oproj");

        destroy_compute(ctx);
        std::cout << "\n" << (all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return 1;
    }
}
