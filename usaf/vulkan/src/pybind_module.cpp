#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vulkan_core.hpp"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <vector>
#include <cstring>

namespace py = pybind11;
using namespace usaf::vkcore;

// Global Vulkan context with lazy init + proper cleanup
static ComputeContext* g_ctx = nullptr;
static std::string g_spirv_dir;
static std::once_flag g_init_flag;

static void ensure_init() {
    std::call_once(g_init_flag, []() {
        g_ctx = new ComputeContext();
        *g_ctx = init_compute("usaf_pybind");
    });
}

static ComputeContext& ctx() { return *g_ctx; }

// Shader + pipeline cache to avoid reloading SPIR-V per call
struct CachedPipeline {
    ComputePipeline pipeline;
    vk::DescriptorSetLayoutBinding layout_bindings[6];
    uint32_t num_bindings = 0;
    uint32_t push_size = 0;
    bool owned = true;
};

static std::unordered_map<std::string, std::unique_ptr<CachedPipeline>> g_pipeline_cache;
static std::mutex g_cache_mutex;

static CachedPipeline* get_or_create_pipeline(
    const std::string& name,
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
    uint32_t push_size)
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto it = g_pipeline_cache.find(name);
    if (it != g_pipeline_cache.end()) {
        return it->second.get();
    }
    auto cp = std::make_unique<CachedPipeline>();
    cp->num_bindings = std::min((uint32_t)bindings.size(), 6u);
    for (uint32_t i = 0; i < cp->num_bindings; i++) {
        cp->layout_bindings[i] = bindings[i];
    }
    cp->push_size = push_size;
    auto* ptr = cp.get();
    g_pipeline_cache[name] = std::move(cp);
    return ptr;
}

// ── Fase 8: Persistent Buffer Management ──
static std::atomic<int> g_next_handle{1};
static std::mutex g_buf_mutex;
static std::unordered_map<int, std::unique_ptr<Buffer>> g_buffers;

static int alloc_handle() { return g_next_handle.fetch_add(1); }

static Buffer& get_buf(int h) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    auto it = g_buffers.find(h);
    if (it == g_buffers.end()) throw std::runtime_error("Invalid buffer handle: " + std::to_string(h));
    return *it->second;
}

static int create_device_buf(size_t nbytes, bool host_visible) {
    ensure_init();
    vk::MemoryPropertyFlags mem_flags;
    if (host_visible) {
        mem_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    } else {
        mem_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    }
    auto buf = std::make_unique<Buffer>(create_buffer(ctx(), nbytes,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eTransferDst,
        mem_flags));
    int h = alloc_handle();
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    g_buffers[h] = std::move(buf);
    return h;
}

static void upload_to_buf(int h, py::array data) {
    auto& buf = get_buf(h);
    auto req = data.request();
    size_t sz = req.size * req.itemsize;
    if (sz > buf.size) throw std::runtime_error("Upload exceeds buffer size");
    upload_buffer(ctx(), buf, req.ptr, sz);
}

static py::array_t<uint16_t> download_from_buf(int h, const std::vector<py::ssize_t>& shape) {
    auto& buf = get_buf(h);
    auto result = py::array_t<uint16_t>(shape);
    download_buffer(ctx(), buf, result.request().ptr, buf.size);
    return result;
}

static void destroy_device_buf(int h) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    auto it = g_buffers.find(h);
    if (it != g_buffers.end()) {
        destroy_buffer(ctx(), *it->second);
        g_buffers.erase(it);
    }
}

// ── Fase 8: Barrier between dispatches ──
static void memory_barrier() {
    ensure_init();
    // Memory barrier: make writes visible to subsequent reads.
    // In Vulkan, barriers are inserted via pipeline barriers between dispatches.
    // For simplicity, we do a full device wait — ensures all prior dispatches complete.
    wait_idle(ctx());
}

// ── Fase 8: Pipelined kernel dispatches (buffer handles, no round-trips) ──

static void rmsnorm_pipelined(int x_h, int w_h, int out_h, int rows, int cols, float eps) {
    ensure_init();
    auto& bx = get_buf(x_h); auto& bw = get_buf(w_h); auto& bo = get_buf(out_h);
    auto* cp = get_or_create_pipeline("rmsnorm_fp16",
        {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
        sizeof(uint32_t)*2 + sizeof(float));
    if (!cp->pipeline.pipeline) {
        auto shader = load_shader(ctx(), g_spirv_dir.empty() ? "spirv/rmsnorm_fp16.spv" : (g_spirv_dir + "/rmsnorm_fp16.spv").c_str(), "main");
        std::vector<vk::PushConstantRange> push = {{vk::ShaderStageFlagBits::eCompute, 0, (uint32_t)(sizeof(uint32_t)*2 + sizeof(float))}};
        cp->pipeline = create_compute_pipeline(ctx(), shader, {cp->layout_bindings[0], cp->layout_bindings[1], cp->layout_bindings[2]}, push);
    }
    vk::DescriptorBufferInfo dbis[3] = {{bx.buffer, 0, bx.size}, {bw.buffer, 0, bw.size}, {bo.buffer, 0, bo.size}};
    std::vector<vk::WriteDescriptorSet> writes(3);
    for (int i = 0; i < 3; i++) writes[i].setDstSet(cp->pipeline.desc_set).setDstBinding(i)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbis[i]);
    update_descriptor_set(ctx(), cp->pipeline, writes);
    struct { uint32_t r, c; float e; } pc = {(uint32_t)rows, (uint32_t)cols, eps};
    dispatch(ctx(), cp->pipeline, rows, 1, 1, &pc, sizeof(pc));
}

static void gemm_pipelined(int a_h, int b_h, int c_h, int M, int K, int N) {
    ensure_init();
    auto& ba = get_buf(a_h); auto& bb = get_buf(b_h); auto& bc = get_buf(c_h);
    auto* cp = get_or_create_pipeline("gemm_fp16",
        {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
        sizeof(uint32_t)*3);
    if (!cp->pipeline.pipeline) {
        auto shader = load_shader(ctx(), g_spirv_dir.empty() ? "spirv/gemm_fp16.spv" : (g_spirv_dir + "/gemm_fp16.spv").c_str(), "main");
        std::vector<vk::PushConstantRange> push = {{vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)*3}};
        cp->pipeline = create_compute_pipeline(ctx(), shader, {cp->layout_bindings[0], cp->layout_bindings[1], cp->layout_bindings[2]}, push);
    }
    vk::DescriptorBufferInfo dbis[3] = {{ba.buffer, 0, ba.size}, {bb.buffer, 0, bb.size}, {bc.buffer, 0, bc.size}};
    std::vector<vk::WriteDescriptorSet> writes(3);
    for (int i = 0; i < 3; i++) writes[i].setDstSet(cp->pipeline.desc_set).setDstBinding(i)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbis[i]);
    update_descriptor_set(ctx(), cp->pipeline, writes);
    struct { uint32_t m, k, n; } pc = {(uint32_t)M, (uint32_t)K, (uint32_t)N};
    uint32_t gx = (N + 127) / 128, gy = (M + 127) / 128;
    dispatch(ctx(), cp->pipeline, gx, gy, 1, &pc, sizeof(pc));
}

static void dequant_pipelined(int q_h, int s_h, int z_h, int out_h, 
                               int out_rows, int in_features, int group_size) {
    ensure_init();
    auto& bq = get_buf(q_h); auto& bs = get_buf(s_h); auto& bz = get_buf(z_h); auto& bo = get_buf(out_h);
    int n_groups = (in_features + group_size - 1) / group_size;
    auto* cp = get_or_create_pipeline("dequant_q4_v2",  // v2: 4 push constants (was 2)
        {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
        sizeof(uint32_t)*4);
    if (!cp->pipeline.pipeline) {
        auto shader = load_shader(ctx(), g_spirv_dir.empty() ? "spirv/dequant_q4.spv" : (g_spirv_dir + "/dequant_q4.spv").c_str(), "main");
        std::vector<vk::PushConstantRange> push = {{vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)*4}};
        cp->pipeline = create_compute_pipeline(ctx(), shader, {cp->layout_bindings[0], cp->layout_bindings[1], cp->layout_bindings[2], cp->layout_bindings[3]}, push);
    }
    vk::DescriptorBufferInfo dbis[4] = {{bq.buffer, 0, bq.size}, {bs.buffer, 0, bs.size}, {bz.buffer, 0, bz.size}, {bo.buffer, 0, bo.size}};
    std::vector<vk::WriteDescriptorSet> writes(4);
    for (int i = 0; i < 4; i++) writes[i].setDstSet(cp->pipeline.desc_set).setDstBinding(i)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbis[i]);
    update_descriptor_set(ctx(), cp->pipeline, writes);
    struct { uint32_t out_rows, in_cols, gs, ng; } pc = {(uint32_t)out_rows, (uint32_t)in_features, (uint32_t)group_size, (uint32_t)n_groups};
    uint32_t total = (uint32_t)out_rows * (uint32_t)in_features;
    uint32_t gx = (total + 255) / 256;
    dispatch(ctx(), cp->pipeline, gx, 1, 1, &pc, sizeof(pc));
}

static py::tuple rope_pipelined(int q_h, int k_h, int cos_h, int sin_h,
                                 int qo_h, int ko_h,
                                 int B, int nH, int nKV, int S, int hd) {
    ensure_init();
    auto& bq = get_buf(q_h); auto& bk = get_buf(k_h);
    auto& bc = get_buf(cos_h); auto& bs = get_buf(sin_h);
    auto& bqo = get_buf(qo_h); auto& bko = get_buf(ko_h);

    std::vector<vk::DescriptorSetLayoutBinding> bnd = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    struct { uint32_t B, nH, nKV, S, hd; } pc = {(uint32_t)B, (uint32_t)nH, (uint32_t)nKV, (uint32_t)S, (uint32_t)hd};

    auto* cp = get_or_create_pipeline("rope_fp16", bnd, sizeof(pc));
    if (!cp->pipeline.pipeline) {
        auto shader = load_shader(ctx(), g_spirv_dir.empty() ? "spirv/rope_fp16.spv" : (g_spirv_dir + "/rope_fp16.spv").c_str(), "main");
        std::vector<vk::PushConstantRange> push = {{vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc)}};
        cp->pipeline = create_compute_pipeline(ctx(), shader, bnd, push);
    }

    vk::DescriptorBufferInfo dbis[6] = {
        {bq.buffer, 0, bq.size}, {bk.buffer, 0, bk.size},
        {bc.buffer, 0, bc.size}, {bs.buffer, 0, bs.size},
        {bqo.buffer, 0, bqo.size}, {bko.buffer, 0, bko.size},
    };
    std::vector<vk::WriteDescriptorSet> writes(6);
    for (int i = 0; i < 6; i++)
        writes[i].setDstSet(cp->pipeline.desc_set).setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbis[i]);
    update_descriptor_set(ctx(), cp->pipeline, writes);

    uint32_t g = (std::max((uint32_t)(B * nH * S * hd), (uint32_t)(B * nKV * S * hd)) + 255) / 256;
    dispatch(ctx(), cp->pipeline, g, 1, 1, &pc, sizeof(pc));
    return py::make_tuple(0, 0);  // output is in qo_h, ko_h buffers
}

// Helper: create buffer from numpy array, bind descriptors, dispatch, read back
static py::array_t<uint16_t> run_kernel(
    const std::string& pipeline_name,
    std::vector<py::array_t<uint16_t>> inputs,
    std::vector<py::ssize_t> out_shape,
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
    const void* push_data, uint32_t push_size,
    uint32_t gx, uint32_t gy, uint32_t gz)
{
    ensure_init();
    auto* cp = get_or_create_pipeline(pipeline_name, bindings, push_size);

    std::vector<Buffer> vk_bufs;
    std::vector<vk::DescriptorBufferInfo> dbis;
    std::vector<vk::WriteDescriptorSet> writes;

    size_t total_output = 0;
    for (auto& inp : inputs) {
        auto req = inp.request();
        size_t sz = req.size * req.itemsize;
        auto buf = create_buffer(ctx(), sz,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        upload_buffer(ctx(), buf, req.ptr, sz);
        dbis.push_back({buf.buffer, 0, sz});
        vk_bufs.push_back(std::move(buf));
    }
    // Output buffer
    total_output = 1;
    for (auto s : out_shape) total_output *= (size_t)s;
    total_output *= sizeof(uint16_t);
    auto buf_out = create_buffer(ctx(), total_output,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    dbis.push_back({buf_out.buffer, 0, total_output});

    // Create pipeline if not yet created (needs the actual shader module)
    if (!cp->owned || !cp->pipeline.pipeline) {
    // Load shader (SPIR-V path derived from pipeline name)
    std::string spirv = g_spirv_dir + "/" + pipeline_name + ".spv";
    if (!g_spirv_dir.empty()) {
        spirv = g_spirv_dir + "/" + pipeline_name + ".spv";
    } else {
        spirv = std::string("spirv/") + pipeline_name + ".spv";
    }
    auto shader = load_shader(ctx(), spirv, "main");
        std::vector<vk::DescriptorSetLayoutBinding> bnd;
        for (uint32_t i = 0; i < cp->num_bindings; i++) bnd.push_back(cp->layout_bindings[i]);
        std::vector<vk::PushConstantRange> push = {};
        if (push_size > 0) push.push_back({vk::ShaderStageFlagBits::eCompute, 0, push_size});
        auto pip = create_compute_pipeline(ctx(), shader, bnd, push);
        if (cp->pipeline.pipeline) destroy_pipeline(ctx(), cp->pipeline);
        cp->pipeline = pip;
    }

    // Update descriptor set with current buffers
    writes.resize(dbis.size());
    for (size_t i = 0; i < dbis.size(); i++) {
        writes[i].setDstSet(cp->pipeline.desc_set).setDstBinding((uint32_t)i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbis[i]);
    }
    update_descriptor_set(ctx(), cp->pipeline, writes);

    dispatch(ctx(), cp->pipeline, gx, gy, gz, push_data, push_size);

    auto result = py::array_t<uint16_t>(out_shape);
    download_buffer(ctx(), buf_out, result.request().ptr, total_output);

    // Cleanup buffers (keep pipeline for reuse)
    for (auto& b : vk_bufs) destroy_buffer(ctx(), b);
    destroy_buffer(ctx(), buf_out);

    return result;
}

py::array_t<uint16_t> run_rmsnorm(
    py::array_t<uint16_t> x_np,
    py::array_t<uint16_t> w_np,
    int rows, int cols, float eps)
{
    struct { uint32_t r, c; float e; } pc = {(uint32_t)rows, (uint32_t)cols, eps};
    return run_kernel("rmsnorm_fp16",
        {x_np, w_np},
        {rows, cols},
        {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
        &pc, sizeof(pc), rows, 1, 1);
}

py::array_t<uint16_t> run_gemm(
    py::array_t<uint16_t> a_np,
    py::array_t<uint16_t> b_np,
    int M, int K, int N)
{
    struct { uint32_t m, k, n; } pc = {(uint32_t)M, (uint32_t)K, (uint32_t)N};
    uint32_t gx = (N + 127) / 128, gy = (M + 127) / 128;
    return run_kernel("gemm_fp16",
        {a_np, b_np},
        {M, N},
        {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
        &pc, sizeof(pc), gx, gy, 1);
}

py::tuple run_rope(
    py::array_t<uint16_t> q_np,
    py::array_t<uint16_t> k_np,
    py::array_t<uint16_t> cos_np,
    py::array_t<uint16_t> sin_np,
    int B, int nH, int nKV, int S, int hd)
{
    std::vector<vk::DescriptorSetLayoutBinding> bnd = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    struct { uint32_t B, nH, nKV, S, hd; } pc = {(uint32_t)B, (uint32_t)nH, (uint32_t)nKV, (uint32_t)S, (uint32_t)hd};
    // 1 thread por ELEMENTO (gid < B*nH*S*hd no shader) — precisa incluir hd
    uint32_t g = (std::max(B * nH * S * hd, B * nKV * S * hd) + 255) / 256;

    ensure_init();
    auto* cp = get_or_create_pipeline("rope_fp16", bnd, sizeof(pc));

    size_t q_bytes = B * nH * S * hd * 2;
    size_t k_bytes = B * nKV * S * hd * 2;
    size_t cs_bytes = S * hd * 2;

    // Input buffers
    auto buf_q = create_buffer(ctx(), q_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_k = create_buffer(ctx(), k_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_c = create_buffer(ctx(), cs_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_s = create_buffer(ctx(), cs_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_qo = create_buffer(ctx(), q_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto buf_ko = create_buffer(ctx(), k_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    upload_buffer(ctx(), buf_q, q_np.request().ptr, q_bytes);
    upload_buffer(ctx(), buf_k, k_np.request().ptr, k_bytes);
    upload_buffer(ctx(), buf_c, cos_np.request().ptr, cs_bytes);
    upload_buffer(ctx(), buf_s, sin_np.request().ptr, cs_bytes);

    // Create pipeline if needed
    if (!cp->owned || !cp->pipeline.pipeline) {
        auto shader = load_shader(ctx(), g_spirv_dir.empty() ? "spirv/rope_fp16.spv" : (g_spirv_dir + "/rope_fp16.spv").c_str(), "main");
        std::vector<vk::PushConstantRange> push = {{vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc)}};
        auto pip = create_compute_pipeline(ctx(), shader, bnd, push);
        if (cp->pipeline.pipeline) destroy_pipeline(ctx(), cp->pipeline);
        cp->pipeline = pip;
    }

    std::vector<vk::DescriptorBufferInfo> dbis = {
        {buf_q.buffer, 0, q_bytes}, {buf_k.buffer, 0, k_bytes},
        {buf_c.buffer, 0, cs_bytes}, {buf_s.buffer, 0, cs_bytes},
        {buf_qo.buffer, 0, q_bytes}, {buf_ko.buffer, 0, k_bytes},
    };
    std::vector<vk::WriteDescriptorSet> writes(6);
    for (int i = 0; i < 6; i++)
        writes[i].setDstSet(cp->pipeline.desc_set).setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(dbis[i]);
    update_descriptor_set(ctx(), cp->pipeline, writes);

    dispatch(ctx(), cp->pipeline, g, 1, 1, &pc, sizeof(pc));

    auto res_q = py::array_t<uint16_t>({B, nH, S, hd});
    auto res_k = py::array_t<uint16_t>({B, nKV, S, hd});
    download_buffer(ctx(), buf_qo, res_q.request().ptr, q_bytes);
    download_buffer(ctx(), buf_ko, res_k.request().ptr, k_bytes);

    for (auto* b : {&buf_q, &buf_k, &buf_c, &buf_s, &buf_qo, &buf_ko}) destroy_buffer(ctx(), *b);
    return py::make_tuple(res_q, res_k);
}

PYBIND11_MODULE(usaf_vk, m) {
    m.doc() = "USAF Vulkan compute kernels";

    m.def("set_spirv_path", [](const std::string& path) {
        g_spirv_dir = path;
    }, "Set the directory containing SPIR-V shader files");

    m.def("init", &ensure_init, "Initialize Vulkan context (idempotent)");

    // ── Legacy API (round-trip per op) ──
    m.def("rmsnorm", &run_rmsnorm, "RMSNorm: y = (x / rms(x)) * w",
          py::arg("x"), py::arg("w"), py::arg("rows"), py::arg("cols"), py::arg("eps") = 1e-6f);
    m.def("gemm", &run_gemm, "GEMM: C[M,N] = A[M,K] @ B[K,N]",
          py::arg("a"), py::arg("b"), py::arg("M"), py::arg("K"), py::arg("N"));
    m.def("rope", &run_rope, "RoPE: apply rotary position embedding to Q and K",
          py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
          py::arg("B"), py::arg("nH"), py::arg("nKV"), py::arg("S"), py::arg("hd"));

    // ── Fase 8: Persistent Buffer API (no round-trips) ──
    m.def("create_buf", &create_device_buf, "Create a device buffer (GPU-resident)",
          py::arg("nbytes"), py::arg("host_visible") = true);
    m.def("upload", &upload_to_buf, "Upload numpy data to a buffer",
          py::arg("handle"), py::arg("data"));
    m.def("download", &download_from_buf, "Download buffer to numpy array",
          py::arg("handle"), py::arg("shape"));
    m.def("destroy_buf", &destroy_device_buf, "Destroy a buffer",
          py::arg("handle"));
    m.def("barrier", &memory_barrier, "Insert a memory barrier between dispatches");

    // ── Fase 8: Pipelined kernels (dispatch to buffers, no upload/download) ──
    m.def("rmsnorm_pipe", &rmsnorm_pipelined, "RMSNorm pipelined: out = rmsnorm(x, w)",
          py::arg("x_handle"), py::arg("w_handle"), py::arg("out_handle"),
          py::arg("rows"), py::arg("cols"), py::arg("eps") = 1e-6f);
    m.def("gemm_pipe", &gemm_pipelined, "GEMM pipelined: C = A @ B",
          py::arg("a_handle"), py::arg("b_handle"), py::arg("c_handle"),
          py::arg("M"), py::arg("K"), py::arg("N"));
    m.def("dequant_pipe", &dequant_pipelined, "Dequant Q4 pipelined: out = dequant(q, s, z)",
          py::arg("q_handle"), py::arg("s_handle"), py::arg("z_handle"), py::arg("out_handle"),
          py::arg("out_rows"), py::arg("in_features"), py::arg("group_size") = 128);
    m.def("rope_pipe", &rope_pipelined, "RoPE pipelined: apply rotary position embedding to Q and K buffers",
          py::arg("q_handle"), py::arg("k_handle"), py::arg("cos_handle"), py::arg("sin_handle"),
          py::arg("qo_handle"), py::arg("ko_handle"),
          py::arg("B"), py::arg("nH"), py::arg("nKV"), py::arg("S"), py::arg("hd"));

    // Cleanup on module unload
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        // Destroy all persistent buffers
        for (auto& [h, buf] : g_buffers) {
            destroy_buffer(*g_ctx, *buf);
        }
        g_buffers.clear();
        if (g_ctx) {
            for (auto& [name, cp] : g_pipeline_cache) {
                if (cp->pipeline.pipeline) destroy_pipeline(*g_ctx, cp->pipeline);
            }
            g_pipeline_cache.clear();
            destroy_compute(*g_ctx);
            delete g_ctx;
            g_ctx = nullptr;
        }
    }));
}
