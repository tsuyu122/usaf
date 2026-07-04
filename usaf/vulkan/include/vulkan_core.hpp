#pragma once
#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <iostream>

namespace usaf::vkcore {

struct ComputeContext {
    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::Queue compute_queue;
    uint32_t compute_family = 0;
    vk::CommandPool cmd_pool;
    vk::DescriptorPool desc_pool;
    vk::PhysicalDeviceProperties props;
    vk::PhysicalDeviceMemoryProperties mem_props;
    bool supports_fp16 = false;
    bool supports_int8 = false;
    uint32_t subgroup_size = 32;
    uint32_t max_compute_shared_memory_size = 0;
    uint32_t max_compute_work_group_size[3] = {0,0,0};
    uint32_t max_compute_work_group_count[3] = {0,0,0};
    uint32_t max_compute_work_group_invocations = 0;
};

struct Buffer {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::DeviceSize size;
    void* mapped = nullptr;
    bool is_coherent = false;
};

struct ShaderModule {
    vk::ShaderModule module;
    std::string entry_point;
    vk::SpecializationInfo spec_info;
    std::vector<vk::SpecializationMapEntry> spec_entries;
    std::vector<uint8_t> spec_data;
};

struct ComputePipeline {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    vk::DescriptorSetLayout desc_set_layout;
    vk::DescriptorSet desc_set;
    vk::ShaderModule shader_module;
};


ComputeContext init_compute(const std::string& app_name = "USAF Vulkan");
void destroy_compute(ComputeContext& ctx);

Buffer create_buffer(ComputeContext& ctx, vk::DeviceSize size,
                     vk::BufferUsageFlags usage,
                     vk::MemoryPropertyFlags mem_flags,
                     bool persistently_mapped = false);
void upload_buffer(ComputeContext& ctx, Buffer& buf, const void* data, vk::DeviceSize size);
void download_buffer(ComputeContext& ctx, Buffer& buf, void* data, vk::DeviceSize size);
void destroy_buffer(ComputeContext& ctx, Buffer& buf);

ShaderModule load_shader(ComputeContext& ctx, const std::string& spirv_path,
                         const std::string& entry = "main",
                         const std::vector<vk::SpecializationMapEntry>& spec_entries = {},
                         const void* spec_data_ptr = nullptr, size_t spec_data_size = 0);

std::vector<uint32_t> read_spirv(const std::string& path);

ComputePipeline create_compute_pipeline(
    ComputeContext& ctx,
    const ShaderModule& shader,
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
    const std::vector<vk::PushConstantRange>& push_constants = {});

void destroy_pipeline(ComputeContext& ctx, ComputePipeline& pipeline);

void update_descriptor_set(ComputeContext& ctx, ComputePipeline& pipeline,
                           const std::vector<vk::WriteDescriptorSet>& writes);

void dispatch(ComputeContext& ctx, ComputePipeline& pipeline,
              uint32_t gx, uint32_t gy = 1, uint32_t gz = 1,
              const void* push_data = nullptr, uint32_t push_size = 0);

void wait_idle(ComputeContext& ctx);

uint32_t find_memory_type(ComputeContext& ctx, uint32_t type_bits,
                          vk::MemoryPropertyFlags props);

} // namespace usaf::vkcore
