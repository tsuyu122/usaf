#include "vulkan_core.hpp"
#include <algorithm>
#include <set>
#include <cassert>
#include <cstring>

namespace usaf::vkcore {

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* user_data)
{
    (void)type; (void)user_data;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[VK] " << data->pMessage << std::endl;
    }
    return VK_FALSE;
}

ComputeContext init_compute(const std::string& app_name) {
    ComputeContext ctx;

    vk::ApplicationInfo app_info(app_name.c_str(), 1, "USAF", 1, VK_API_VERSION_1_3);
    
    bool enable_validation = false;
    const char* val_env = getenv("VK_VALIDATION");
    if (val_env && strcmp(val_env, "1") == 0) enable_validation = true;
    
    uint32_t layer_count = enable_validation ? 1 : 0;
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    uint32_t ext_count = enable_validation ? 1 : 0;
    const char* inst_exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

    vk::DebugUtilsMessengerCreateInfoEXT debug_ci;
    if (enable_validation) {
        debug_ci = vk::DebugUtilsMessengerCreateInfoEXT(
            {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
            debug_callback);
    }

    vk::InstanceCreateInfo instance_ci({}, &app_info, layer_count, layers, ext_count, inst_exts,
                                       enable_validation ? &debug_ci : nullptr);
    ctx.instance = vk::createInstance(instance_ci);

    auto phys_devices = ctx.instance.enumeratePhysicalDevices();
    if (phys_devices.empty()) throw std::runtime_error("No Vulkan devices");

    for (auto& pd : phys_devices) {
        auto props = pd.getProperties();
        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            ctx.physical_device = pd;
            ctx.props = props;
            break;
        }
    }
    if (!ctx.physical_device) {
        ctx.physical_device = phys_devices[0];
        ctx.props = ctx.physical_device.getProperties();
    }

    std::cout << "[VK] Device: " << ctx.props.deviceName << " | Vulkan " 
              << VK_API_VERSION_MAJOR(ctx.props.apiVersion) << "."
              << VK_API_VERSION_MINOR(ctx.props.apiVersion) << "."
              << VK_API_VERSION_PATCH(ctx.props.apiVersion) << std::endl;

    ctx.mem_props = ctx.physical_device.getMemoryProperties();

    auto qprops = ctx.physical_device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < qprops.size(); ++i) {
        if (qprops[i].queueFlags & vk::QueueFlagBits::eCompute) {
            ctx.compute_family = i;
            break;
        }
    }

    float qp = 1.0f;
    vk::DeviceQueueCreateInfo dq_ci({}, ctx.compute_family, 1, &qp);

    std::vector<const char*> dev_exts = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME
    };

    auto check_ext = [&](const char* name) -> bool {
        auto avail = ctx.physical_device.enumerateDeviceExtensionProperties();
        for (auto& e : avail) {
            if (strcmp(e.extensionName, name) == 0) return true;
        }
        return false;
    };

    std::vector<const char*> enabled_dev_exts;
    for (auto* ext : dev_exts) {
        if (check_ext(ext)) {
            enabled_dev_exts.push_back(ext);
            if (strcmp(ext, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0) {
                ctx.supports_fp16 = true;
                ctx.supports_int8 = true;
            }
        }
    }
    std::cout << "[VK] fp16: " << (ctx.supports_fp16 ? "yes" : "no")
              << " | int8: " << (ctx.supports_int8 ? "yes" : "no") << std::endl;

    // Enable Vulkan 1.1/1.2 features for 16-bit and 8-bit storage buffer access
    vk::PhysicalDeviceVulkan11Features vk11_feats{};
    vk11_feats.storageBuffer16BitAccess = VK_TRUE;

    vk::PhysicalDeviceVulkan12Features vk12_feats{};
    vk12_feats.shaderFloat16 = ctx.supports_fp16;
    vk12_feats.shaderInt8 = ctx.supports_int8;
    vk12_feats.storageBuffer8BitAccess = VK_TRUE;
    vk11_feats.pNext = &vk12_feats;

    vk::PhysicalDeviceFeatures2 feats2{};
    feats2.pNext = &vk11_feats;

    vk::DeviceCreateInfo dev_ci({}, dq_ci, {}, enabled_dev_exts, nullptr, &feats2);
    ctx.device = ctx.physical_device.createDevice(dev_ci);
    ctx.compute_queue = ctx.device.getQueue(ctx.compute_family, 0);

    vk::CommandPoolCreateInfo cp_ci(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, ctx.compute_family);
    ctx.cmd_pool = ctx.device.createCommandPool(cp_ci);

    // Descriptor pool with support for many storage buffer sets
    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        {vk::DescriptorType::eStorageBuffer, 4096},
    };
    vk::DescriptorPoolCreateInfo dp_ci(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1024, pool_sizes);
    ctx.desc_pool = ctx.device.createDescriptorPool(dp_ci);

    auto subgroup_props = ctx.physical_device.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceSubgroupProperties>();
    ctx.subgroup_size = subgroup_props.get<vk::PhysicalDeviceSubgroupProperties>().subgroupSize;
    std::cout << "[VK] subgroup_size: " << ctx.subgroup_size << std::endl;

    auto comp_props = ctx.physical_device.getProperties().limits;
    ctx.max_compute_shared_memory_size = comp_props.maxComputeSharedMemorySize;
    ctx.max_compute_work_group_size[0] = comp_props.maxComputeWorkGroupSize[0];
    ctx.max_compute_work_group_size[1] = comp_props.maxComputeWorkGroupSize[1];
    ctx.max_compute_work_group_size[2] = comp_props.maxComputeWorkGroupSize[2];
    ctx.max_compute_work_group_count[0] = comp_props.maxComputeWorkGroupCount[0];
    ctx.max_compute_work_group_count[1] = comp_props.maxComputeWorkGroupCount[1];
    ctx.max_compute_work_group_count[2] = comp_props.maxComputeWorkGroupCount[2];
    ctx.max_compute_work_group_invocations = comp_props.maxComputeWorkGroupInvocations;

    return ctx;
}

void destroy_compute(ComputeContext& ctx) {
    ctx.device.waitIdle();
    ctx.device.destroyCommandPool(ctx.cmd_pool);
    ctx.device.destroyDescriptorPool(ctx.desc_pool);
    ctx.device.destroy();
    ctx.instance.destroy();
}

uint32_t find_memory_type(ComputeContext& ctx, uint32_t type_bits, vk::MemoryPropertyFlags props) {
    for (uint32_t i = 0; i < ctx.mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1 << i)) && (ctx.mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("No suitable memory type");
}

Buffer create_buffer(ComputeContext& ctx, vk::DeviceSize size, vk::BufferUsageFlags usage,
                     vk::MemoryPropertyFlags mem_flags, bool persistently_mapped) {
    Buffer buf;
    buf.size = size;

    vk::BufferCreateInfo bci({}, size, usage);
    buf.buffer = ctx.device.createBuffer(bci);

    auto reqs = ctx.device.getBufferMemoryRequirements(buf.buffer);
    uint32_t mem_type = find_memory_type(ctx, reqs.memoryTypeBits, mem_flags);

    vk::MemoryAllocateInfo mai(reqs.size, mem_type);
    buf.memory = ctx.device.allocateMemory(mai);
    ctx.device.bindBufferMemory(buf.buffer, buf.memory, 0);

    if (mem_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        buf.mapped = ctx.device.mapMemory(buf.memory, 0, size);
        buf.is_coherent = (mem_flags & vk::MemoryPropertyFlagBits::eHostCoherent) != vk::MemoryPropertyFlags{};
    }

    return buf;
}

void upload_buffer(ComputeContext& ctx, Buffer& buf, const void* data, vk::DeviceSize size) {
    assert(buf.mapped && "Buffer must be host-visible");
    if (!buf.is_coherent) {
        vk::MappedMemoryRange range(buf.memory, 0, size);
        ctx.device.invalidateMappedMemoryRanges(range);
    }
    std::memcpy(buf.mapped, data, size);
    if (!buf.is_coherent) {
        vk::MappedMemoryRange range(buf.memory, 0, size);
        ctx.device.flushMappedMemoryRanges(range);
    }
}

void download_buffer(ComputeContext& ctx, Buffer& buf, void* data, vk::DeviceSize size) {
    if (!buf.is_coherent) {
        vk::MappedMemoryRange mmr(buf.memory, 0, size);
        ctx.device.invalidateMappedMemoryRanges(mmr);
    }
    if (size > buf.size) size = buf.size;
    memcpy(data, buf.mapped, size);
}

void destroy_buffer(ComputeContext& ctx, Buffer& buf) {
    if (buf.mapped) ctx.device.unmapMemory(buf.memory);
    ctx.device.destroyBuffer(buf.buffer);
    ctx.device.freeMemory(buf.memory);
    buf.mapped = nullptr;
}

std::vector<uint32_t> read_spirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open SPIR-V: " + path);
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint32_t> data(size / 4);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

ShaderModule load_shader(ComputeContext& ctx, const std::string& spirv_path,
                         const std::string& entry,
                         const std::vector<vk::SpecializationMapEntry>& spec_entries,
                         const void* spec_data_ptr, size_t spec_data_size) {
    ShaderModule sm;
    auto code = read_spirv(spirv_path);
    vk::ShaderModuleCreateInfo smci({}, code);
    sm.module = ctx.device.createShaderModule(smci);
    sm.entry_point = entry;
    sm.spec_entries = spec_entries;
    if (spec_data_ptr && spec_data_size) {
        sm.spec_data.resize(spec_data_size);
        std::memcpy(sm.spec_data.data(), spec_data_ptr, spec_data_size);
        sm.spec_info = vk::SpecializationInfo(
            (uint32_t)spec_entries.size(), spec_entries.data(),
            spec_data_size, sm.spec_data.data());
    }
    return sm;
}

ComputePipeline create_compute_pipeline(
    ComputeContext& ctx, const ShaderModule& shader,
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
    const std::vector<vk::PushConstantRange>& push_constants) {

    ComputePipeline cp;

    vk::DescriptorSetLayoutCreateInfo dslci({}, bindings);
    cp.desc_set_layout = ctx.device.createDescriptorSetLayout(dslci);

    vk::PipelineLayoutCreateInfo plci({}, cp.desc_set_layout, push_constants);
    cp.layout = ctx.device.createPipelineLayout(plci);

    vk::ComputePipelineCreateInfo cpci({},
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
            shader.module, shader.entry_point.c_str(),
            shader.spec_data.empty() ? nullptr : &shader.spec_info),
        cp.layout);
    cp.pipeline = ctx.device.createComputePipeline({}, cpci).value;
    cp.shader_module = shader.module;

    vk::DescriptorSetAllocateInfo dsai(ctx.desc_pool, cp.desc_set_layout);
    cp.desc_set = ctx.device.allocateDescriptorSets(dsai)[0];

    return cp;
}

void destroy_pipeline(ComputeContext& ctx, ComputePipeline& pipeline) {
    ctx.device.destroyPipeline(pipeline.pipeline);
    ctx.device.destroyPipelineLayout(pipeline.layout);
    ctx.device.destroyDescriptorSetLayout(pipeline.desc_set_layout);
    ctx.device.destroyShaderModule(pipeline.shader_module);
}

void update_descriptor_set(ComputeContext& ctx, ComputePipeline& pipeline,
                           const std::vector<vk::WriteDescriptorSet>& writes) {
    ctx.device.updateDescriptorSets(writes, {});
}

void dispatch(ComputeContext& ctx, ComputePipeline& pipeline,
              uint32_t gx, uint32_t gy, uint32_t gz,
              const void* push_data, uint32_t push_size) {
    vk::CommandBufferAllocateInfo cbai(ctx.cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
    auto cmd_buf = ctx.device.allocateCommandBuffers(cbai)[0];

    vk::CommandBufferBeginInfo bbi(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd_buf.begin(bbi);
    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.layout, 0, pipeline.desc_set, {});
    if (push_data && push_size) {
        cmd_buf.pushConstants(pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, push_size, push_data);
    }
    cmd_buf.dispatch(gx, gy, gz);
    cmd_buf.end();

    vk::SubmitInfo si({}, {}, cmd_buf);
    vk::Fence fence = ctx.device.createFence({});
    ctx.compute_queue.submit(si, fence);
    (void)ctx.device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    ctx.device.destroyFence(fence);
    ctx.device.freeCommandBuffers(ctx.cmd_pool, cmd_buf);
}

void wait_idle(ComputeContext& ctx) {
    ctx.device.waitIdle();
}

} // namespace usaf::vkcore
