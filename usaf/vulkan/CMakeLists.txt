cmake_minimum_required(VERSION 3.20)
project(usaf_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "C:/Users/hm/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

set(VULKAN_SDK_INCLUDE "${CMAKE_SOURCE_DIR}/include")
set(SPIRV_DIR "${CMAKE_BINARY_DIR}/spirv")
file(MAKE_DIRECTORY ${SPIRV_DIR})

# ── SPIR-V compilation ──
set(KERNEL_SRC
    src/kernels/test_add.comp
    src/kernels/rmsnorm_fp16.comp
    src/kernels/rmsnorm_simple.comp
    src/kernels/gemm_fp16.comp
    src/kernels/rope_fp16.comp
    src/kernels/dequant_q4.comp
)

set(SPIRV_DEPS "")
foreach(KERNEL ${KERNEL_SRC})
    get_filename_component(KERNEL_NAME ${KERNEL} NAME_WE)
    set(SPIRV_OUT "${SPIRV_DIR}/${KERNEL_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV_OUT}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3
            -O -o ${SPIRV_OUT} ${CMAKE_SOURCE_DIR}/${KERNEL}
        DEPENDS ${CMAKE_SOURCE_DIR}/${KERNEL}
        COMMENT "Compiling ${KERNEL_NAME}.comp -> SPIR-V"
    )
    add_custom_target(spirv_${KERNEL_NAME} DEPENDS ${SPIRV_OUT})
    list(APPEND SPIRV_DEPS spirv_${KERNEL_NAME})
endforeach()

# ── Vulkan core library ──
add_library(usaf_vulkan_core STATIC
    src/vulkan_core.cpp
)
target_include_directories(usaf_vulkan_core PUBLIC
    ${VULKAN_SDK_INCLUDE}
    ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(usaf_vulkan_core PUBLIC Vulkan::Vulkan)

# ── Test binaries ──
add_executable(test_vulkan src/test_vulkan.cpp)
target_include_directories(test_vulkan PRIVATE ${VULKAN_SDK_INCLUDE} ${Vulkan_INCLUDE_DIRS})
target_link_libraries(test_vulkan PRIVATE usaf_vulkan_core Vulkan::Vulkan)
add_dependencies(test_vulkan spirv_test_add)

add_executable(test_kernels src/test_kernels.cpp)
target_include_directories(test_kernels PRIVATE ${VULKAN_SDK_INCLUDE} ${Vulkan_INCLUDE_DIRS})
target_link_libraries(test_kernels PRIVATE usaf_vulkan_core Vulkan::Vulkan)
add_dependencies(test_kernels ${SPIRV_DEPS})

# Copy SPIR-V to build dir
add_custom_target(copy_spirv ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SPIRV_DIR} ${CMAKE_BINARY_DIR}/spirv
    DEPENDS ${SPIRV_DEPS}
)
add_dependencies(test_vulkan copy_spirv)
add_dependencies(test_kernels copy_spirv)

add_executable(rmsnorm_debug src/rmsnorm_debug.cpp)
target_include_directories(rmsnorm_debug PRIVATE ${VULKAN_SDK_INCLUDE} ${Vulkan_INCLUDE_DIRS})
target_link_libraries(rmsnorm_debug PRIVATE usaf_vulkan_core Vulkan::Vulkan)
add_dependencies(rmsnorm_debug copy_spirv)

add_executable(rope_debug src/rope_debug_gen.cpp)
target_include_directories(rope_debug PRIVATE ${VULKAN_SDK_INCLUDE} ${Vulkan_INCLUDE_DIRS})
target_link_libraries(rope_debug PRIVATE usaf_vulkan_core Vulkan::Vulkan)
add_dependencies(rope_debug copy_spirv)

# ── Python module (pybind11) ──
pybind11_add_module(usaf_vk src/pybind_module.cpp)
target_include_directories(usaf_vk PRIVATE ${VULKAN_SDK_INCLUDE} ${Vulkan_INCLUDE_DIRS})
target_link_libraries(usaf_vk PRIVATE usaf_vulkan_core Vulkan::Vulkan)
add_dependencies(usaf_vk copy_spirv)
