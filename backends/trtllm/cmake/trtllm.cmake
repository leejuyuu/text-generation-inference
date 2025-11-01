set(TRT_INCLUDE_DIR ${TGI_TRTLLM_BACKEND_TRT_INCLUDE_DIR})
set(TRT_LIB_DIR ${TGI_TRTLLM_BACKEND_TRT_LIB_DIR})

set(USE_CXX11_ABI ON)
set(BUILD_PYT OFF)
set(BUILD_PYBIND OFF)
set(BUILD_MICRO_BENCHMARKS OFF)
set(BUILD_BENCHMARKS OFF)
set(BUILD_TESTS OFF)

# TODO: BINDING_TYPE, DEEP_EP and DEEP_GEMM requires python torch
set(BINDING_TYPE "none")
set(BUILD_DEEP_EP OFF)
set(BUILD_DEEP_GEMM OFF)

set(CMAKE_CUDA_ARCHITECTURES ${TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST})

message(STATUS "Building for CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(ENABLE_UCX OFF)
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(FAST_BUILD ON)
    set(NVTX_DISABLE ON)
    set(INDEX_RANGE_CHECK ON)
else ()
    set(FAST_BUILD OFF)
    set(FAST_MATH ON)
    set(NVTX_DISABLE OFF)
    set(INDEX_RANGE_CHECK OFF)
endif ()

find_package(Python3 REQUIRED Interpreter)

fetchcontent_declare(
        trtllm
        GIT_REPOSITORY https://github.com/nvidia/TensorRT-LLM.git
        GIT_TAG v1.0.0
        GIT_SHALLOW ON
        DOWNLOAD_EXTRACT_TIMESTAMP
)
fetchcontent_makeavailable(trtllm)

message(STATUS "Found TensorRT-LLM: ${trtllm_SOURCE_DIR}")
execute_process(COMMAND git lfs install WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")
execute_process(COMMAND git lfs pull WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")

# Patch the wrong command name
execute_process(COMMAND sed -i "s/Python_EXECUTABLE/Python3_EXECUTABLE/" cpp/CMakeLists.txt WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")

# Generate fmha cu
# See https://github.com/NVIDIA/TensorRT-LLM/blob/ae8270b713446948246f16fadf4e2a32e35d0f62/scripts/build_wheel.py#L248-L279
execute_process(COMMAND rm -r "cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu" WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")
execute_process(COMMAND mkdir -p "cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu" WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/")
execute_process(COMMAND rm -r "generated" WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/cpp/kernels/fmha_v2")
execute_process(COMMAND rm -r "temp" WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/cpp/kernels/fmha_v2")
execute_process(COMMAND rm -r "obj" WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/cpp/kernels/fmha_v2")
execute_process(
    COMMAND
        ${CMAKE_COMMAND} -E env 
        TORCH_CUDA_ARCH_LIST=9.0
        ENABLE_SM89_QMMA=1
        ENABLE_HMMA_FP32=1
        GENERATE_CUBIN=1
        SCHEDULING_MODE=1
        ENABLE_SM100=1
        ENABLE_SM120=1
        GENERATE_CU_TRTLLM=true
        ${Python3_EXECUTABLE} "setup.py"
    WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/cpp/kernels/fmha_v2"
    RESULT_VARIABLE _PYTHON_SUCCESS
)
if(NOT _PYTHON_SUCCESS EQUAL 0)
    message(FATAL_ERROR "generate fmha cu")
endif()

execute_process(COMMAND mv "generated/fmha_cubin.h" "${trtllm_SOURCE_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin"  WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/cpp/kernels/fmha_v2")
file(GLOB GENERATED_FMHA_CUS RELATIVE "${trtllm_SOURCE_DIR}/cpp/kernels/fmha_v2" "${trtllm_SOURCE_DIR}/cpp/kernels/fmha_v2/generated/*sm*.cu")
foreach(GENERATED_FMHA_CU ${GENERATED_FMHA_CUS})
    execute_process(COMMAND mv ${GENERATED_FMHA_CU} "${trtllm_SOURCE_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu"  WORKING_DIRECTORY "${trtllm_SOURCE_DIR}/cpp/kernels/fmha_v2" RESULT_VARIABLE _MV_SUCCESS)
    if(NOT _MV_SUCCESS EQUAL 0)
        message(FATAL_ERROR "move fmha cu")
    endif()
endforeach()

# The same Executor Static library
set(TRTLLM_EXECUTOR_STATIC_LIBRARY_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}tensorrt_llm_executor_static${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE INTERNAL "executor_static library name")
set(TRTLLM_EXECUTOR_STATIC_LIBRARY_PATH "${trtllm_SOURCE_DIR}/cpp/tensorrt_llm/executor/${CMAKE_LIBRARY_ARCHITECTURE}/${TRTLLM_EXECUTOR_STATIC_LIBRARY_NAME}" CACHE INTERNAL "executor_static library path")
