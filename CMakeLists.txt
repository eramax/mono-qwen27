cmake_minimum_required(VERSION 3.18)
project(mono27b LANGUAGES CXX CUDA)

option(MONO27B_TIMING "Enable per-step CUDA timing instrumentation" OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(mono27b_chat
    src/mono27b_chat.cpp
    src/mono27b_cli.cpp
    src/mono27b_blob.cpp
    src/mono27b_gguf.cpp
    src/mono27b_tokenizer.cpp
    src/mono27b_executor.cu
)

add_executable(mono27b_pack
    src/mono27b_pack.cpp
    src/mono27b_cli.cpp
    src/mono27b_blob.cpp
    src/mono27b_packer.cpp
    src/mono27b_gguf.cpp
)

target_compile_features(mono27b_chat PRIVATE cxx_std_17)
target_compile_features(mono27b_pack PRIVATE cxx_std_17)

if(MONO27B_TIMING)
    target_compile_definitions(mono27b_chat PRIVATE MONO27B_TIMING)
endif()

target_include_directories(mono27b_chat PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_include_directories(mono27b_pack PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set_target_properties(mono27b_chat PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

set_target_properties(mono27b_pack PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(mono27b_chat PRIVATE CUDA::cudart)
target_compile_options(mono27b_chat PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>
)
