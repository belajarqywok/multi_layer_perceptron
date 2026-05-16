cmake_minimum_required(VERSION 3.14)
project(mlp_from_scratch C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "clang++")

# Optional: Add compiler flags for warnings and optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Create source directory structure if not exists
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Find Protobuf
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Generate protobuf cpp sources
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS src/onnx.proto)

# Core source files
set(SOURCES
    src/main.cpp
    ${PROTO_SRCS}
)

# Executable
add_executable(mlp ${SOURCES})

# Link Protobuf and OpenCL
find_package(OpenCL REQUIRED)
target_link_libraries(mlp ${Protobuf_LIBRARIES} OpenCL::OpenCL)

# Inference C++ Executable
add_executable(inference_cpp inferences/inference.cpp)
include_directories(${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-x64-1.20.0/include)
target_link_libraries(inference_cpp ${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so)

# Inference C Executable
add_executable(inference_c inferences/inference.c)
target_link_libraries(inference_c ${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so)

