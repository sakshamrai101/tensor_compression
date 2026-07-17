cmake_minimum_required(VERSION 3.20)
project(tc_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library: codec + format + tensor helpers (Phase 0 stubs; Phase 1 fills codec)
add_library(tc_runtime
    src/tensor.cpp
    src/compressor.cpp
    src/format.cpp
)

target_include_directories(tc_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# CLI entrypoint
add_executable(tc src/main.cpp)
target_link_libraries(tc PRIVATE tc_runtime)
