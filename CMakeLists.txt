cmake_minimum_required(VERSION 3.20)
project(tc_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(TC_SANITIZE   "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(TC_WITH_LZ4   "Use lz4 as a benchmark baseline when found" ON)
option(TC_WITH_ZSTD  "Use zstd as a benchmark baseline when found" ON)
option(TC_WITH_ZLIB  "Use zlib as a benchmark baseline when found" ON)
option(TC_BUILD_BENCH "Build tc_bench" ON)

# ---------------------------------------------------------------------------
# SIMD detection. The scalar path always exists; a SIMD TU is compiled only
# when the *target* supports it. Never hard-require AVX2.
# ---------------------------------------------------------------------------
set(TC_HAVE_NEON OFF)
set(TC_HAVE_AVX2 OFF)

# CMAKE_OSX_ARCHITECTURES wins over the host processor when cross-building.
set(_tc_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
    set(_tc_arch "${CMAKE_OSX_ARCHITECTURES}")
endif()

if(_tc_arch MATCHES "^(arm64|aarch64|ARM64)$")
    # Advanced SIMD (NEON) is part of the AArch64 baseline: no flag needed.
    set(TC_HAVE_NEON ON)
elseif(_tc_arch MATCHES "^(x86_64|AMD64|amd64)$")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" TC_COMPILER_HAS_MAVX2)
    if(TC_COMPILER_HAS_MAVX2)
        set(TC_HAVE_AVX2 ON)
    endif()
endif()
message(STATUS "TC: arch=${_tc_arch} NEON=${TC_HAVE_NEON} AVX2=${TC_HAVE_AVX2}")

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
set(TC_SOURCES
    src/tensor.cpp
    src/format.cpp
    src/compressor.cpp
    src/synthetic.cpp
    src/analyze.cpp
    src/codec/byte_shuffle.cpp
    src/codec/xor_delta.cpp
    src/codec/rle.cpp
    src/codec/block.cpp
    src/runtime/layer_pager.cpp
)
if(TC_HAVE_NEON)
    list(APPEND TC_SOURCES src/simd/byte_shuffle_neon.cpp)
endif()
if(TC_HAVE_AVX2)
    list(APPEND TC_SOURCES src/simd/byte_shuffle_avx2.cpp)
    # Only this one TU is built with -mavx2; it is entered only after a
    # runtime CPUID check, so the rest of the library stays baseline x86-64.
    set_source_files_properties(src/simd/byte_shuffle_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

add_library(tc_runtime ${TC_SOURCES})
target_include_directories(tc_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src   # codec stage headers, used by tests/bench for ablations
)
if(TC_HAVE_NEON)
    target_compile_definitions(tc_runtime PUBLIC TC_HAVE_NEON=1)
endif()
if(TC_HAVE_AVX2)
    target_compile_definitions(tc_runtime PUBLIC TC_HAVE_AVX2=1)
endif()

add_library(tc_build_flags INTERFACE)
if(MSVC)
    target_compile_options(tc_build_flags INTERFACE /W4)
else()
    target_compile_options(tc_build_flags INTERFACE -Wall -Wextra -Wpedantic)
endif()
if(TC_SANITIZE)
    # ASan's dyld interpose deadlocks during process init on some recent
    # macOS + AppleClang combinations (spin in AsanInitInternal). Use UBSan
    # alone on Apple; keep ASan+UBSan elsewhere.
    if(APPLE)
        message(STATUS "TC: TC_SANITIZE on Apple -> UndefinedBehaviorSanitizer only (ASan init deadlock workaround)")
        target_compile_options(tc_build_flags INTERFACE
            -fsanitize=undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
        target_link_options(tc_build_flags INTERFACE -fsanitize=undefined)
    else()
        target_compile_options(tc_build_flags INTERFACE
            -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
        target_link_options(tc_build_flags INTERFACE -fsanitize=address,undefined)
    endif()
endif()
target_link_libraries(tc_runtime PUBLIC tc_build_flags)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
add_executable(tc src/main.cpp)
target_link_libraries(tc PRIVATE tc_runtime)

# ---------------------------------------------------------------------------
# Tests (in-repo harness, one CTest entry per test group)
# ---------------------------------------------------------------------------
enable_testing()

add_executable(tc_tests
    tests/test_main.cpp
    tests/test_format.cpp
    tests/test_tensor.cpp
    tests/test_codec.cpp
    tests/test_roundtrip.cpp
    tests/test_malformed.cpp
    tests/test_simd.cpp
    tests/test_analyze.cpp
    tests/test_pager.cpp
    tests/test_huffman.cpp
    benchmarks/huffman_experiment.cpp
)
target_include_directories(tc_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
target_link_libraries(tc_tests PRIVATE tc_runtime)

foreach(group format tensor codec roundtrip malformed simd analyze pager huffman)
    add_test(NAME ${group} COMMAND tc_tests ${group})
endforeach()

# Separate binary: it replaces global operator new to count allocations.
add_executable(tc_alloc_test tests/test_alloc.cpp)
target_link_libraries(tc_alloc_test PRIVATE tc_runtime)
add_test(NAME alloc COMMAND tc_alloc_test)

# ---------------------------------------------------------------------------
# Benchmarks. Baselines are optional: a missing library is reported by
# tc_bench as "not built", never silently dropped.
# ---------------------------------------------------------------------------
if(TC_BUILD_BENCH)
    find_package(Threads REQUIRED)
    add_executable(tc_bench
        benchmarks/bench_codec.cpp
        benchmarks/baselines.cpp
        benchmarks/sysinfo.cpp
        benchmarks/huffman_experiment.cpp
    )
    target_include_directories(tc_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    target_link_libraries(tc_bench PRIVATE tc_runtime Threads::Threads)
    target_compile_definitions(tc_bench PRIVATE
        TC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        TC_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        TC_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        TC_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        TC_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}"
    )

    set(_tc_hints /opt/homebrew /usr/local)

    if(TC_WITH_ZLIB)
        find_package(ZLIB QUIET)
        if(ZLIB_FOUND)
            target_link_libraries(tc_bench PRIVATE ZLIB::ZLIB)
            target_compile_definitions(tc_bench PRIVATE TC_BENCH_HAVE_ZLIB=1)
        endif()
    endif()

    if(TC_WITH_LZ4)
        find_path(TC_LZ4_INCLUDE lz4.h HINTS ${_tc_hints} PATH_SUFFIXES include)
        find_library(TC_LZ4_LIB lz4 HINTS ${_tc_hints} PATH_SUFFIXES lib)
        if(TC_LZ4_INCLUDE AND TC_LZ4_LIB)
            target_include_directories(tc_bench SYSTEM PRIVATE ${TC_LZ4_INCLUDE})
            target_link_libraries(tc_bench PRIVATE ${TC_LZ4_LIB})
            target_compile_definitions(tc_bench PRIVATE TC_BENCH_HAVE_LZ4=1)
        endif()
    endif()

    if(TC_WITH_ZSTD)
        find_path(TC_ZSTD_INCLUDE zstd.h HINTS ${_tc_hints} PATH_SUFFIXES include)
        find_library(TC_ZSTD_LIB zstd HINTS ${_tc_hints} PATH_SUFFIXES lib)
        if(TC_ZSTD_INCLUDE AND TC_ZSTD_LIB)
            target_include_directories(tc_bench SYSTEM PRIVATE ${TC_ZSTD_INCLUDE})
            target_link_libraries(tc_bench PRIVATE ${TC_ZSTD_LIB})
            target_compile_definitions(tc_bench PRIVATE TC_BENCH_HAVE_ZSTD=1)
        endif()
    endif()

    message(STATUS "TC bench baselines: zlib=${ZLIB_FOUND} lz4=${TC_LZ4_LIB} zstd=${TC_ZSTD_LIB}")
endif()

# Stretch: fused streaming / multithreaded pager microbench
option(TC_BUILD_FUSED "Build tc_fused_stream stretch microbench" ON)
if(TC_BUILD_FUSED)
    find_package(Threads REQUIRED)
    add_executable(tc_fused_stream benchmarks/fused_stream.cpp)
    target_link_libraries(tc_fused_stream PRIVATE tc_runtime Threads::Threads)
endif()
