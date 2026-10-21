# SetupTtsBackbone.cmake — build an ISOLATED llama.cpp backbone for tts-cli.
#
# codec.cpp links ggml 0.9 (the standalone ggml submodule) as a shared
# libggml.so.0; the pinned llama.cpp submodule bundles ggml 0.15 with the
# SAME soname.  add_subdirectory'ing llama.cpp would (a) collide the `ggml`
# CMake target and (b) at runtime load only one libggml.so.0 → ABI crash.
#
# Isolation strategy (proven): build llama.cpp + its ggml STATIC + PIC via
# ExternalProject (cmake/ttsbackbone) and wrap them into ONE shared object,
# libttsbackbone, whose dynamic symbol table exports ONLY `llama_*` and the
# common_sampler API (every ggml_* symbol is local).  tts-cli then links
# codec (→ shared ggml) + libttsbackbone (self-contained ggml), and the two
# ggml instances never see each other's symbols.
#
# The backbone is built for the platform codec.cpp is: every GGML_* option
# (backends, CPU ISA / arch, backend tuning) and the toolchain / cross-compile
# settings are forwarded, so `tts-cli synthesize --gpu` offloads both, as a
# llama.cpp host would.
# Set -DCODEC_TTS_BACKBONE=OFF to skip (tts-cli then builds without
# `synthesize`, which #ifdef's out).

include(ExternalProject)

set(_ll_dir "${CMAKE_CURRENT_SOURCE_DIR}/common/third-party/llama.cpp")
set(TTS_BACKBONE_SO
    "${CMAKE_BINARY_DIR}/ttsbackbone/${CMAKE_SHARED_LIBRARY_PREFIX}ttsbackbone${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Every GGML_* cache option, at the value codec.cpp's ggml sees.  Left out:
# the ones the isolation fixes (static, no backend DL), install paths and
# find results.  The two ggml copies share their option names, so an option
# one of them lacks is just unused.
set(_ll_args "")
get_cmake_property(_ll_cache_vars CACHE_VARIABLES)
foreach(_v ${_ll_cache_vars})
    if(NOT _v MATCHES "^GGML_" OR
       _v MATCHES "^GGML_(BACKEND_DL|BACKEND_DIR|STATIC|BUILD_TESTS|BUILD_EXAMPLES|ALL_WARNINGS.*)$" OR
       _v MATCHES "(_INSTALL_DIR|_FOUND)$")
        continue()
    endif()
    get_property(_t CACHE ${_v} PROPERTY TYPE)
    if(_t STREQUAL "INTERNAL" OR _t STREQUAL "STATIC")
        continue()
    endif()
    string(REPLACE ";" "|" _val "${${_v}}")
    list(APPEND _ll_args "-D${_v}:${_t}=${_val}")
endforeach()

# Toolchain / target platform (cross builds: Android NDK, iOS / macOS
# architectures, CUDA / HIP GPU archs).
foreach(_v CMAKE_TOOLCHAIN_FILE CMAKE_C_COMPILER CMAKE_CXX_COMPILER
           CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER CMAKE_C_FLAGS CMAKE_CXX_FLAGS
           CMAKE_SYSTEM_NAME CMAKE_SYSTEM_PROCESSOR CMAKE_SYSTEM_VERSION
           CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_OSX_SYSROOT
           CMAKE_FIND_ROOT_PATH CMAKE_PREFIX_PATH
           ANDROID_ABI ANDROID_PLATFORM ANDROID_NDK ANDROID_STL
           CMAKE_CUDA_COMPILER CMAKE_CUDA_HOST_COMPILER CMAKE_CUDA_ARCHITECTURES
           CMAKE_HIP_COMPILER CMAKE_HIP_ARCHITECTURES AMDGPU_TARGETS
           Vulkan_INCLUDE_DIR Vulkan_LIBRARY Vulkan_GLSLC_EXECUTABLE)
    if(DEFINED ${_v} AND NOT "${${_v}}" STREQUAL "")
        string(REPLACE ";" "|" _val "${${_v}}")
        list(APPEND _ll_args "-D${_v}=${_val}")
    endif()
endforeach()

ExternalProject_Add(ttsbackbone
    SOURCE_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ttsbackbone"
    PREFIX          "${CMAKE_BINARY_DIR}/ttsbackbone-build"
    LIST_SEPARATOR  |
    CMAKE_ARGS
        -DLLAMA_DIR=${_ll_dir}
        -DTTSBACKBONE_OUTPUT_DIR=${CMAKE_BINARY_DIR}/ttsbackbone
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_TOOLS=OFF
        -DLLAMA_BUILD_SERVER=OFF
        -DLLAMA_BUILD_APP=OFF
        -DLLAMA_BUILD_COMMON=ON
        -DLLAMA_CURL=OFF
        -DLLAMA_OPENSSL=OFF
        ${_ll_args}
        -DGGML_BACKEND_DL=OFF
    BUILD_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttsbackbone --config Release
    # Rebuild on every build: the patched sources change outside ExternalProject's stamps.
    BUILD_ALWAYS    ON
    BUILD_BYPRODUCTS "${TTS_BACKBONE_SO}"
    INSTALL_COMMAND ""
    UPDATE_COMMAND  ""
)

# Exported for the tts-cli target.  `common` is added so the runner can
# include common.h / sampling.h (the common_sampler API).
set(TTS_BACKBONE_INCLUDE_DIRS
    "${_ll_dir}/include"
    "${_ll_dir}/ggml/include"
    "${_ll_dir}/common"
    CACHE INTERNAL "llama backbone include dirs")
