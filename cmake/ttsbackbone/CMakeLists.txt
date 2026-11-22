# libttsbackbone: the pinned llama.cpp (+ its bundled ggml, static) wrapped
# into one shared library that exports only llama_* and the common_sampler
# C++ API; every ggml_* symbol stays local (see SetupTtsBackbone.cmake).
# Built as an ExternalProject of codec.cpp, which forwards its GGML_* options
# and toolchain, so each backend's link dependencies come from llama.cpp's
# own targets.

cmake_minimum_required(VERSION 3.16)
project(ttsbackbone C CXX)

set(LLAMA_DIR "" CACHE PATH "llama.cpp source directory")
set(TTSBACKBONE_OUTPUT_DIR "${CMAKE_BINARY_DIR}" CACHE PATH "where libttsbackbone goes")

add_subdirectory("${LLAMA_DIR}" llama EXCLUDE_FROM_ALL)

# ggml-vulkan.cpp at -O3: LLVM's register coalescer takes ~14 minutes on it
# (NDK clang 21, 822 of 880 s); -O2 compiles it in under a minute.  It is
# host-side dispatch code, so the GPU kernels run as fast either way.
if(TARGET ggml-vulkan AND NOT MSVC)
    target_compile_options(ggml-vulkan PRIVATE -O2)
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ttsbackbone.c" "")
add_library(ttsbackbone SHARED "${CMAKE_CURRENT_BINARY_DIR}/ttsbackbone.c")
set_target_properties(ttsbackbone PROPERTIES
    LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY "${TTSBACKBONE_OUTPUT_DIR}")
target_link_libraries(ttsbackbone PRIVATE llama llama-common)

# Nothing in the wrapper references llama / common, so their archives go in
# whole; ggml's objects are pulled by their references (static backend
# registry, GGML_BACKEND_DL=OFF).
if(WIN32)
    # A PE image binds each import to a named DLL, so the two ggml copies
    # cannot clash; the export list is a .def file taken from the llama /
    # common objects (llama.cpp built static has no dllexport).
    set_target_properties(ttsbackbone PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TTSBACKBONE_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${TTSBACKBONE_OUTPUT_DIR}")
    set(_objs "${CMAKE_CURRENT_BINARY_DIR}/ttsbackbone.objs")
    set(_def  "${CMAKE_CURRENT_BINARY_DIR}/ttsbackbone.def")
    file(GENERATE OUTPUT "${_objs}"
         CONTENT "$<JOIN:$<TARGET_OBJECTS:llama>;$<TARGET_OBJECTS:llama-common>,\n>\n")
    add_custom_command(TARGET ttsbackbone PRE_LINK
        COMMAND ${CMAKE_COMMAND} -DOBJS=${_objs} -DDEF=${_def}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/windows-def.cmake"
        VERBATIM)
    target_link_options(ttsbackbone PRIVATE
        "/WHOLEARCHIVE:$<TARGET_FILE:llama>" "/WHOLEARCHIVE:$<TARGET_FILE:llama-common>" "/DEF:${_def}")
elseif(APPLE)
    target_link_options(ttsbackbone PRIVATE
        "LINKER:-force_load,$<TARGET_FILE:llama>"
        "LINKER:-force_load,$<TARGET_FILE:llama-common>")
    set(_exports "${CMAKE_CURRENT_BINARY_DIR}/ttsbackbone.exports")
    file(WRITE "${_exports}" "_llama_*\n__Z*common_sampler*\n")
    target_link_options(ttsbackbone PRIVATE "LINKER:-exported_symbols_list,${_exports}")
else()
    target_link_options(ttsbackbone PRIVATE
        "LINKER:--whole-archive" "$<TARGET_FILE:llama>" "$<TARGET_FILE:llama-common>" "LINKER:--no-whole-archive")
    # The `extern "C++"` block matches the demangled name; the glob must be
    # unquoted to act as a wildcard.
    set(_exports "${CMAKE_CURRENT_BINARY_DIR}/ttsbackbone.map")
    file(WRITE "${_exports}"
        "{\n  global:\n    llama_*;\n    extern \"C++\" {\n      common_sampler*;\n    };\n  local:\n    *;\n};\n")
    target_link_options(ttsbackbone PRIVATE "LINKER:--version-script=${_exports}")
endif()
