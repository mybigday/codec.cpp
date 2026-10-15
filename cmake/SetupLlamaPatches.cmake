# SetupLlamaPatches.cmake — apply codec.cpp's llama.cpp arch patches to the
# pinned submodule at configure time, idempotently, in order.
#
# llama.cpp lives at common/third-party/llama.cpp as a submodule pinned to the
# commit below. Arch additions are kept as patches (not committed into the
# submodule); each patch is generated on top of the previous ones:
#   1. patches/barbet-llamacpp.patch     (BlueMagpie backbone, see barbet-llamacpp-README.md)
#   2. patches/cosyvoice-llamacpp.patch  (CosyVoice-300M / BreezyVoice speech LM)
#
# Behaviour:
#   - submodule not checked out          -> status message, skip.
#   - the last N patches reverse-apply   -> those are present; apply the rest.
#   - a remaining patch applies cleanly  -> apply it.
#   - a remaining patch does not apply   -> FATAL: upstream drifted past the pin.
#
# Set -DCODEC_LLAMA_PATCHES=OFF to disable entirely.

option(CODEC_LLAMA_PATCHES "Apply codec.cpp's arch patches to the llama.cpp submodule at configure time" ON)
if(NOT CODEC_LLAMA_PATCHES)
    return()
endif()

set(_lp_base "f708a5b2caaee0226c0af220e366785699ba41e2")
set(_lp_dir  "${CMAKE_CURRENT_SOURCE_DIR}/common/third-party/llama.cpp")
set(_lp_patches
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/barbet-llamacpp.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/cosyvoice-llamacpp.patch")

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(STATUS "llama.cpp patches: git not found — skipping")
    return()
endif()
if(NOT EXISTS "${_lp_dir}/CMakeLists.txt")
    message(STATUS "llama.cpp patches: submodule not initialized — skipping.\n"
                   "        Run: git submodule update --init -- common/third-party/llama.cpp")
    return()
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${_lp_dir}"
    OUTPUT_VARIABLE _lp_head OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _rc)
if(_rc EQUAL 0 AND NOT _lp_head STREQUAL _lp_base)
    message(WARNING "llama.cpp patches: submodule HEAD is ${_lp_head},\n"
                    "        expected pinned base ${_lp_base}.")
endif()

# Patches are stacked, so only the newest applied one reverse-applies cleanly:
# scan from the end for it; everything before it is present.
list(LENGTH _lp_patches _lp_n)
set(_lp_first 0)
math(EXPR _lp_i "${_lp_n} - 1")
while(_lp_i GREATER_EQUAL 0)
    list(GET _lp_patches ${_lp_i} _p)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_p}"
        WORKING_DIRECTORY "${_lp_dir}"
        RESULT_VARIABLE _reverse_rc ERROR_QUIET OUTPUT_QUIET)
    if(_reverse_rc EQUAL 0)
        math(EXPR _lp_first "${_lp_i} + 1")
        break()
    endif()
    math(EXPR _lp_i "${_lp_i} - 1")
endwhile()

set(_lp_i ${_lp_first})
while(_lp_i LESS _lp_n)
    list(GET _lp_patches ${_lp_i} _p)
    get_filename_component(_pn "${_p}" NAME)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --check "${_p}"
        WORKING_DIRECTORY "${_lp_dir}"
        RESULT_VARIABLE _check_rc ERROR_QUIET OUTPUT_QUIET)
    if(NOT _check_rc EQUAL 0)
        message(FATAL_ERROR
            "llama.cpp patches: ${_pn} does not apply to llama.cpp at ${_lp_head}.\n"
            "        Upstream has likely drifted past the pinned base ${_lp_base};\n"
            "        re-pin the submodule and regenerate the patches, or set -DCODEC_LLAMA_PATCHES=OFF.")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply "${_p}"
        WORKING_DIRECTORY "${_lp_dir}"
        RESULT_VARIABLE _apply_rc)
    if(NOT _apply_rc EQUAL 0)
        message(FATAL_ERROR "llama.cpp patches: failed to apply ${_pn} (rc=${_apply_rc})")
    endif()
    message(STATUS "llama.cpp patches: applied ${_pn}")
    math(EXPR _lp_i "${_lp_i} + 1")
endwhile()
# A newly applied patch can add sources (llama.cpp globs src/models/*.cpp at
# configure time): drop the backbone ExternalProject's stamps so it re-runs
# configure + build (cmake/SetupTtsBackbone.cmake).
if(_lp_first LESS _lp_n)
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/llama-static/src/llama_static-stamp")
endif()
if(_lp_first EQUAL _lp_n)
    message(STATUS "llama.cpp patches: all ${_lp_n} patches already applied")
endif()
