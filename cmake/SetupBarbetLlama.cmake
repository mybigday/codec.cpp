# SetupBarbetLlama.cmake — apply the Barbet arch patch to the pinned llama.cpp
# submodule at configure time, idempotently.
#
# llama.cpp lives at common/third-party/llama.cpp as a submodule pinned to the
# commit below (see patches/barbet-llamacpp-README.md). The Barbet changes are
# kept as patches/barbet-llamacpp.patch rather than committed into the submodule;
# this module is what actually applies them, so the patch is part of the build
# instead of a manual README step.
#
# Behaviour:
#   - submodule not checked out      -> status message, skip (no forced download).
#   - patch already applied          -> no-op (idempotent across re-configures).
#   - patch applies cleanly          -> apply it.
#   - patch no longer applies        -> FATAL: upstream drifted past the pin,
#                                       re-pin + regenerate the patch.
#
# Set -DCODEC_BARBET_PATCH=OFF to disable entirely.

option(CODEC_BARBET_PATCH "Apply the Barbet patch to the llama.cpp submodule at configure time" ON)
if(NOT CODEC_BARBET_PATCH)
    return()
endif()

set(_barbet_base  "f708a5b2caaee0226c0af220e366785699ba41e2")
set(_barbet_dir   "${CMAKE_CURRENT_SOURCE_DIR}/common/third-party/llama.cpp")
set(_barbet_patch "${CMAKE_CURRENT_SOURCE_DIR}/patches/barbet-llamacpp.patch")

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(STATUS "Barbet: git not found — skipping llama.cpp patch setup")
    return()
endif()

# Submodule checked out? (cheap content probe — its top-level CMakeLists.)
if(NOT EXISTS "${_barbet_dir}/CMakeLists.txt")
    message(STATUS "Barbet: llama.cpp submodule not initialized — skipping patch.\n"
                   "        Run: git submodule update --init -- common/third-party/llama.cpp")
    return()
endif()

# Warn (don't fail) if HEAD has drifted off the pinned base — the patch may still
# apply, but the build is no longer reproducing the recorded combination.
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${_barbet_dir}"
    OUTPUT_VARIABLE _barbet_head OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _rc)
if(_rc EQUAL 0 AND NOT _barbet_head STREQUAL _barbet_base)
    message(WARNING "Barbet: llama.cpp submodule HEAD is ${_barbet_head},\n"
                    "        expected pinned base ${_barbet_base}.")
endif()

# Already applied? `git apply --reverse --check` succeeds iff the patch is present.
execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_barbet_patch}"
    WORKING_DIRECTORY "${_barbet_dir}"
    RESULT_VARIABLE _reverse_rc ERROR_QUIET OUTPUT_QUIET)
if(_reverse_rc EQUAL 0)
    message(STATUS "Barbet: patch already applied to llama.cpp submodule")
    return()
endif()

# Not applied — verify it still applies cleanly to this checkout.
execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --check "${_barbet_patch}"
    WORKING_DIRECTORY "${_barbet_dir}"
    RESULT_VARIABLE _check_rc ERROR_QUIET OUTPUT_QUIET)
if(NOT _check_rc EQUAL 0)
    message(FATAL_ERROR
        "Barbet: ${_barbet_patch}\n"
        "        does not apply to llama.cpp at ${_barbet_head}.\n"
        "        Upstream has likely drifted past the pinned base ${_barbet_base};\n"
        "        re-pin the submodule and regenerate the patch, or set -DCODEC_BARBET_PATCH=OFF.")
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply "${_barbet_patch}"
    WORKING_DIRECTORY "${_barbet_dir}"
    RESULT_VARIABLE _apply_rc)
if(NOT _apply_rc EQUAL 0)
    message(FATAL_ERROR "Barbet: failed to apply ${_barbet_patch} (rc=${_apply_rc})")
endif()
message(STATUS "Barbet: applied barbet-llamacpp.patch to llama.cpp submodule")
