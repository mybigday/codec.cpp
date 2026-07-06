# SetupTtsBackbone.cmake — build an ISOLATED llama.cpp backbone for tts-cli.
#
# codec.cpp links ggml 0.9 (the standalone ggml submodule) as a shared
# libggml.so.0; the pinned llama.cpp submodule bundles ggml 0.15 with the
# SAME soname.  add_subdirectory'ing llama.cpp would (a) collide the `ggml`
# CMake target and (b) at runtime load only one libggml.so.0 → ABI crash.
#
# Isolation strategy (proven): build llama.cpp + its ggml 0.15 STATIC + PIC
# via ExternalProject, then wrap the static archives into ONE shared object
# `libttsbackbone.so` whose dynamic symbol table exports ONLY `llama_*`
# (version script hides every ggml_* symbol as local).  tts-cli then links
# codec (→ shared ggml 0.9) + libttsbackbone.so (self-contained ggml 0.15),
# and the two ggml instances never see each other's symbols.
#
# CPU-only; the backbone is a small semantic LM, GPU offload is not needed
# for the reference host.  Set -DCODEC_TTS_BACKBONE=OFF to skip (tts-cli
# then builds without `synthesize`, which #ifdef's out).

include(ExternalProject)

set(_ll_dir      "${CMAKE_CURRENT_SOURCE_DIR}/common/third-party/llama.cpp")
set(_ll_prefix   "${CMAKE_BINARY_DIR}/llama-static")
set(_ll_libdir   "${_ll_prefix}/src/llama-build/src")
set(_ll_ggmldir  "${_ll_prefix}/src/llama-build/ggml/src")
set(_ll_commondir "${_ll_prefix}/src/llama-build/common")
set(_ll_vendordir "${_ll_prefix}/src/llama-build/vendor")

# Static archives ExternalProject will produce.
#
# The `common` layer (LLAMA_BUILD_COMMON=ON below) gives us the shared
# sampling API (common_sampler_*, common_params_sampling, common_grammar
# GBNF integration) instead of hand-rolled llama_sampler chains.  Its
# CMake target is `llama-common` → libllama-common.a, which drags in
# libllama-common-base.a (build-info) and vendor/cpp-httplib (a PRIVATE
# header-lib TU).  All three MUST sit inside the --whole-archive group at
# wrapper-link time (below): nothing in llama/ggml references common's
# symbols, so outside --whole-archive the linker would drop the objects
# before the version script could export common_sampler_*.  Verified: none
# of these archives DEFINE bare ggml_* symbols (they reference them as
# undefined, resolved internally against the bundled ggml 0.15) so the
# `local: *` catch-all still hides every ggml_* — no ABI leak.
set(_ll_archives
    "${_ll_libdir}/libllama.a"
    "${_ll_commondir}/libllama-common.a"
    "${_ll_commondir}/libllama-common-base.a"
    "${_ll_vendordir}/cpp-httplib/libcpp-httplib.a"
    "${_ll_ggmldir}/libggml.a"
    "${_ll_ggmldir}/libggml-cpu.a"
    "${_ll_ggmldir}/libggml-base.a")

ExternalProject_Add(llama_static
    SOURCE_DIR      "${_ll_dir}"
    PREFIX          "${_ll_prefix}"
    BINARY_DIR      "${_ll_prefix}/src/llama-build"
    CMAKE_ARGS
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
        -DGGML_NATIVE=ON
        -DGGML_VULKAN=OFF
        -DGGML_BACKEND_DL=OFF
    BUILD_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-common --config Release
    BUILD_BYPRODUCTS ${_ll_archives}
    INSTALL_COMMAND ""
    UPDATE_COMMAND  ""
)

# Version script: export llama_* (C ABI) + the common_sampler_* C++ API on
# the wrapper's dynamic table; hide everything else (ggml_* included) as
# local.  The `extern "C++"` block matches the DEMANGLED name, and the glob
# MUST be unquoted for `*` to act as a wildcard (a quoted "common_sampler*"
# is a literal and matches nothing).  This exports common_sampler_init /
# _free / _accept / _reset / _sample / _get and friends.  common_grammar
# and common_params_sampling need no exported symbols (plain struct; its
# ctor/dtor are emitted inline in the codec_tts_runner TU).
set(_ll_vscript "${CMAKE_BINARY_DIR}/ttsbackbone.map")
file(WRITE "${_ll_vscript}"
"{\n  global:\n    llama_*;\n    extern \"C++\" {\n      common_sampler*;\n    };\n  local:\n    *;\n};\n")

set(TTS_BACKBONE_SO "${CMAKE_BINARY_DIR}/ttsbackbone/libttsbackbone.so")
add_custom_command(
    OUTPUT  "${TTS_BACKBONE_SO}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/ttsbackbone"
    COMMAND ${CMAKE_CXX_COMPILER} -shared -fPIC -o "${TTS_BACKBONE_SO}"
            -Wl,--whole-archive ${_ll_archives} -Wl,--no-whole-archive
            -Wl,--version-script=${_ll_vscript}
            -lpthread -lm -ldl -lgomp
    DEPENDS llama_static ${_ll_archives} "${_ll_vscript}"
    COMMENT "Linking isolated llama backbone → libttsbackbone.so (ggml_* hidden)"
    VERBATIM)
add_custom_target(ttsbackbone DEPENDS "${TTS_BACKBONE_SO}")

# Exported for the tts-cli target.  `common` is added so the runner can
# include common.h / sampling.h (the common_sampler API).
set(TTS_BACKBONE_INCLUDE_DIRS
    "${_ll_dir}/include"
    "${_ll_dir}/ggml/include"
    "${_ll_dir}/common"
    CACHE INTERNAL "llama backbone include dirs")
