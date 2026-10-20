cmake_minimum_required(VERSION 3.16)
project(codec C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GGML Backend Options (pass-through to ggml subproject)
# Goal: allow building with any ggml-native accelerator backend on other machines.
# These options mirror ggml/llama.cpp-style flags.
option(GGML_CUDA        "Enable CUDA backend"       OFF)
option(GGML_HIP         "Enable HIP/ROCm backend"   OFF)
option(GGML_VULKAN      "Enable Vulkan backend"     OFF)
option(GGML_METAL       "Enable Metal backend"      OFF)
option(GGML_SYCL        "Enable SYCL backend"       OFF)
option(GGML_OPENCL      "Enable OpenCL backend"     OFF)
option(GGML_CANN        "Enable CANN backend"       OFF)
option(GGML_MUSA        "Enable MUSA backend"       OFF)
option(GGML_WEBGPU      "Enable WebGPU backend"     OFF)
option(GGML_ZDNN        "Enable zDNN backend"       OFF)
option(GGML_VIRTGPU     "Enable VirtGPU backend"    OFF)
option(GGML_RPC         "Enable RPC backend"        OFF)
option(GGML_BLAS        "Enable BLAS backend"       OFF)

# CPU SIMD optimizations (default: auto-detect enabled)
option(GGML_AVX         "Enable AVX"                ON)
option(GGML_AVX2        "Enable AVX2"               ON)
option(GGML_AVX512      "Enable AVX512"             ON)
option(GGML_FMA         "Enable FMA"                ON)
option(GGML_NEON        "Enable ARM NEON"           ON)
option(GGML_SVE         "Enable ARM SVE"            ON)

set(GGML_CPU ON)

# Installed libraries find each other in their own directory (libggml loads
# its backends, libcodec loads libggml); the tools override this below.
if(APPLE)
    set(_codec_origin "@loader_path")
else()
    set(_codec_origin "$ORIGIN")
endif()
if(NOT CMAKE_INSTALL_RPATH)
    set(CMAKE_INSTALL_RPATH "${_codec_origin}")
endif()

add_subdirectory(ggml)

set(CODEC_SOURCES
    src/codec.cpp
    src/runtime/gguf_kv.cpp
    src/runtime/tensor_utils.cpp
    src/runtime/audio_dsp.cpp
    src/runtime/graph.cpp
    src/runtime/graph_exec.cpp
    src/runtime/perf_log.cpp
    src/runtime/log.cpp
    src/runtime/speaker.cpp
    src/ops/safe_math.cpp
    src/ops/ggml_ops.cpp
    src/ops/lm_attn.cpp
    src/ops/rope.cpp
    src/ops/rvq.cpp
    src/ops/conv1d.cpp
    src/ops/convtr1d.cpp
    src/ops/pool1d.cpp
    src/ops/local_attn.cpp
    src/ops/audio_frontend.cpp
    src/blocks/s3_tokenizer.cpp
    src/blocks/campplus.cpp
    src/blocks/hift.cpp
    src/blocks/cfm.cpp
    src/blocks/conformer.cpp
    src/blocks/dit.cpp
    src/blocks/xvector_sincnet.cpp
    src/blocks/chatterbox_voice_encoder.cpp
    src/blocks/ecapa_tdnn.cpp
    src/batch/batch.cpp
    src/models/wavtokenizer.cpp
    src/models/dac.cpp
    src/models/mimi.cpp
    src/models/qwen3_tts_tokenizer.cpp
    src/models/chatterbox_s3t.cpp
    src/models/chatterbox_s3g.cpp
    src/models/cosyvoice.cpp
    src/models/soprano.cpp
    src/models/nemo_nano_codec.cpp
    src/models/neucodec.cpp
    src/models/xcodec2.cpp
    src/models/snac.cpp
    src/models/moss_audio.cpp
    src/models/xy_tokenizer.cpp
    src/models/bluemagpie_audiovae.cpp
    src/models/bluemagpie_blocks.cpp
    src/models/pocket_mimi.cpp
    src/lm/lm.cpp
    src/lm/bluemagpie_cfm.cpp
    src/lm/flow_lm.cpp
    src/lm/spm_unigram.cpp
    src/lm/parallel_heads_delay.cpp
    src/lm/residual_depth_ar.cpp
    src/lm/chatterbox_t3.cpp
    src/lm/cosyvoice_lm.cpp
    src/lm/speaker_chatterbox.cpp
    src/lm/speaker_qwen3_tts.cpp
)

# Build the codec library as both static (linked into the example binaries
# in this tree) and shared (loadable via Python ctypes for parity tests
# and external integrations).  Use OBJECT to compile sources once.
add_library(codec_objs OBJECT ${CODEC_SOURCES})
set_target_properties(codec_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Development hooks read from environment variables (CODEC_PERF_LOG,
# CODEC_OP_PROFILE, CODEC_FORCE_COMPOSITE_OPS, CODEC_NO_OP_OFFLOAD,
# CODEC_AUDIT_UPCAST).  Off for production builds: the code is not compiled in.
option(CODEC_DEBUG_TOOLS "Compile the env-var debug/profiling hooks into libcodec" OFF)
if (CODEC_DEBUG_TOOLS)
    target_compile_definitions(codec_objs PRIVATE CODEC_DEBUG_TOOLS)
endif()
# The audio-LM host layer (common/audio_lm.cpp) and its C API
# (include/codec_audio_lm.h) go into libcodec.so as well, so FFI callers
# (python/codec_cpp) reach it; the static codec_common below reuses the objects.
add_library(codec_audio_lm_objs OBJECT
    common/audio_lm.cpp
    common/audio_lm_c.cpp
    common/audio_lm_family.cpp
    common/audio_lm_sampler.cpp
    common/audio_lm_family_soprano.cpp
    common/audio_lm_family_token.cpp
    common/audio_lm_family_kitten.cpp
    common/audio_lm_family_codebook.cpp
    common/audio_lm_family_text_lane.cpp
    common/tts_segment.cpp
    common/kitten_text.cpp
)
set_target_properties(codec_audio_lm_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(codec_audio_lm_objs
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

add_library(codec        STATIC $<TARGET_OBJECTS:codec_objs>)
add_library(codec_shared SHARED $<TARGET_OBJECTS:codec_objs> $<TARGET_OBJECTS:codec_audio_lm_objs>)
set_target_properties(codec_shared PROPERTIES OUTPUT_NAME codec)

target_include_directories(codec_objs
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_include_directories(codec
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_include_directories(codec_shared
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

target_link_libraries(codec        PRIVATE ggml)
target_link_libraries(codec_shared PRIVATE ggml)

# Python wheel (pyproject.toml, scikit-build-core): install libcodec into the
# codec_cpp package so the bindings find it next to themselves.  Wheel builds
# link ggml statically (BUILD_SHARED_LIBS=OFF) so libcodec is self-contained.
if(SKBUILD)
    install(TARGETS codec_shared
        LIBRARY DESTINATION codec_cpp/lib COMPONENT python
        RUNTIME DESTINATION codec_cpp/lib COMPONENT python)
endif()

# Small reusable example IO helpers (WAV / NPY).  Declared before
# codec_common so the self-contained FlowLM runner (which loads ref-audio
# WAVs) can link it.  `examples` is on the public include path so consumers
# resolve "utils/wav_io.h".
add_library(codec-example-utils STATIC
    examples/utils/wav_io.cpp
    examples/utils/npy_io.cpp
)
target_include_directories(codec-example-utils
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# codec_common — generic audio-LM API mirroring llama.cpp's `common/`.
# See docs/codec_common_api.md.  Built as a static lib so example
# binaries can link it without pulling it into libcodec.so.  The
# self-contained (no-backbone) FlowLM reference synthesize lives here in
# tts_runner_flow.cpp so it stays available even in CODEC_TTS_BACKBONE=OFF
# builds; the full backbone-driven runner lives in codec_tts_runner.
add_library(codec_common STATIC
    $<TARGET_OBJECTS:codec_audio_lm_objs>
    common/tts_runner_flow.cpp
)
target_include_directories(codec_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(codec_common PUBLIC codec PRIVATE codec-example-utils)

add_executable(inspect-codec
    examples/inspect-codec.cpp
)

target_link_libraries(inspect-codec PRIVATE codec)

add_executable(codec-cli
    examples/codec-cli.cpp
)

target_link_libraries(codec-cli PRIVATE codec codec-example-utils)

add_executable(codec-lm-cli
    examples/codec-lm-cli.cpp
)

target_link_libraries(codec-lm-cli PRIVATE codec codec-example-utils)

add_executable(codec-batch-decode
    examples/codec-batch-decode.cpp
)

target_include_directories(codec-batch-decode PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(codec-batch-decode PRIVATE codec codec-example-utils ggml ggml-cpu)

option(CODEC_BUILD_TTS_CLI "Build the tts-cli tool" ON)
if(CODEC_BUILD_TTS_CLI)
    # The TTS path uses backbone archs added to the pinned llama.cpp submodule
    # (Barbet, CosyVoice v1); apply their patches (idempotent, in order, skips
    # when the submodule isn't checked out).
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupLlamaPatches.cmake)

    add_executable(tts-cli
        tools/tts-cli.cpp
    )
    target_link_libraries(tts-cli PRIVATE codec_common codec-example-utils)

    # Optional isolated llama.cpp backbone → enables the full `synthesize`
    # reference loop.  The loop itself lives in codec_tts_runner (common/
    # tts_runner.*), a standalone library that links the backbone — so other
    # hosts (tts-server) link it without tts-cli.
    option(CODEC_TTS_BACKBONE "Build the isolated llama.cpp backbone for tts-cli synthesize" ON)
    if(CODEC_TTS_BACKBONE)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupTtsBackbone.cmake)

        add_library(codec_tts_runner STATIC
            common/tts_runner.cpp
        )
        add_dependencies(codec_tts_runner ttsbackbone)
        target_include_directories(codec_tts_runner
            PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/common
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${TTS_BACKBONE_INCLUDE_DIRS})
        target_link_libraries(codec_tts_runner PUBLIC codec_common PRIVATE codec-example-utils "${TTS_BACKBONE_SO}")

        add_dependencies(tts-cli ttsbackbone)
        target_compile_definitions(tts-cli PRIVATE TTS_CLI_HAVE_BACKBONE=1)
        target_link_libraries(tts-cli PRIVATE codec_tts_runner "${TTS_BACKBONE_SO}")
        set_target_properties(tts-cli PROPERTIES
            BUILD_RPATH "${CMAKE_BINARY_DIR}/ttsbackbone;${CMAKE_BINARY_DIR}/ggml/src")

        add_executable(tts-constraint-test tests/tts_constraint_test.cpp)
        target_link_libraries(tts-constraint-test PRIVATE $<TARGET_PROPERTY:tts-cli,LINK_LIBRARIES>)
        target_include_directories(tts-constraint-test PRIVATE $<TARGET_PROPERTY:tts-cli,INCLUDE_DIRECTORIES>)

        add_executable(neutts-detect-test tests/neutts_detect_test.cpp)
        target_link_libraries(neutts-detect-test PRIVATE $<TARGET_PROPERTY:tts-cli,LINK_LIBRARIES>)
        target_include_directories(neutts-detect-test PRIVATE
            $<TARGET_PROPERTY:tts-cli,INCLUDE_DIRECTORIES>
            ${TTS_BACKBONE_INCLUDE_DIRS})

        # OpenAI-compatible /v1/audio/speech over the same runner.  HTTP + JSON
        # come from llama.cpp's vendored cpp-httplib / nlohmann (compiled here:
        # the copy inside libttsbackbone.so is hidden).  wav / pcm responses are
        # always available; each compressed format needs its encoder library.
        option(CODEC_TTS_SERVER "Build tts-server (OpenAI-compatible speech endpoint)" ON)
        if(CODEC_TTS_SERVER)
            set(_ll_vendor "${CMAKE_CURRENT_SOURCE_DIR}/common/third-party/llama.cpp/vendor")
            find_package(Threads REQUIRED)
            add_executable(tts-server
                tools/tts-server.cpp
                tools/audio_encode.cpp
                ${_ll_vendor}/cpp-httplib/httplib.cpp)
            target_include_directories(tts-server PRIVATE ${_ll_vendor} ${_ll_vendor}/cpp-httplib)
            target_link_libraries(tts-server PRIVATE codec_tts_runner "${TTS_BACKBONE_SO}" Threads::Threads)
            add_dependencies(tts-server ttsbackbone)
            set_target_properties(tts-server PROPERTIES
                BUILD_RPATH "${CMAKE_BINARY_DIR}/ttsbackbone;${CMAKE_BINARY_DIR}/ggml/src")

            # libraries (comma-separated);header — a format whose option is ON must be found.
            set(_tts_enc_MP3  "mp3lame"        "lame/lame.h")
            set(_tts_enc_OPUS "opus,ogg"       "opus/opus.h")
            set(_tts_enc_AAC  "fdk-aac"        "fdk-aac/aacenc_lib.h")
            set(_tts_enc_FLAC "FLAC"           "FLAC/stream_encoder.h")
            foreach(_fmt MP3 OPUS AAC FLAC)
                option(CODEC_TTS_SERVER_${_fmt} "tts-server: ${_fmt} responses" OFF)
                if(CODEC_TTS_SERVER_${_fmt})
                    list(GET _tts_enc_${_fmt} 0 _libs)
                    list(GET _tts_enc_${_fmt} 1 _hdr)
                    string(REPLACE "," ";" _libs "${_libs}")
                    find_path(_hdr_dir_${_fmt} NAMES ${_hdr})
                    if(NOT _hdr_dir_${_fmt})
                        message(FATAL_ERROR "CODEC_TTS_SERVER_${_fmt}=ON but ${_hdr} was not found")
                    endif()
                    target_include_directories(tts-server PRIVATE ${_hdr_dir_${_fmt}})
                    foreach(_lib ${_libs})
                        find_library(_lib_${_lib} NAMES ${_lib})
                        if(NOT _lib_${_lib})
                            message(FATAL_ERROR "CODEC_TTS_SERVER_${_fmt}=ON but lib${_lib} was not found")
                        endif()
                        target_link_libraries(tts-server PRIVATE ${_lib_${_lib}})
                    endforeach()
                    target_compile_definitions(tts-server PRIVATE CODEC_TTS_SERVER_${_fmt}=1)
                endif()
            endforeach()
        endif()

    endif()

    # Model-less unit test for the multi-codebook Type A observe_token accumulator.
    # Depends only on codec_common (no backbone, no GGUF required).
    add_executable(outetts-observe-test tests/outetts_observe_test.cpp)
    target_link_libraries(outetts-observe-test PRIVATE codec_common)
    target_include_directories(outetts-observe-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

    option(CODEC_TTS_ESPEAK "Link espeak-ng into tts-cli for NeuTTS phonemization" OFF)
    if(CODEC_TTS_ESPEAK)
        find_library(ESPEAK_NG_LIB NAMES espeak-ng libespeak-ng.so.1)
        if(NOT ESPEAK_NG_LIB)
            message(FATAL_ERROR "CODEC_TTS_ESPEAK=ON but espeak-ng library not found. "
                "Install libespeak-ng-dev (or libespeak-ng1 on systems without a .so symlink).")
        endif()
        find_path(ESPEAK_NG_INCLUDE_DIR NAMES espeak-ng/speak_lib.h
            HINTS /usr/include /usr/local/include)
        if(NOT ESPEAK_NG_INCLUDE_DIR)
            message(FATAL_ERROR "CODEC_TTS_ESPEAK=ON but espeak-ng headers not found "
                "(espeak-ng/speak_lib.h). Install libespeak-ng-dev.")
        endif()
        target_compile_definitions(tts-cli PRIVATE CODEC_TTS_ESPEAK=1)
        target_include_directories(tts-cli PRIVATE ${ESPEAK_NG_INCLUDE_DIR})
        target_link_libraries(tts-cli PRIVATE ${ESPEAK_NG_LIB})
    endif()
endif()

# Install: libcodec + headers, the TTS tools and the llama.cpp backbone they
# load (ggml installs its own libraries).  The tools find their libraries in
# ../lib next to bin.  Wheel builds (SKBUILD) install libcodec into the Python
# package instead, above.
if(NOT SKBUILD)
    include(GNUInstallDirs)
    file(RELATIVE_PATH _codec_bin_to_lib "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")

    install(TARGETS codec_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES include/codec.h include/codec_lm.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    foreach(_tool tts-cli tts-server)
        if(TARGET ${_tool})
            set_target_properties(${_tool} PROPERTIES INSTALL_RPATH "${_codec_origin}/${_codec_bin_to_lib}")
            install(TARGETS ${_tool} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif()
    endforeach()
    if(TARGET ttsbackbone)
        install(FILES "${TTS_BACKBONE_SO}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
endif()
