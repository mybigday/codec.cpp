# ttsbackbone.def: the llama_* C API and the common_sampler C++ API, as the
# version script / exported-symbols list do on ELF / Mach-O.
#   cmake -DOBJS=<file listing .obj paths> -DDEF=<output .def> -P windows-def.cmake
# CMake's own WINDOWS_EXPORT_ALL_SYMBOLS generator lists every external symbol
# (`__create_def`: native COFF reader on Windows, llvm-nm elsewhere); keep ours.
execute_process(COMMAND ${CMAKE_COMMAND} -E __create_def "${DEF}.all" "${OBJS}"
                RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "listing the ttsbackbone exports failed")
endif()
file(STRINGS "${DEF}.all" _lines)
set(_out "EXPORTS\n")
foreach(_l IN LISTS _lines)
    # C names are bare (llama_x); MSVC-mangled C++ names start with ?name@.
    if(_l MATCHES "^[ \t]*(llama_|\\?common_sampler)")
        string(APPEND _out "${_l}\n")
    endif()
endforeach()
file(WRITE "${DEF}" "${_out}")
