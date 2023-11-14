# Copyright (c) 2017-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

set(SOURCES src/nv_utils.c
            src/nvlog.c
            src/stat_log.c
            src/aerial_event_code.c
            src/exit_handler.cpp
            )

if(NVIPC_FMTLOG_ENABLE)
    list(APPEND SOURCES src/nvlog.cpp)
endif()

######## libnvlog.so ###################
add_library(nvlog SHARED ${SOURCES})
set_property(TARGET nvlog PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvlog PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(nvlog INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NVIPC_FMTLOG_ENABLE)
    target_include_directories(nvlog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../external/fmtlog_flat)
    target_include_directories(nvlog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmtlog_flat)
endif()
target_include_directories(nvlog PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(nvlog PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(nvlog ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} rt -pthread yaml)
if(NVIPC_FMTLOG_ENABLE)
    target_link_libraries(nvlog fmtlog-shared fmt::fmt)
endif()

######## libnvlog_static.a #############
add_library(nvlog_static STATIC ${SOURCES})
set_property(TARGET nvlog_static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvlog_static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(nvlog_static INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NVIPC_FMTLOG_ENABLE)
    target_include_directories(nvlog_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../external/fmtlog_flat)
    target_include_directories(nvlog_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmtlog_flat)
endif()
target_include_directories(nvlog_static PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(nvlog_static PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(nvlog_static ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} rt -pthread yaml)
if(NVIPC_FMTLOG_ENABLE)
    target_link_libraries(nvlog_static ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} rt -pthread yaml fmtlog-static fmt::fmt)
endif()
target_compile_options(nvlog_static PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

if(NVIPC_FMTLOG_ENABLE)
    add_subdirectory(test)
endif()
