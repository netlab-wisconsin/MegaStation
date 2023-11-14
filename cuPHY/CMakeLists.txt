# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# Minimum CMake version required.
# Using CUDA support in CMake 3.8+
# Using SHELL escape support for argument de-duplication added in CMake 3.12+
# https://devblogs.nvidia.com/building-cuda-applications/cmake/
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Get system ARCH
    execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
    string(STRIP ${ARCH} ARCH)
    if ("${ARCH}" STREQUAL "aarch64")
        set(CMAKE_TOOLCHAIN_FILE ../cmake/toolchains/bf3)
    else ()
        set(CMAKE_TOOLCHAIN_FILE ../cmake/toolchains/x86-64)
    endif()
endif()
message(STATUS "Using toolchain ${CMAKE_TOOLCHAIN_FILE}")

if (NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()


# ----------------------------------------------------------------------
# Build options
option(USE_MATLAB "Build MEX libraries to allow use from MATLAB" OFF)
option(BUILD_DOCS "Generate Doxygen documentation" OFF)
option(CUPHY_BUILD_PTXAS_VERBOSE "Enable verbose mode for the PTX assembler to show local memory usage" OFF)
option(CUPHY_HDF5_VERSION_CHECK "Force HDF5 library version checking, necessary for interoperability with binary cuPHY packages" ON)
option(ENABLE_NVTX  "Enable use of NVTX" OFF)
option(BUILD_SHARED_LIBS "Build cuPHY as a shared library" ON)
option(ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR  "Enable PUSCH per-UE pre-EQ noise-interference power reporting" ON)
OPTION(ENABLE_CUPHY_PTI_TRACING "Enable kernel tracing using cuPHY PTI" OFF)

if(ENABLE_CUPHY_PTI_TRACING)
    add_definitions(-DCUPHY_PTI_ENABLE_TRACING)
endif()
# ----------------------------------------------------------------------
# Optionally include internal cuPHY CMake commands
include(cuphy_internal.cmake OPTIONAL RESULT_VARIABLE CUPHY_INTERNAL_CMAKE_FILE)

# ----------------------------------------------------------------------
# Set the version number for this project
set(cuPHY_VERSION_MAJOR 0)
set(cuPHY_VERSION_MINOR 8)

# ----------------------------------------------------------------------
# Enable testing support
enable_testing()

# ----------------------------------------------------------------------
# Global C++ options
set(CUDAToolkit_ROOT /usr/local/cuda)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_VERSION 12.2)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
project(cuPHY LANGUAGES C CXX CUDA ASM)

# ----------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND CUPHY_NVCC_FLAGS -lineinfo)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND CUPHY_NVCC_FLAGS -Werror all-warnings)
endif()

# ----------------------------------------------------------------------
# cuPHY CUDA Options
#     Generate a string to specify CUDA target architectures
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
  message(ERROR "CUDA versions lower than 11.0 not supported")
else()
  set(CUPHY_GENCODE_ARCH_LIST "70" CACHE STRING "List of target CUDA architectures for cuPHY")
endif()

message(STATUS "CUPHY_GENCODE_ARCH_LIST is ${CUPHY_GENCODE_ARCH_LIST}")

# Convert comma-delimited string to CMake list (semicolon-delimited). CMake 3.18
# and above will take care of the architecture format
string(REPLACE "," ";" CUPHY_GENCODE_ARCH_LIST_ "${CUPHY_GENCODE_ARCH_LIST}")
set(CMAKE_CUDA_ARCHITECTURES ${CUPHY_GENCODE_ARCH_LIST_})


# Old way of setting global CMAKE_CUDA_FLAGS variable...
#set(CMAKE_CUDA_FLAGS " --cudart static -lineinfo ${CUDA_GENCODE_STRING}" CACHE STRING "CUDA Global Flags")
#set(CMAKE_CUDA_FLAGS " --cudart static -lineinfo --keep ${CUDA_GENCODE_STRING}" CACHE STRING "CUDA Global Flags")
# CUDA flags for device debugging with cuda-gdb
# string(APPEND CMAKE_CUDA_FLAGS " --cudart static -lineinfo -g -G")

if(CUPHY_BUILD_PTXAS_VERBOSE)
    #set(CUPHY_PTXAS_OPTIONS "--ptxas-options=-v")
    set(CUPHY_PTXAS_OPTIONS ${CUPHY_PTXAS_OPTIONS} "--resource-usage")
endif(CUPHY_BUILD_PTXAS_VERBOSE)

# ----------------------------------------------------------------------
# Additional packages
set(THREADS_PREFER_PTHREAD_FLAG OFF)
find_package(Threads REQUIRED)
# https://cmake.org/cmake/help/latest/module/FindHDF5.html
# set(HDF5_USE_STATIC_LIBRARIES ON)
if (CUPHY_HDF5_VERSION_CHECK)
    find_package(HDF5 1.10 REQUIRED COMPONENTS C)
else (CUPHY_HDF5_VERSION_CHECK)
    find_package(HDF5 REQUIRED COMPONENTS C)
endif (CUPHY_HDF5_VERSION_CHECK)
if (USE_MATLAB)
    find_package(Matlab REQUIRED)
endif (USE_MATLAB)

# Link with nvToolsExt
if (ENABLE_NVTX)
    set(NVTX_LIB nvToolsExt)
    add_definitions(-DUSE_NVTX=1)
endif (ENABLE_NVTX)

if (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=1)
else (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=0)
endif (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)

# ----------------------------------------------------------------------
# Generation of documentation
if (BUILD_DOCS)
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        set(DOXYGEN_IN  ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        message(STATUS "Configuring ${DOXYGEN_OUT}")
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        add_custom_target(docs_doxygen ALL
          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          COMMENT "Generating API documentation (doxygen)"
          VERBATIM)
    else (DOXYGEN_FOUND)
        message(FATAL_ERROR "Documentation generation requested, but Doxygen package not found")
    endif (DOXYGEN_FOUND)
endif (BUILD_DOCS)

project(cuPHY)
# ----------------------------------------------------------------------
# Subdirectories
if (NOT TARGET nvlog)
    message(STATUS "including nvlog")
    if(NOT NVIPC_FMTLOG_ENABLE)
        message(STATUS "Setting NVIPC_FMTLOG_ENABLE ON")
        set(NVIPC_FMTLOG_ENABLE ON)
        add_definitions(-DNVIPC_FMTLOG_ENABLE)
    endif()
    add_subdirectory(nvlog)
endif()
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(examples)
#add_subdirectory(test)


if (USE_MATLAB)
    add_subdirectory(util/matlab/mex)
endif (USE_MATLAB)

# ----------------------------------------------------------------------
# Installation
install(DIRECTORY examples/common DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/bfc DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/ch_est DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/channel_eq DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pusch_rx_multi_pipe DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/error_correction DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/ldpc_encode DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pdsch_tx DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pucch_receiver DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/prach_receiver_multi_cell DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/prach_test DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/ss DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/polar_encoder DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/srs_ch_est DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(FILES CMakeLists.release.txt DESTINATION . RENAME CMakeLists.txt)
if (BUILD_DOCS)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs DESTINATION . FILES_MATCHING PATTERN "*")
endif (BUILD_DOCS)

