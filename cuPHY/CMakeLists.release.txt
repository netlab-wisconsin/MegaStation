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
# Using CUDA support inn CMake 3.8+
# https://devblogs.nvidia.com/building-cuda-applications/cmake/
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# ----------------------------------------------------------------------
# Build options
option(USE_MATLAB "Build MEX libraries to allow use from MATLAB" OFF)

# ----------------------------------------------------------------------
# Set the version number for this project
set(cuPHY_VERSION_MAJOR 0)
set(cuPHY_VERSION_MINOR 8)

# ----------------------------------------------------------------------
# Enable testing support
enable_testing()

# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------
# CUDA Options
#     Generate a string to specify CUDA target architectures
set(CUDA_GENCODE_STRING "" )

#     List of target architectures
set(CUDA_GENCODE_ARCH_LIST "70,72,75" CACHE STRING "List of target CUDA architectures")
message("CUDA_GENCODE_ARCH_LIST is ${CUDA_GENCODE_ARCH_LIST}")

#     Convert comma-delimited string to CMake list (semicolon-delimited)
string(REPLACE "," ";" CUDA_GENCODE_ARCH_LIST_ ${CUDA_GENCODE_ARCH_LIST})
foreach(CUDA_ARCH_ ${CUDA_GENCODE_ARCH_LIST_})
  string(APPEND CUDA_GENCODE_STRING "-gencode arch=compute_${CUDA_ARCH_},code=sm_${CUDA_ARCH_} ")
  #message("CUDA_ARCH_=${CUDA_ARCH_}")
endforeach(CUDA_ARCH_)

set(CMAKE_CUDA_FLAGS " --cudart static -lineinfo ${CUDA_GENCODE_STRING}" CACHE STRING "CUDA Global Flags")

#     CUDA flags for device debugging with cuda-gdb
# string(APPEND CMAKE_CUDA_FLAGS " --cudart static -lineinfo -g -G")
set(CMAKE_CUDA_STANDARD 14)
project(cuPHY LANGUAGES C CXX CUDA)

message("CMAKE_CUDA_FLAGS is ${CMAKE_CUDA_FLAGS}")

# ----------------------------------------------------------------------
# Additional packages
set(THREADS_PREFER_PTHREAD_FLAG OFF)
find_package(Threads REQUIRED)
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)
if (USE_MATLAB)
    find_package(Matlab REQUIRED)
endif (USE_MATLAB)

# ----------------------------------------------------------------------
# Subdirectories
add_subdirectory(examples)

if (USE_MATLAB)
    add_subdirectory(util/matlab/mex)
endif (USE_MATLAB)

