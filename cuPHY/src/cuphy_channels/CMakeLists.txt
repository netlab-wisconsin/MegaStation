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

include_directories(common)
add_library(cuphy_channels STATIC pusch_rx.cpp pdsch_tx.cpp pdcch_tx.cpp pucch_rx.cpp csirs_tx.cpp prach_rx.cpp ssb_tx.cpp bfw_tx.cpp srs_rx.cu)
set_property(TARGET cuphy_channels PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cuphy_channels PRIVATE ../../examples/common)
target_include_directories(cuphy_channels PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                                             $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                             ${HDF5_INCLUDE_DIRS})

target_compile_options(cuphy_channels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       ${CUPHY_NVCC_FLAGS}
                       >)

target_compile_options(cuphy_channels PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})                       

target_link_libraries(cuphy_channels PUBLIC cuphy_hdf5 cuphy ${HDF5_C_LIBRARIES})

install(TARGETS cuphy_channels DESTINATION lib)
install(FILES pusch_rx.hpp pdsch_tx.hpp pdcch_tx.hpp pucch_rx.hpp csirs_tx.hpp prach_rx.hpp ssb_tx.hpp bfw_tx.hpp srs_rx.hpp DESTINATION include)
