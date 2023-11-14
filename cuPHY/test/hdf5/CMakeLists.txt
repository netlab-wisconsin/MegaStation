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

add_executable(test_hdf5 test_hdf5.cpp dummy.cu)
target_include_directories(test_hdf5 PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_hdf5 cuphy)
target_link_libraries(test_hdf5 cuphy_hdf5)
target_link_libraries(test_hdf5 gtest gtest_main)
add_test(NAME test_hdf5 COMMAND test_hdf5 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/hdf5)

add_executable(test_hdf5_host test_hdf5_host.cpp)
target_include_directories(test_hdf5_host PRIVATE ${PROJECT_SOURCE_DIR}/examples/common ${HDF5_INCLUDE_DIRS})
target_link_libraries(test_hdf5_host ${HDF5_C_LIBRARIES})
target_link_libraries(test_hdf5_host gtest gtest_main)
add_test(NAME test_hdf5_host COMMAND test_hdf5_host WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/hdf5)
