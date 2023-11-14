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

########################################################################
# test_ldpc_internal_app_addr
set(SRCS test_ldpc_internal_app_addr.cu
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_desc.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_desc_half2.cpp)
add_executable(test_ldpc_internal_app_addr ${SRCS})
target_include_directories(test_ldpc_internal_app_addr PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_app_addr cuphy)
target_link_libraries(test_ldpc_internal_app_addr gtest gtest_main)
target_compile_options(test_ldpc_internal_app_addr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_app_addr COMMAND test_ldpc_internal_app_addr)
#add_test(NAME test_ldpc_internal COMMAND test_ldpc_internal WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/error_correction)

########################################################################
# test_ldpc_internal_rc
add_executable(test_ldpc_internal_rc test_ldpc_internal_rc.cu)
target_include_directories(test_ldpc_internal_rc PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_rc cuphy)
target_link_libraries(test_ldpc_internal_rc gtest gtest_main)
target_compile_options(test_ldpc_internal_rc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_rc COMMAND test_ldpc_internal_rc)

########################################################################
# test_ldpc_internal_loader
add_executable(test_ldpc_internal_loader test_ldpc_internal_loader.cu)
target_include_directories(test_ldpc_internal_loader PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_loader cuphy)
target_link_libraries(test_ldpc_internal_loader gtest gtest_main)
target_compile_options(test_ldpc_internal_loader PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_loader COMMAND test_ldpc_internal_loader)

#                        --keep
#                       --cudart static



