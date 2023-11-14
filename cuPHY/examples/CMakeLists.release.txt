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

# Add cuPHY headers to our include path
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add cuPHY libraries to our link path
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_library(cuphy_cpp_header_only INTERFACE)
target_include_directories(cuphy_cpp_header_only INTERFACE common/)

include_directories(common)
add_subdirectory(bfc)
add_subdirectory(ch_est)
add_subdirectory(cfo_ta_est)
add_subdirectory(channel_eq)
add_subdirectory(pusch_rx_multi_pipe)
add_subdirectory(multi_slot)
add_subdirectory(channels)
add_subdirectory(psch_rx_tx)
add_subdirectory(error_correction)
add_subdirectory(pdsch_tx)
add_subdirectory(pucch_receiver)
add_subdirectory(pucch_F0_receiver)
add_subdirectory(pucch_F1_receiver)
add_subdirectory(pucch_F2_front_end)
add_subdirectory(pucch_F3_front_end)
add_subdirectory(pucch_F3_segLLRs)
add_subdirectory(pucch_F234_uci_seg)
add_subdirectory(simplex)
add_subdirectory(prach_receiver_multi_cell)
add_subdirectory(prach_test)
add_subdirectory(ss)
add_subdirectory(polar_encoder)
add_subdirectory(srs_ch_est)
add_subdirectory(sym_demod)
add_subdirectory(pusch_crc)
add_subdirectory(pucch_rx_pipeline)
add_subdirectory(csi_rs)
add_subdirectory(polar_seg_deRm_deItl)
add_subdirectory(pusch_rateMatch)
add_subdirectory(ReedMuller_decoder)
add_subdirectory(uciOnPusch_deSeg_part2)
add_subdirectory(uciOnPusch_deSeg_part0)
add_subdirectory(uciOnPusch_csi2_ctrl)
add_subdirectory(srs_chEst0)
add_subdirectory(srs_rx_pipeline)
add_subdirectory(fading_chan)

