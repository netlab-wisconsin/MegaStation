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

# Generation of compile time LUTs for CRC

set (LUTS_DIR "${CMAKE_CURRENT_BINARY_DIR}/LUTS")

file(MAKE_DIRECTORY ${LUTS_DIR})

add_executable(genLUT crc/gen_crc_LUTs.cpp descrambling/descrambling.hpp)

target_include_directories(genLUT PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

add_custom_command( OUTPUT
            ${LUTS_DIR}/G_CRC_24_A_P_LUT.h
            ${LUTS_DIR}/G_CRC_24_B_P_LUT.h
            ${LUTS_DIR}/G_CRC_16_P_LUT.h
            ${LUTS_DIR}/GOLD_1_SEQ_LUT.h
            ${LUTS_DIR}/GOLD_2_32_P_LUT.h
            DEPENDS crc/crc.hpp crc/gen_crc_LUTs.cpp descrambling/descrambling.hpp cuphy.h
            COMMAND genLUT ${LUTS_DIR}
            COMMENT "Generating LUTs for CRC and Descrambling")

add_custom_target(CRCLUTGEN DEPENDS
            ${LUTS_DIR}/G_CRC_24_A_P_LUT.h
            ${LUTS_DIR}/G_CRC_24_B_P_LUT.h
            ${LUTS_DIR}/G_CRC_16_P_LUT.h
            ${LUTS_DIR}/GOLD_1_SEQ_LUT.h
            ${LUTS_DIR}/GOLD_2_32_P_LUT.h
            )

# Generation of compile time LUTs for PUCCH receiver Format 1

add_executable(genPucchLUT pucch_receiver/gen_pucch_receiver_LUTs.cpp)
target_include_directories(genPucchLUT PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

add_custom_command( OUTPUT
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TOCC_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TIME_SHIFT_SEQ_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_PAPR_SEQ_VALUES_LUT.h
            DEPENDS pucch_receiver/gen_pucch_receiver_LUTs.cpp
            COMMAND genPucchLUT ${LUTS_DIR}
            COMMENT "Generating LUTs for PUCCH receiver Format 1")

add_custom_target(PUCCHLUTGEN DEPENDS
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TOCC_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TIME_SHIFT_SEQ_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_PAPR_SEQ_VALUES_LUT.h
            )

if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND CUPHY_NVCC_FLAGS -lineinfo)
endif()
#message("lineinfo flag value: ${CUPHY_NVCC_FLAGS}")

set(CUPHY_SOURCES cuphy.h
                  cuphy.cpp
                  cuphy_internal.h
                  cuphy_context.hpp
                  cuphy_context.cpp
                  tensor_desc.cpp
                  device.hpp
                  device.cpp
                  type_convert.hpp
                  convert_tensor.cuh
                  convert_tensor.cu
                  empty_kernels.hpp
                  empty_kernels.cuh
                  empty_kernels.cu
                  cuphy_kernel_util.cuh
                  bfc/bfc.cu
                  channel_est/channel_est.cu
                  ch_est/ch_est.cu
                  pusch_noise_intf_est/pusch_noise_intf_est.cu
                  cfo_ta_est/cfo_ta_est.hpp
                  cfo_ta_est/cfo_ta_est.cu
                  channel_eq/channel_eq.cu
                  error_correction/nrLDPC.cuh
                  error_correction/nrLDPC_flat.cuh
                  error_correction/ldpc.cuh
                  error_correction/ldpc2_kernel.cuh
                  error_correction/ldpc2_sign.cuh
                  error_correction/ldpc2_sign_split.cuh
                  error_correction/ldpc2_llr_loader.cuh
                  error_correction/ldpc2_min_sum_update_half_0.cuh
                  error_correction/ldpc2_min_sum_update_half_1.cuh
                  error_correction/ldpc2_c2v.cuh
                  error_correction/ldpc2_box_plus.cuh
                  error_correction/ldpc.hpp
                  error_correction/ldpc_load_store.cuh
                  error_correction/ldpc2_dec_output.cuh
                  error_correction/ldpc.cpp
                  error_correction/ldpc2.hpp
                  error_correction/ldpc2_bg_desc.hpp
                  error_correction/ldpc2_bg_desc.cpp
                  error_correction/ldpc2_bg_desc_half2.cpp
                  error_correction/ldpc2_c2v_cache_register.cuh
                  error_correction/ldpc2_c2v_cache_global.cuh
                  error_correction/ldpc2_c2v_cache_shared.cuh
                  error_correction/ldpc2_c2v_cache_split.cuh
                  error_correction/ldpc2_schedule_fixed.cuh
                  error_correction/ldpc2_schedule_dynamic.cuh
                  error_correction/ldpc2_schedule_dynamic_desc.cuh
                  error_correction/ldpc2_schedule_cluster.cuh
                  error_correction/ldpc2_app_address.cuh
                  error_correction/ldpc2_app_address_fp.cuh
                  error_correction/ldpc2_app_address_fp_desc.cuh
                  error_correction/ldpc2_app_address_fp_dp_desc.cuh
                  error_correction/ldpc2_reg_index_fp_desc_dyn.hpp
                  error_correction/ldpc2_reg_index_fp_desc_dyn.cu
                  error_correction/ldpc2_reg_index_fp_desc_dyn_small.hpp
                  error_correction/ldpc2_reg_index_fp_desc_dyn_small.cu
                  error_correction/ldpc2_reg_index_fp_x2_desc_dyn.hpp
                  error_correction/ldpc2_reg_index_fp_x2_desc_dyn.cu
                  error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep.hpp
                  error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep.cu
                  error_correction/ldpc2_reg_index_fp_dp_desc_dyn_row_dep.hpp
                  error_correction/ldpc2_reg_index_fp_dp_desc_dyn_row_dep.cu
                  error_correction/ldpc2_reg_index_fp_desc_dyn_sm80.hpp
                  error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep_sm80.hpp
                  error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep_sm86.hpp
                  error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep_sm90.hpp
                  error_correction/ldpc2_shm_index_fp_desc_dyn.hpp
                  error_correction/ldpc2_shm_index_fp_desc_dyn.cu
                  error_correction/ldpc2_split_index_fp_x2_desc_dyn.hpp
                  error_correction/ldpc2_split_index_fp_x2_desc_dyn.cu
                  error_correction/ldpc2_split_index_fp_x2_desc_dyn_sm86.hpp
                  error_correction/ldpc2_split_index_fp_x2_desc_dyn_sm86.cu
                  error_correction/ldpc2_split_index_fp_x2_desc_dyn_sm90.hpp
                  error_correction/ldpc2_c2v_x2.cuh
                  error_correction/ldpc2.cuh
                  error_correction/ldpc2.cpp
                  error_correction/ldpc_encode.cu
                  error_correction/nrLDPC_templates.cuh
                  error_correction/ldpc_decoder_cubin/ldpc_decoder_cubin.s
                  rate_matching/rate_matching.cu
                  dl_rate_matching/dl_rate_matching.cuh
                  dl_rate_matching/dl_rate_matching.cu
                  crc/crc.cu
                  descrambling/descrambling.cu
                  pusch_rssi/pusch_rssi.cu
                  pucch_receiver/pucch_receiver.cu
                  pucch_receiver/rm_decoder.cu
                  pucch_receiver/rm_decoder.hpp
                  simplex_decoder/simplex_decoder.cu
                  simplex_decoder/simplex_decoder.hpp
                  prach_receiver/prach_receiver.cu
                  modulation_mapper/modulation_mapper.cu
                  pdcch/embed_pdcch_tf_signal.cu
                  pdsch_dmrs/pdsch_dmrs.cu
                  ss/ss.cu
                  polar_encoder/polar_encoder.cu
                  comp_cwTreeTypes/comp_cwTreeTypes.cu
                  comp_cwTreeTypes/comp_cwTreeTypes.hpp
                  srs_ch_est/srs_ch_est.cu
                  soft_demapper/soft_demapper.hpp
                  soft_demapper/soft_demapper.cu
                  soft_demapper/soft_demapper.cuh
                  soft_demapper/soft_demapper_tables.cuh
                  soft_demapper/soft_demapper_tables.h
                  pucch_F0_receiver/pucch_F0_receiver.cu
                  pucch_F0_receiver/pucch_F0_receiver.hpp
                  pucch_F1_receiver/pucch_F1_receiver.cu
                  pucch_F1_receiver/pucch_F1_receiver.hpp
                  pucch_F2_front_end/pucch_F2_front_end.cu
                  pucch_F2_front_end/pucch_F2_front_end.hpp
                  pucch_F3_front_end/pucch_F3_front_end.cu
                  pucch_F3_front_end/pucch_F3_front_end.hpp
                  pucch_F3_front_end/pucch_F3_csi2Ctrl.cu
                  pucch_F3_front_end/pucch_F3_csi2Ctrl.hpp
                  pucch_F3_front_end/pucch_F3_segLLRs.cu
                  pucch_F3_front_end/pucch_F3_segLLRs.hpp
                  pucch_F234_uci_seg/pucch_F234_uci_seg.cu
                  pucch_F234_uci_seg/pucch_F234_uci_seg.hpp
                  csirs/csirs.cuh
                  csirs/csirs_tf_signal.cu
                  rng/rng.hpp
                  rng/rng.cu
                  variant.hpp
                  variant.cpp
                  tensor_fill.hpp
                  tensor_fill.cu
                  tensor_tile.hpp
                  tensor_tile.cu
                  tensor_elementwise.hpp
                  tensor_elementwise.cu
                  tensor_reduction.hpp
                  tensor_reduction.cu
                  polar_seg_deRm_deItl/polar_seg_deRm_deItl.cu
                  polar_seg_deRm_deItl/polar_seg_deRm_deItl.hpp
                  uci_on_pusch/uciOnPusch_segLLRs1.cu
                  uci_on_pusch/uciOnPusch_segLLRs1.hpp
                  polar_decoder/polar_decoder.cu
                  polar_decoder/polar_decoder.hpp
                  uci_on_pusch/uciOnPusch_segLLRs2.cu
                  uci_on_pusch/uciOnPusch_segLLRs2.hpp
                  uci_on_pusch/uciOnPusch_segLLRs0.cu
                  uci_on_pusch/uciOnPusch_segLLRs0.hpp
                  uci_on_pusch/uciOnPusch_csi2Ctrl.cu
                  uci_on_pusch/uciOnPusch_csi2Ctrl.hpp
                  srs_chEst0/srs_chEst0.cu
                  srs_chEst0/srs_chEst0.hpp
                  cuphy_pti.cpp
                  cuphy_pti.cu
                  )

if (CUPHY_INTERNAL_CMAKE_FILE)
    include(cuphy_lib_internal.cmake OPTIONAL)
else (CUPHY_INTERNAL_CMAKE_FILE)
    set(CUPHY_SOURCES ${CUPHY_SOURCES} error_correction/ldpc2_reg_index_fp_desc_dyn_sm80.cu
                                       error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep_sm80.cu
                                       error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep_sm86.cu
                                       error_correction/ldpc2_reg_index_fp_desc_dyn_row_dep_sm90.cu
                                       error_correction/ldpc2_split_index_fp_x2_desc_dyn_sm90.cu)
endif()

add_library(cuphy ${CUPHY_SOURCES})
set_property(TARGET cuphy PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(cuphy PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

# CMake LINK_FLAGS_<CONFIG> was superseded by LINK_OPTIONS around version
# v3.13, but we still may be running on older versions.
set_property(TARGET cuphy APPEND PROPERTY LINK_FLAGS_RELEASE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cuphy.map -s")

add_dependencies(cuphy CRCLUTGEN)
add_dependencies(cuphy PUCCHLUTGEN)
target_include_directories(cuphy PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(cuphy PRIVATE crc
                                         ../../examples/common/
                                         ../../external/mathdx/22.02/include/
                                         ../compression_decompression
       	                                 bfc
                                         channel_est
                                         channel_eq
                                         ch_est
                                         csirs
                                         pusch_noise_intf_est
                                         Gold_sequence
                                         pucch_F0_receiver
                                         pucch_F1_receiver
                                         pucch_F2_front_end
                                         pucch_F3_front_end
                                         pucch_F234_uci_seg
                                         cfo_ta_est
                                         descrambling
                                         dl_rate_matching
                                         rate_matching
                                         modulation_mapper
                                         "${CMAKE_CURRENT_BINARY_DIR}/LUTS"
                                         error_correction
                                         error_correction/ldpc_decoder_cubin
                                         pusch_rssi
                                         pucch_receiver
                                         simplex_decoder
                                         prach_receiver
                                         pdsch_dmrs
                                         polar_encoder
                                         pdcch
                                         srs_ch_est
                                         ss
                                         soft_demapper
                                         rng
                                         comp_cwTreeTypes
                                         polar_seg_deRm_deItl
                                         uci_on_pusch
                                         polar_decoder
                                         srs_chEst0
                                         ../../external/fmtlog_flat
                                         ../../nvlog
                                         )
# Use target_compile_options() to set CUDA-specific flags:
#target_compile_options(<my_target> PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:my_cuda_option>)
#target_compile_options(cuphy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUPHY_OPTS}>)

#set(CUPHY_OPTS --ptxas-options=-v --resource-usage --cudart static ${CUPHY_GENCODE_STRING} -lineinfo)
#set(CUPHY_OPTS --ptxas-options=-v --resource-usage --cudart static "SHELL:-gencode arch=compute_70,code=sm_70" "SHELL:-gencode arch=compute_75,code=sm_75" -lineinfo)
#message(${CUPHY_OPTS})

target_compile_options(cuphy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       ${CUPHY_PTXAS_OPTIONS}
                       --cudart static
                       ${CUPHY_NVCC_FLAGS}
                       --threads 0
                       --expt-relaxed-constexpr
                       >)

# Explicitly add a dependency for the .cubin files that are included by
# ldpc_decoder_cubin.s
SET(LDPC_DECODER_CUBIN_LIST ${CMAKE_CURRENT_SOURCE_DIR}/error_correction/ldpc_decoder_cubin/ldpc2_reg_index_fp_desc_dyn_row_dep_sm80_internal.sm_80.cubin
                            ${CMAKE_CURRENT_SOURCE_DIR}/error_correction/ldpc_decoder_cubin/ldpc2_reg_index_fp_desc_dyn_row_dep_sm86_internal.sm_86.cubin
                            ${CMAKE_CURRENT_SOURCE_DIR}/error_correction/ldpc_decoder_cubin/ldpc2_reg_index_fp_desc_dyn_sm80_internal.sm_80.cubin
                            ${CMAKE_CURRENT_SOURCE_DIR}/error_correction/ldpc_decoder_cubin/ldpc2_split_index_fp_x2_desc_dyn_sm86_internal.sm_86.cubin)

set_source_files_properties(error_correction/ldpc_decoder_cubin/ldpc_decoder_cubin.s PROPERTIES
                            OBJECT_DEPENDS "${LDPC_DECODER_CUBIN_LIST}")

# Add the line below to the list above to keep intermediate nvcc files
#                      --keep
# Add the line below to the list above to compile GPU kernelw with debugging info:
#                      -g -G
#set_source_files_properties(error_correction/ldpc2_split_cluster_96KB_BG1_Z384.cu PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_split_dynamic_96KB_BG1_Z384.cu PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_reg_index_BG1_Z384.cu PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_reg_index_fp_BG1_Z384.cu PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_reg_index_fp_x2_BG1_Z384.cu PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_reg_index_fp_desc_dyn.cu PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_reg_index_fp_dp_desc_dyn_row_dep.cu PROPERTIES COMPILE_FLAGS --keep)
set_source_files_properties(error_correction/ldpc_ms_cta_shmem_layered.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(pucch_receiver/pucch_receiver.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(prach_receiver/prach_receiver.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(ss/ss.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(srs_ch_est/srs_ch_est.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(cfo_est/cfo_est.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(channel_eq/channel_eq.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(ch_est/ch_est.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(pusch_noise_intf_est/pusch_noise_intf_est.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(pusch_rssi/pusch_rssi.cu PROPERTIES COMPILE_FLAGS --use_fast_math)

install(TARGETS cuphy DESTINATION lib)
install(FILES cuphy.h utils.cuh DESTINATION include)

# Link with cuda library for driver API
target_link_libraries(cuphy cufft cuda ${NVTX_LIB} nvlog)
