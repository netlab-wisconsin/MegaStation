/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC_2_ATTIC_HPP_INCLUDED_)
#define LDPC_2_ATTIC_HPP_INCLUDED_

////////////////////////////////////////////////////////////////////////
// ldpc
// Exported functions called by cuPHY LDPC code
namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address()
cuphyStatus_t decode_ldpc2_reg_address(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       cuphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       cuphyLDPCDiagnostic_t* diag,
                                       cudaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_address_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);
    
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index()
cuphyStatus_t decode_ldpc2_reg_index(decoder&               dec,
                                     LDPC_output_t&         tDst,
                                     const_tensor_pair&     tLLR,
                                     const LDPC_config&     config,
                                     float                  normalization,
                                     cuphyLDPCResults_t*    results,
                                     void*                  workspace,
                                     cuphyLDPCDiagnostic_t* diag,
                                     cudaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_workspace_size(const decoder&     dec,
                                                              const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp()
cuphyStatus_t decode_ldpc2_reg_index_fp(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_fp_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2()
cuphyStatus_t decode_ldpc2_reg_index_fp_x2(decoder&               dec,
                                           LDPC_output_t&         tDst,
                                           const_tensor_pair&     tLLR,
                                           const LDPC_config&     config,
                                           float                  normalization,
                                           cuphyLDPCResults_t*    results,
                                           void*                  workspace,
                                           cuphyLDPCDiagnostic_t* diag,
                                           cudaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_fp_x2_workspace_size(const decoder&     dec,
                                                                    const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address()
cuphyStatus_t decode_ldpc2_global_address(decoder&               dec,
                                          LDPC_output_t&         tDst,
                                          const_tensor_pair&     tLLR,
                                          const LDPC_config&     config,
                                          float                  normalization,
                                          cuphyLDPCResults_t*    results,
                                          void*                  workspace,
                                          cuphyLDPCDiagnostic_t* diag,
                                          cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_address_workspace_size(const decoder&     dec,
                                                                   const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index()
cuphyStatus_t decode_ldpc2_global_index(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_index_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index()
cuphyStatus_t decode_ldpc2_shared_index(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_index_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index()
cuphyStatus_t decode_ldpc2_shared_cluster_index(decoder&               dec,
                                                LDPC_output_t&         tDst,
                                                const_tensor_pair&     tLLR,
                                                const LDPC_config&     config,
                                                float                  normalization,
                                                cuphyLDPCResults_t*    results,
                                                void*                  workspace,
                                                cuphyLDPCDiagnostic_t* diag,
                                                cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_cluster_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2()
cuphyStatus_t decode_ldpc2_shared_index_fp_x2(decoder&               dec,
                                              LDPC_output_t&         tDst,
                                              const_tensor_pair&     tLLR,
                                              const LDPC_config&     config,
                                              float                  normalization,
                                              cuphyLDPCResults_t*    results,
                                              void*                  workspace,
                                              cuphyLDPCDiagnostic_t* diag,
                                              cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_index_fp_x2_workspace_size(const decoder&     dec,
                                                                       const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_dynamic_index()
cuphyStatus_t decode_ldpc2_shared_dynamic_index(decoder&               dec,
                                                LDPC_output_t&         tDst,
                                                const_tensor_pair&     tLLR,
                                                const LDPC_config&     config,
                                                float                  normalization,
                                                cuphyLDPCResults_t*    results,
                                                void*                  workspace,
                                                cuphyLDPCDiagnostic_t* diag,
                                                cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_dynamic_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_dynamic_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_index()
cuphyStatus_t decode_ldpc2_split_index(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       cuphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       cuphyLDPCDiagnostic_t* diag,
                                       cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_index_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_dynamic_index()
cuphyStatus_t decode_ldpc2_split_dynamic_index(decoder&               dec,
                                               LDPC_output_t&         tDst,
                                               const_tensor_pair&     tLLR,
                                               const LDPC_config&     config,
                                               float                  normalization,
                                               cuphyLDPCResults_t*    results,
                                               void*                  workspace,
                                               cuphyLDPCDiagnostic_t* diag,
                                               cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_dynamic_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_dynamic_index_workspace_size(const decoder&     dec,
                                                                        const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_index()
cuphyStatus_t decode_ldpc2_split_cluster_index(decoder&               dec,
                                               LDPC_output_t&         tDst,
                                               const_tensor_pair&     tLLR,
                                               const LDPC_config&     config,
                                               float                  normalization,
                                               cuphyLDPCResults_t*    results,
                                               void*                  workspace,
                                               cuphyLDPCDiagnostic_t* diag,
                                               cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_cluster_index_workspace_size(const decoder&     dec,
                                                                        const LDPC_config& cfg);


////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index()
cuphyStatus_t decode_ldpc2_reg_index(decoder&               dec,
                                     LDPC_output_t&         tDst,
                                     const_tensor_pair&     tLLR,
                                     const LDPC_config&     config,
                                     float                  normalization,
                                     cuphyLDPCResults_t*    results,
                                     void*                  workspace,
                                     cuphyLDPCDiagnostic_t* diag,
                                     cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_workspace_size(const decoder&     dec,
                                                              const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address()
cuphyStatus_t decode_ldpc2_reg_address(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       cuphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       cuphyLDPCDiagnostic_t* diag,
                                       cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_address_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);

} // namespace ldpc

#endif // !defined(LDPC_2_ATTIC_HPP_INCLUDED_)
