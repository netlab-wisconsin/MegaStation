/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


//#define CUPHY_DEBUG 1

#include "ldpc2_desc.cuh"
#include "ldpc2_c2v.cuh"
#include "ldpc2_app_address_fp_desc.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_min_sum_update_half_0.cuh"
#include "ldpc2_reg_index_fp_desc_dyn_small.hpp"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "ldpc2_c2v_cache_register.cuh"

using namespace ldpc2;

#define USE_APP_ADDR_FP_DP 1

namespace
{
    // Single set of values for all kernels in this module, for now...
    const int MAX_THREADS_PER_CTA = 32;
    const int MIN_CTA_PER_SM      = 1;

    //------------------------------------------------------------------
    // Sign manager policies (for compressed C2V row processors)
    typedef ldpc2::sign_store_policy_split_dst<__half, ldpc2::split_sign_update_fp,      false> sign_dst_fp_t;
    typedef ldpc2::sign_store_policy_split_dst<__half, ldpc2::split_sign_update_bit_ops, false> sign_dst_bit_t;
    typedef ldpc2::sign_store_policy_split_src<__half, ldpc2::split_sign_update_fp,      false> sign_src_fp_t;
    typedef ldpc2::sign_store_policy_split_src<__half, ldpc2::split_sign_update_bit_ops, false> sign_src_bit_t;
    
    // Sign updates with FP unit A couple of microseconds faster on V100...
    typedef sign_dst_fp_t                                                                       sign_mgr_t;

    // APP address calculation
    // Using floating point instruction APP address calculation
    // sequence
#if USE_APP_ADDR_FP_DP
    template <int BG> using app_loc_t = app_loc_address_fp_dp_desc<__half, BG>;
#else
    template <int BG> using app_loc_t = app_loc_address_fp_desc<__half, BG>;
#endif

    typedef C2V_storage_t<__half, 2>                                                            c2v_storage_t;
    //------------------------------------------------------------------
    // Kernel configuration structure, with typedefs for kernel execution
    // BG_: base graph (1 or 2)
    // TKernelParams: Class/struct used for kernel parameters
    // TLLR: Source LLR data type (__half or float)
    // TLoader: LLR loader template struct (e.g. llr_loader_batch)
    template <int                                                                BG_,
              class                                                              TKernelParams,
              typename                                                           TLLR,
              template <typename, int, template<typename, typename> class> class TLoader>
    struct ldpc2_reg_index_fp_desc_dyn_kernel_config
    {
        static constexpr int BG              = BG_;
        static constexpr int MIN_PARITY_ROWS = 4;
        static constexpr int MAX_PARITY_ROWS = ldpc2::max_parity_nodes<BG>::value;

        typedef TKernelParams                   kernel_params_t;
        
        // We use the "checked" variants of app_loader and app_writer, since we may
        // have extra threads that are not participating in a valid codeword.
        typedef cC2V_index<__half,
                           BG,
                           sign_mgr_t,
                           min_sum_update_half_0,
                           app_loader_checked,
                           app_writer_checked> C2V_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V message cache (register memory here)
        typedef ldpc2::c2v_cache_register<BG,
                                          MAX_PARITY_ROWS,
                                          C2V_t,
                                          c2v_storage_t,
                                          kernel_params_t>                      c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        typedef TLoader<TLLR, 4, llr_op_clamp>                                  llr_loader_t;
        // Data type in APP shared memory buffer (__half or __half2)
        typedef typename llr_loader_t::app_buf_t                                app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG_>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG_>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };
} // namespace


////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_desc_dyn_small()
// Base graph 1 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      ldpc2::LDPC_kernel_params,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(params, blockIdx.x, params.num_codewords);
    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, params, mconfig);

#if 0
    print_array_sync("LLR:", reinterpret_cast<__half*>(smem), 960);
#endif
    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(params) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES)           :
                                 -1;
    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(params,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_desc_dyn_small()
// Base graph 2 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      ldpc2::LDPC_kernel_params,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(params, blockIdx.x, params.num_codewords);

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, params, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(params) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES)           :
                                 -1;
    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(params,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_desc_dyn_small_tb()
// Base graph 1 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES)           :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_desc_dyn_small_tb()
// Base graph 2 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET       = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                  (mconfig.thread_codeword_index * LLR_STRIDE_BYTES)           :
                                  -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32()
// Base graph 1 kernel, "legacy" tensor interface, fp32->fp16 on input
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      ldpc2::LDPC_kernel_params,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(params, blockIdx.x, params.num_codewords);

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, params, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(params) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES)           :
                                 -1;
    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(params,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32()
// Base graph 2 kernel, "legacy" tensor interface, fp32->fp16 on input
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      ldpc2::LDPC_kernel_params,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(params, blockIdx.x, params.num_codewords);

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, params, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(params) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES)           :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(params,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32()
// Base graph 1 kernel, transport block interface, fp32->fp16 on input
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET       = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                  (mconfig.thread_codeword_index * LLR_STRIDE_BYTES)           :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32()
// Base graph 2 kernel, transport block interface, fp32->fp16 on input
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES)           :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_small::decode()
cuphyStatus_t reg_index_fp_desc_dyn_small::decode(ldpc::decoder&                     dec,
                                                  LDPC_output_t&                     tDst,
                                                  const_tensor_pair&                 tLLR,
                                                  const cuphyLDPCDecodeConfigDesc_t& config,
                                                  cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::reg_index_fp_desc_dyn_small::decode()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    const int       THREADS_PER_CTA = MAX_THREADS_PER_CTA;
    const int       CW_PER_CTA = THREADS_PER_CTA / config.Z;
    //------------------------------------------------------------------
    dim3 grdDim((NUM_CW + (CW_PER_CTA - 1)) / CW_PER_CTA);
    dim3 blkDim(THREADS_PER_CTA);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory, which is the same
    // for both fp16 and fp32 (after conversion).
    const uint32_t SHMEM_SIZE_PER_CW = round_up_to_next(shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
                                                                              params.Z,             // lifting size
                                                                              sizeof(__half)),      // element size
                                                        static_cast<unsigned int>(sizeof(ldpc_traits<__half>::llr_sts_t)));
    const uint32_t SHMEM_SIZE = CW_PER_CTA * SHMEM_SIZE_PER_CW;
                                                                   
    //printf("grdDim = (%u), blkDim = (%u), CW_PER_CTA = %i, SHMEM_SIZE = %u, SHMEM_SIZE_PER_CW = %u, NUM_VAR_NODES = %i, func = %i, size = %u, test = %i\n",
    //       grdDim.x,
    //       blkDim.x,
    //       CW_PER_CTA,
    //       SHMEM_SIZE,
    //       SHMEM_SIZE_PER_CW,
    //       params.num_var_nodes,
    //       shmem_llr_buffer_size(params.num_var_nodes, params.Z, sizeof(__half)),
    //       static_cast<unsigned int>(sizeof(ldpc_traits<__half>::llr_sts_t)),
    //       ((68 * 2 + 15) / 16) * 16);
    
    if(llrType == CUPHY_R_16F)
    {
        switch(config.BG)
        {
        case 1:
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(params.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_desc_dyn_small, blkDim, SHMEM_SIZE);
                
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_desc_dyn_small<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;

            }
            break;
        case 2:
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_desc_dyn_small, blkDim, SHMEM_SIZE);
                
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_desc_dyn_small<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    }
    else if(llrType == CUPHY_R_32F)
    {
        switch(config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(params.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    }
    
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_small::decode_tb()
cuphyStatus_t reg_index_fp_desc_dyn_small::decode_tb(ldpc::decoder&               dec,
                                                     const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                     cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::reg_index_fp_desc_dyn_small::decode_tb()\n");    
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    //------------------------------------------------------------------
    const int NUM_CW          = ldpc::decoder::get_total_num_codewords(decodeDesc);
    const int THREADS_PER_CTA = MAX_THREADS_PER_CTA;
    const int CW_PER_CTA      = THREADS_PER_CTA / decodeDesc.config.Z;
    //------------------------------------------------------------------
    dim3 grdDim((NUM_CW + (CW_PER_CTA - 1)) / CW_PER_CTA);
    dim3 blkDim(THREADS_PER_CTA);

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory, which is the same
    // for both fp16 and fp32 (after conversion).
    const int      NUM_VAR_NODES = decodeDesc.config.num_parity_nodes +
                                   ((1 == decodeDesc.config.BG) ? max_info_nodes<1>::value : max_info_nodes<2>::value);
    const uint32_t SHMEM_SIZE    = CW_PER_CTA * round_up_to_next(shmem_llr_buffer_size(NUM_VAR_NODES,       // num shared memory nodes
                                                                                       decodeDesc.config.Z, // lifting size
                                                                                       sizeof(__half)),     // element size
                                                                 static_cast<unsigned int>(sizeof(ldpc2::ldpc_traits<__half>::llr_sts_t)));

    //printf("grdDim = (%u), blkDim = (%u), CW_PER_CTA = %i, SHMEM_SIZE = %u, NUM_VAR_NODES = %i\n", grdDim.x, blkDim.x, CW_PER_CTA, SHMEM_SIZE, NUM_VAR_NODES);
    
    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_desc_dyn_small_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_desc_dyn_small_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_desc_dyn_small_tb, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_desc_dyn_small_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    }
    else if(decodeDesc.config.llr_type == CUPHY_R_32F)
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    }
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_small::get_workspace_size()
std::pair<bool, size_t> reg_index_fp_desc_dyn_small::get_workspace_size(const ldpc::decoder&               dec,
                                                                        const cuphyLDPCDecodeConfigDesc_t& config,
                                                                        int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_small::reg_index_fp_desc_dyn_small()
reg_index_fp_desc_dyn_small::reg_index_fp_desc_dyn_small(ldpc::decoder& desc)
{
    // Maximum shared memory configuration
    // Assume: THREADS_PER_CTA = 32
    //
    // (Changing THREADS_PER_CTA will change the analysis below. For
    // example, with Z = 12, we can do 2 codewords in 32 threads. But if
    // we do 96 threads, we can do 8 codewords.)
    //
    // We will launch THREADS_PER_CTA, and process as many codewords
    // in the main loop as we can.
    //
    // The amount of shared memory required is Z * (mb + Kb) * sizeof(T).
    // However, shared memory for each codeword needs to be aligned to
    // the STS size used by the LLR loader (currently uint4, or 8
    // elements). For the maximum case and 32 threads per warp, it
    // turns out that the amount of shared memory with padding is equal to
    // the amount of shared memory that would be required for Z = 32
    // without padding.
    // MAX_CODEWORDS_PER_CTA = (THREADS_PER_CTA / MIN_Z)  = (32 / 2) = 16
    // MAX_LLR_PER_CODEWORD: 68 * Z (BG1)
    //                       52 * Z (BG2)
    // MAX_SHMEM_PER_CTA = round_up(MAX_LLR_PER_CODEWORD, sizeof(sts_type)) * MAX_CODEWORDS_PER_CTA
    
    // BG1: 32 * sizeof(half) * 68 = 4352 bytes
    // BG2: 32 * sizeof(half) * 52 = 3328 bytes
    const uint32_t MAX_VAR_NODES_BG1 = ldpc2::max_variable_nodes<1>::value;
    const uint32_t MAX_VAR_NODES_BG2 = ldpc2::max_variable_nodes<2>::value;
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory that could be used
    // by a kernel
    const int MAX_BG1_SHMEM_SIZE = static_cast<int>(shmem_llr_buffer_size(MAX_VAR_NODES_BG1, // num shared memory nodes
                                                                          32,                // lifting size
                                                                          sizeof(__half)));  // element size
    const int MAX_BG2_SHMEM_SIZE = static_cast<int>(shmem_llr_buffer_size(MAX_VAR_NODES_BG2, // num shared memory nodes
                                                                          32,                // lifting size
                                                                          sizeof(__half)));  // element size

    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 4> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_desc_dyn_small,    MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_desc_dyn_small,    MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_desc_dyn_small_tb, MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_desc_dyn_small_tb, MAX_BG2_SHMEM_SIZE)
    };
    for(func_attr_t f_a : func_attrs)
    {
        cudaError_t e = cudaFuncSetAttribute(f_a.first,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             f_a.second);
        if(cudaSuccess != e)
        {
            throw cuphy_i::cuda_exception(e);
        }
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_index_fp_desc_dyn_small);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_index_fp_desc_dyn_small);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_index_fp_desc_dyn_small_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_index_fp_desc_dyn_small_tb);
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_small::get_launch_config()
cuphyStatus_t reg_index_fp_desc_dyn_small::get_launch_config(const ldpc::decoder&           dec,
                                                             cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int NUM_CW           = ldpc::decoder::get_total_num_codewords(launchConfig.decode_desc);
    const int THREADS_PER_CTA  = MAX_THREADS_PER_CTA;
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    const int MAX_PARITY_NODES = (1 == BG)                  ?
                                 max_parity_nodes<1>::value :
                                 max_parity_nodes<2>::value;
    const int NUM_VAR_NODES    = ldpc::decoder::get_num_variable_nodes(BG,
                                                                       NUM_PARITY_NODES);
    const int CW_PER_CTA       = THREADS_PER_CTA / Z;

    const uint32_t SHMEM_SIZE  = CW_PER_CTA * round_up_to_next(shmem_llr_buffer_size(NUM_VAR_NODES,   // num shared memory nodes
                                                                                     Z,               // lifting size
                                                                                     sizeof(__half)), // element size
                                                               static_cast<unsigned int>(sizeof(ldpc2::ldpc_traits<__half>::llr_sts_t)));

    //------------------------------------------------------------------
    // Validate input arguments
    if((Z < 2)                              ||
       (Z > CUPHY_LDPC_MAX_LIFTING_SIZE)    ||
       (NUM_PARITY_NODES < 4)               ||
       (NUM_PARITY_NODES > MAX_PARITY_NODES))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    //------------------------------------------------------------------
    // Set up launch geometry and the kernel function (driver)
    #if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = THREADS_PER_CTA;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = div_round_up(NUM_CW, CW_PER_CTA);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;
    launchConfig.kernel_node_params_driver.sharedMemBytes = SHMEM_SIZE;

    cudaFunction_t deviceFunction;
    cudaError_t    e = (BG == 1) ?  cudaGetFuncBySymbol(&deviceFunction, (void*)ldpc2_BG1_reg_index_fp_desc_dyn_small_tb): 
                                    cudaGetFuncBySymbol(&deviceFunction, (void*)ldpc2_BG2_reg_index_fp_desc_dyn_small_tb);
    if (e != cudaSuccess) 
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    launchConfig.kernel_node_params_driver.func = static_cast<CUfunction>(deviceFunction);
    #endif
    //------------------------------------------------------------------
    // Set kernel arguments:
    // arg 0: decode descriptor
    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    // arg 1: base graph descriptor
    if(1 == BG)
    {
        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    else
    {
        const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
