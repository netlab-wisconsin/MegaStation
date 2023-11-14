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

#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_app_address_dp_desc.cuh"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_split_index_fp_x2_desc_dyn.hpp"
#include "ldpc2_c2v_cache_split.cuh"

#define LDPC_DECODE_USE_TB_SCAN 1

using namespace ldpc2;

namespace
{
    // Single set of values for all kernels in this module, for now...
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;

    //------------------------------------------------------------------
    // Storing compressed check to variable (cC2V) data in registers may
    // not be possible for all code rates. Furthermore, squeezing a
    // larger number of parity node data into registers may actually
    // decrease performance at high code rates.
    template <int BG> struct num_reg_parity;
    template <> struct num_reg_parity<1> { static constexpr int value = 33; };
    template <> struct num_reg_parity<2> { static constexpr int value = 42; };
    template <int BG> struct max_num_parity;
    template <> struct max_num_parity<1> { static constexpr int value = 46; };
    template <> struct max_num_parity<2> { static constexpr int value = 42; };
    
    //------------------------------------------------------------------
    // Sign manager for compressed C2V row processor
    typedef sign_mgr_pair_src<false> sign_mgr_t;

    //------------------------------------------------------------------
    // APP address calculation
    // Using floating point with dot product instruction sequence for
    // this decoder algorithm. Note that the base graph descriptor
    // argument to the kernel needs to be the "adjusted" descriptor
    // structure.
    template <int BG> using app_loc_t = app_loc_address_fp_dp_desc<__half2, BG>;
    // slightly slower on sm86
    //template <int BG> using app_loc_t = app_loc_address_dp_desc<__half2, BG>;

    //------------------------------------------------------------------
    // Template alias for a half2 row context, templated ONLY on the
    // underlying storage type. (For this decoder, we will use different
    // row contexts, and thus slightly different row processors,  for
    // the "high degree" core rows.)
    template <class TStorage> using row_context_t = cC2V_row_context<__half2,
                                                                     sign_mgr_t,
                                                                     unused,
                                                                     TStorage>;
    //------------------------------------------------------------------
    // Template alias for a half2 compressed C2V row processors,
    // templated ONLY on the row context used. This will be used by the
    // row mappers, which will instantiate a cC2V_row_proc_t template
    // instance for the different row context storage types.
    template <class TRowContext> using cC2V_row_proc_t = cC2V_row_proc<__half2,
                                                                       TRowContext>;

    //------------------------------------------------------------------
    // Kernel configuration structure, with typedefs for kernel execution
    //
    // Better perf at very high code rates when the MAX_PARITY_NODES
    // is smaller, but for now we'll prefer to get 2X codewords for
    // as many parity nodes as possible. (Try 32 vs. 28 to see the perf
    // difference.)
    // TODO: small, med, large parity count kernels?
    template <int   BG_,                 // base graph (1 or 2)
              class TKernelParams>       // struct with kernel params
    struct ldpc2_split_index_fp_x2_desc_dyn_kernel_config
    {
        static constexpr int BG                  = BG_;
        static constexpr int MIN_PARITY_ROWS     = 4;
        static constexpr int NUM_REG_PARITY_ROWS = num_reg_parity<BG>::value;
        static constexpr int MAX_PARITY_ROWS     = max_num_parity<BG>::value;
        
        typedef TKernelParams                           kernel_params_t;

        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        // cC2V_storage_row_map_t
        // The C2V_row_proc template requires a row map template with template
        // arguments BG (int), CHECK_IDX (int), TStorage (per-row storage
        // structure.
        template <int   BG,
                  int   CHECK_IDX,
                  class TC2VStorage> using cC2V_row_map_t = context_storage_row_map<BG,
                                                                                    CHECK_IDX,
                                                                                    TC2VStorage,
                                                                                    __half2,
                                                                                    row_context_t,
                                                                                    cC2V_row_proc_t>;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V row dispatch type: uses the row map to determine which
        // C2V processor to call for each row.
        typedef C2V_row_proc<__half2,
                             BG,
                             cC2V_row_map_t,
                             app_loader,
                             app_writer> C2V_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V message cache (split between register and shared memory
        // here). Two storage types are provided: one for the "core"
        // parity rows, and one for the "non-core" rows. For BG1, the
        // core rows are high-degree (19) and the rest are low-degree
        // (10 or less). For BG2, all rows are low-degree (10 or less).
        typedef ldpc2::c2v_cache_split<BG,
                                       NUM_REG_PARITY_ROWS,
                                       C2V_t,
                                       typename core_storage_x2<BG>::type, // core cC2V storage
                                       cC2V_storage_x2_low_degree,         // non-core cC2V storage
                                       kernel_params_t> c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        typedef ldpc2::llr_loader_variable_batch<__half2, 4, llr_op_clamp> llr_loader_t;
        // Data type in APP shared memory buffer (__half or __half2)
        typedef llr_loader_t::app_buf_t                                    app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG_>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };
    //------------------------------------------------------------------
    // get_app_c2v_shmem()
    // Returns the number of bytes required for APP and C2V memory for
    // this kernel.
    template <int BG>
    CUDA_BOTH
    int get_app_c2v_shmem(int num_parity_nodes, int Z)
    {
        const     int32_t NUM_VAR_NODES = ldpc2::max_info_nodes<BG>::value + num_parity_nodes;
        constexpr int32_t NUM_REG_NODES = num_reg_parity<BG>::value;
        const     int32_t APP_SIZE      = static_cast<int32_t>(shmem_llr_buffer_size(NUM_VAR_NODES,     // num shared memory nodes
                                                                                     Z,                 // lifting size
                                                                                     sizeof(__half2))); // element size
        // The first 'NUM_REG_NODES' of C2V data will reside in registers.
        // The remainder will be in shared memory.
        const int32_t C2V_SIZE = (num_parity_nodes > NUM_REG_NODES)                                          ?
                                 (num_parity_nodes - NUM_REG_NODES) * Z * sizeof(cC2V_storage_x2_low_degree) :
                                 0;
        // We need to pad the APP portion of shared memory to make sure
        // that C2V storage is aligned with the C2V type.
        int shmem_size = round_up_to_next(APP_SIZE, static_cast<int>(alignof(cC2V_storage_x2_low_degree))) +
                                          C2V_SIZE;
        return shmem_size;
    }
    //------------------------------------------------------------------
    // get_shmem_required()
    // Calculates the sum of the APP and C2V data storage.
    int get_shmem_required(int BG,
                           int num_parity_nodes,
                           int Z)
    {
        int shmem_size = (1 == BG) ? get_app_c2v_shmem<1>(num_parity_nodes, Z)
                                   : get_app_c2v_shmem<2>(num_parity_nodes, Z);
#if LDPC_DECODE_USE_TB_SCAN
        // When using a scan algorithm to determine the codeword for a CTA,
        // extra shared memory for the token is required.
        shmem_size = round_up_to_next(shmem_size, static_cast<int>(alignof(tb_token))) +
                     sizeof(tb_token);
#endif
        return shmem_size;
    }
#if LDPC_DECODE_USE_TB_SCAN
    //------------------------------------------------------------------
    // get_token_addr()
    // Returns the address of the tb_token value used to store information
    // about the specific codeword being processed by a CTA when the
    // transport block interface is used. The token is assumed to reside
    // immediately after the APP and C2V memory.
    template <int BG>
    __device__
    tb_token* get_token_addr(int num_parity_nodes, int Z, char* smem)
    {
        return reinterpret_cast<tb_token*>(smem + get_app_c2v_shmem<BG>(num_parity_nodes, Z));
    }
    template <int BG>
    __device__
    tb_token* get_token_addr(const cuphyLDPCDecodeDesc_t& decodeDesc,
                             char* smem)
    {
        return get_token_addr<BG>(decodeDesc.config.num_parity_nodes,
                                  decodeDesc.config.Z,
                                  smem);
    }
#endif // if LDPC_DECODE_USE_TB_SCAN
} // namespace

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_split_index_fp_x2_desc_dyn()
// Kernel for base graph 1 (legacy tensor interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_split_index_fp_x2_desc_dyn(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_fp_x2_desc_dyn_kernel_config<1,                                          // BG
                                                           ldpc2::LDPC_kernel_params> kernel_config_t; // params struct

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
        //thread0_dump_app(reinterpret_cast<__half2*>(smem), params.Z_var);
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_split_index_fp_x2_desc_dyn()
// Kernel for base graph 1 (legacy tensor interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_split_index_fp_x2_desc_dyn(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_fp_x2_desc_dyn_kernel_config<2,                                          // BG
                                                           ldpc2::LDPC_kernel_params> kernel_config_t; // params struct

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_split_index_fp_x2_desc_dyn_tb()
// Kernel for base graph 1 (transport block interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_split_index_fp_x2_desc_dyn_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_fp_x2_desc_dyn_kernel_config<1,                                            // BG
                                                           cuphyLDPCDecodeConfigDesc_t> kernel_config_t; // params struct
#if !LDPC_DECODE_USE_TB_SCAN
    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1>(decodeDesc, smem));
#endif

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
#else
    ldpc_dec_output_variable_loop(decodeDesc,
                                  tok,
                                  reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
#endif
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_split_index_fp_x2_desc_dyn_tb()
// Kernel for base graph 2 (transport block interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_split_index_fp_x2_desc_dyn_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_fp_x2_desc_dyn_kernel_config<2,                                            // BG
                                                           cuphyLDPCDecodeConfigDesc_t> kernel_config_t; // params struct

#if !LDPC_DECODE_USE_TB_SCAN
    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<2>(decodeDesc, smem));
#endif
    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
#else
    ldpc_dec_output_variable(decodeDesc,
                             tok,
                             reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
#endif
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// split_index_fp_x2_desc_dyn::decode()
cuphyStatus_t split_index_fp_x2_desc_dyn::decode(ldpc::decoder&                     dec,
                                                 LDPC_output_t&                     tDst,
                                                 const_tensor_pair&                 tLLR,
                                                 const cuphyLDPCDecodeConfigDesc_t& config,
                                                 cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_split_index_fp_x2_desc_dyn()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    dim3 grdDim(div_round_up(NUM_CW, 2));
    // We need to be mindful of the blockDim not being a multiple of 32.
    // The hard decision output writes 32-bit words. We may need to
    // revisit the output function to allow us to truncate the threads
    // that write to the next lowest multiple of 32, but that  may also
    // mean that we need to then have the output function LOOP.
    //dim3 blkDim(((config.Z + 31) / 32) * 32);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = get_shmem_required(config.BG,
                                                   config.num_parity_nodes,
                                                   config.Z);
    
    if(llrType == CUPHY_R_16F)
    {
        switch(config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(params.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_fp_x2_desc_dyn, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_split_index_fp_x2_desc_dyn<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_split_index_fp_x2_desc_dyn, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_split_index_fp_x2_desc_dyn<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
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
// split_index_fp_x2_desc_dyn::decode_tb()
cuphyStatus_t split_index_fp_x2_desc_dyn::decode_tb(ldpc::decoder&               dec,
                                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                    cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::split_index_fp_x2_desc_dyn::decode_tb()\n");
    
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        // We need to be mindful of the blockDim not being a multiple of 32.
        // The hard decision output writes 32-bit words. We may need to
        // revisit the output function to allow us to truncate the threads
        // that write to the next lowest multiple of 32, but that  may also
        // mean that we need to then have the output function LOOP.
        //dim3 blkDim(((config.Z + 31) / 32) * 32);
        dim3 blkDim(decodeDesc.config.Z);
        
        //------------------------------------------------------------------
        // Launch a CTA for each codeword pair. Note that the number of CTAs
        // may be more than the total number of codewords divided by 2 -
        // there may be transport blocks with odd numbers of codewords.
        dim3 grdDim(ldpc::decoder::get_total_num_codeword_pairs(decodeDesc));

        //------------------------------------------------------------------
        // Determine the dynamic amount of shared memory
        const uint32_t SHMEM_SIZE = get_shmem_required(decodeDesc.config.BG,
                                                       decodeDesc.config.num_parity_nodes,
                                                       decodeDesc.config.Z);
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_fp_x2_desc_dyn_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
               ldpc2_BG1_split_index_fp_x2_desc_dyn_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_split_index_fp_x2_desc_dyn_tb, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_split_index_fp_x2_desc_dyn_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
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
// split_index_fp_x2_desc_dyn::get_workspace_size()
std::pair<bool, size_t> split_index_fp_x2_desc_dyn::get_workspace_size(const ldpc::decoder&               dec,
                                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                                       int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// split_index_fp_x2_desc_dyn::split_index_fp_x2_desc_dyn()
split_index_fp_x2_desc_dyn::split_index_fp_x2_desc_dyn(ldpc::decoder& dec)
{
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory that could be used
    // by a kernel
    const int MAX_BG1_SHMEM_SIZE = static_cast<int>(get_shmem_required(1,                             // BG
                                                                       max_num_parity<1>::value,      // max parity nodes
                                                                       CUPHY_LDPC_MAX_LIFTING_SIZE)); // lifting size
    const int MAX_BG2_SHMEM_SIZE = static_cast<int>(get_shmem_required(2,                             // BG
                                                                       max_num_parity<2>::value,      // max parity nodes
                                                                       CUPHY_LDPC_MAX_LIFTING_SIZE)); // lifting size
    //------------------------------------------------------------------
    // Maximum shared memory supported by the device
    const int MAX_SHMEM = dec.max_shmem_per_block_optin();

    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 4> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_split_index_fp_x2_desc_dyn,    std::min(MAX_BG1_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_split_index_fp_x2_desc_dyn,    std::min(MAX_BG2_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_fp_x2_desc_dyn_tb, std::min(MAX_BG1_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_split_index_fp_x2_desc_dyn_tb, std::min(MAX_BG2_SHMEM_SIZE, MAX_SHMEM))
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_split_index_fp_x2_desc_dyn);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_split_index_fp_x2_desc_dyn);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_split_index_fp_x2_desc_dyn_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_split_index_fp_x2_desc_dyn_tb);
}

////////////////////////////////////////////////////////////////////////
// split_index_fp_x2_desc_dyn::can_decode_config()
bool split_index_fp_x2_desc_dyn::can_decode_config(const ldpc::decoder&               dec,
                                                   const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    // Compare shared memory requirements to device maximum, as well as
    // the maximum that the kernel was compiled for.
    
    // Maximum number of parity nodes, as limited by compilation, to
    // limit register usage.
    const uint32_t MAX_NUM_PARITY = (1 == cfg.BG) ? max_num_parity<1>::value : max_num_parity<2>::value;
    // Calculate required shared memory
    const uint32_t SHMEM_BYTES    = get_shmem_required(cfg.BG,
                                                       cfg.num_parity_nodes,
                                                       cfg.Z);
    return (cfg.num_parity_nodes <= MAX_NUM_PARITY) &&
           (SHMEM_BYTES <= dec.max_shmem_per_block_optin());
}

////////////////////////////////////////////////////////////////////////
// split_index_fp_x2_desc_dyn::get_launch_config()
cuphyStatus_t split_index_fp_x2_desc_dyn::get_launch_config(const ldpc::decoder&           dec,
                                                            cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    const int MAX_PARITY_NODES = (1 == BG)                  ?
                                 max_parity_nodes<1>::value :
                                 max_parity_nodes<2>::value;
    const int NUM_VAR_NODES    = ldpc::decoder::get_num_variable_nodes(BG,
                                                                       NUM_PARITY_NODES);
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
    launchConfig.kernel_node_params_driver.blockDimX = Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codeword_pairs(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;

    const uint32_t SHMEM_SIZE = get_shmem_required(launchConfig.decode_desc.config.BG,
                                                   launchConfig.decode_desc.config.num_parity_nodes,
                                                   launchConfig.decode_desc.config.Z);
    launchConfig.kernel_node_params_driver.sharedMemBytes = SHMEM_SIZE;

    cudaFunction_t deviceFunction;
    cudaError_t    e = (BG == 1) ? cudaGetFuncBySymbol(&deviceFunction, (void*)ldpc2_BG1_split_index_fp_x2_desc_dyn_tb) : 
                                   cudaGetFuncBySymbol(&deviceFunction, (void*)ldpc2_BG2_split_index_fp_x2_desc_dyn_tb);
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
        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    else
    {
        const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
