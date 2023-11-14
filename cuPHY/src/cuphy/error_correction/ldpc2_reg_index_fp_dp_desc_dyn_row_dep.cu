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

#include "ldpc2.cuh"
#include "ldpc2_c2v.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_box_plus.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_min_sum_update_half_0.cuh"
#include "ldpc2_reg_index_fp_dp_desc_dyn_row_dep.hpp"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "ldpc2_c2v_cache_register.cuh"

using namespace ldpc2;

namespace
{
    // Single set of values for all kernels in this module, for now...
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;
    
    //------------------------------------------------------------------
    // Sign manager policies (for compressed C2V row processors)
    typedef ldpc2::sign_store_policy_split_dst<__half, ldpc2::split_sign_update_fp,      false> sign_dst_fp_t;
    typedef ldpc2::sign_store_policy_split_dst<__half, ldpc2::split_sign_update_bit_ops, false> sign_dst_bit_t;
    typedef ldpc2::sign_store_policy_split_src<__half, ldpc2::split_sign_update_fp,      false> sign_src_fp_t;
    typedef ldpc2::sign_store_policy_split_src<__half, ldpc2::split_sign_update_bit_ops, false> sign_src_bit_t;

    // APP address calculation
    // Using floating point with dot product instruction sequence for
    // this decoder algorithm. Note that the base graph descriptor
    // argument to the kernel needs to be the "adjusted" descriptor
    // structure.
    template <int BG> using app_loc_t = app_loc_address_fp_dp_desc<__half, BG>;;

    //------------------------------------------------------------------
    // Alias template for compressed C2V row processors, with a template
    // parameter for the C2V row storage. The sign processor and min
    // sum updater have been chosen to be the "fastest" for some
    // architecture and lifting size combinations, but...
    // TODO: develop autotuner
    template <class TStorage> using cC2V_row_proc = ldpc2::cC2V_row_proc<__half,
                                                                         ldpc2::cC2V_row_context<__half,
                                                                                                 sign_dst_fp_t,
                                                                                                 ldpc2::min_sum_update_half_0,
                                                                                                 TStorage>
                                                                        >;
    //------------------------------------------------------------------
    // cC2V_box_plus_row_map
    // Row processing dispatch template structure, which chooses between
    // given compressed C2V and box plus processor classes, based on a
    // comparison between the update row degree and the storage size.
    template <int   BG,
              int   CHECK_IDX,
              class TC2VStorage,
              class TCC2V,
              class TBoxPlus> struct cC2V_box_plus_row_map
    {
        static const int UPDATE_NUM_WORDS = row_num_words<__half, update_row_degree<BG, CHECK_IDX>::value>::value;
        
        // Use box-plus for rows that can store all update data in the
        // C2V storage type, and compressed C2V elsewhere.
        typedef typename std::conditional<UPDATE_NUM_WORDS <= TC2VStorage::NUM_WORDS,
                                          TBoxPlus,
                                          TCC2V>::type row_proc_t;
    };
    //------------------------------------------------------------------
    // cC2v_box_plus_row_map_t
    // Alias template for prescribed cC2V and box plus row processors
    // that have been (manually) determined to be the fastest.
    template <int   BG,
              int   CHECK_IDX,
              class TC2VStorage> using cC2V_box_plus_row_map_t = cC2V_box_plus_row_map<BG,
                                                                                       CHECK_IDX,
                                                                                       TC2VStorage,
                                                                                       cC2V_row_proc<TC2VStorage>,
                                                                                       box_plus_row_proc<ldpc2::box_plus_op>>;
    
    //------------------------------------------------------------------
    // Kernel configuration structure, with typedefs for kernel execution
    // Template parameters:
    // BG_: base graph (1 or 2)
    // NUM_STORAGE_WORDS: Number of storage words per parity row
    // TKernelParams: Class/struct used for kernel parameters
    // TLLR: Source LLR data type (__half or float)
    // TLoader: LLR loader template struct (e.g. llr_loader_batch)
    template <int                                                                BG_,
              int                                                                NUM_STORAGE_WORDS,
              class                                                              TKernelParams,
              typename                                                           TLLR,
              template <typename, int, template<typename, typename> class> class TLoader>
    struct ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config
    {
        static constexpr int BG              = BG_;
        static constexpr int MIN_PARITY_ROWS = 4;
        static constexpr int MAX_PARITY_ROWS = ldpc2::max_parity_nodes<BG>::value;

        template <int bg_> using bg_desc_t   = ldpc2::BG_adj_desc<bg_>;

        // C2V per-row storage. Larger storage allows faster row
        // processing, but increases register pressure (and may incur
        // register spills).
        typedef ldpc2::C2V_storage_t<__half, NUM_STORAGE_WORDS>                  c2v_storage_t;
        typedef TKernelParams                                                    kernel_params_t;
        typedef C2V_row_proc<__half,
                             BG,
                             cC2V_box_plus_row_map_t,
                             app_loader,
                             app_writer>               C2V_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V message cache (register memory here)
        typedef ldpc2::c2v_cache_register<BG,
                                          MAX_PARITY_ROWS,
                                          C2V_t,
                                          c2v_storage_t,
                                          kernel_params_t>                      c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        typedef TLoader<TLLR, 4, llr_op_clamp>                                llr_loader_t;

        // Data type in APP shared memory buffer (__half or __half2)
        typedef typename llr_loader_t::app_buf_t                                app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };
} // namespace

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config<1,
                                                                 3,
                                                                 ldpc2::LDPC_kernel_params,
                                                                 __half,
                                                                 ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config<2, // BG
                                                                 5, // Num storage words
                                                                 ldpc2::LDPC_kernel_params,
                                                                 __half,
                                                                 ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params,
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
// ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template

    typedef ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config<1,
                                                                 3,
                                                                 cuphyLDPCDecodeConfigDesc_t,
                                                                 __half,
                                                                 ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
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
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config<2, // BG
                                                                 5, // Num storage words
                                                                 cuphyLDPCDecodeConfigDesc_t,
                                                                 __half,
                                                                 ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_fp32()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_fp32(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config<1,
                                                                 3,
                                                                 ldpc2::LDPC_kernel_params,
                                                                 float,
                                                                 ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_fp32()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_fp32(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config<2,                         // BG
                                                                 5,                         // Num storage words
                                                                 ldpc2::LDPC_kernel_params, // Kernel params struct
                                                                 float,                     // Source LLR type
                                                                 ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params,
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
// ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config<1,
                                                                 3,
                                                                 cuphyLDPCDecodeConfigDesc_t,
                                                                 float,
                                                                 ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
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
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_dp_desc_dyn_row_dep_kernel_config<2, // BG
                                                                 5, // Num storage words
                                                                 cuphyLDPCDecodeConfigDesc_t,
                                                                 float,
                                                                 ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// reg_index_fp_dp_desc_dyn_row_dep::decode()
cuphyStatus_t reg_index_fp_dp_desc_dyn_row_dep::decode(ldpc::decoder&                     dec,
                                                       LDPC_output_t&                     tDst,
                                                       const_tensor_pair&                 tLLR,
                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                       cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_reg_index_fp_dp_desc_dyn_row_dep()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    dim3 grdDim(NUM_CW);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst);
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
                                                      params.Z,             // lifting size
                                                      sizeof(__half));      // element size

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
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    } else if(llrType == CUPHY_R_32F)
    {
        switch(config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(params.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_fp32, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_fp32, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
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
// reg_index_fp_dp_desc_dyn_row_dep::decode_tb()
cuphyStatus_t reg_index_fp_dp_desc_dyn_row_dep::decode_tb(ldpc::decoder&               dec,
                                                          const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                          cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::reg_index_fp_dp_desc_dyn_row_dep::decode_tb()\n");
    
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    dim3          grdDim(ldpc::decoder::get_total_num_codewords(decodeDesc));
    dim3          blkDim(decodeDesc.config.Z);

    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Determine the dynamic amount of shared memory
                const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<1>::value, // num shared memory nodes
                                                                  decodeDesc.config.Z,                                           // lifting size
                                                                  sizeof(__half));                                               // element size

                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;

            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Determine the dynamic amount of shared memory
                const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<2>::value, // num shared memory nodes
                                                                  decodeDesc.config.Z,                                           // lifting size
                                                                  sizeof(__half));                                               // element size

                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
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
                // Determine the dynamic amount of shared memory
                const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<1>::value, // num shared memory nodes
                                                                  decodeDesc.config.Z,                                           // lifting size
                                                                  sizeof(__half));                                               // element size

                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;

            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Determine the dynamic amount of shared memory
                const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<2>::value, // num shared memory nodes
                                                                  decodeDesc.config.Z,                                           // lifting size
                                                                  sizeof(__half));                                               // element size

                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
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
// reg_index_fp_dp_desc_dyn_row_dep::get_workspace_size()
std::pair<bool, size_t> reg_index_fp_dp_desc_dyn_row_dep::get_workspace_size(const ldpc::decoder&               dec,
                                                                             const cuphyLDPCDecodeConfigDesc_t& config,
                                                                             int                                num_cw)

{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_dp_desc_dyn_row_dep::reg_index_fp_dp_desc_dyn()
reg_index_fp_dp_desc_dyn_row_dep::reg_index_fp_dp_desc_dyn_row_dep(ldpc::decoder& desc)
{
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory. Kernels use __half
    // for shared memory APP storage, even if the source LLR data is
    // float.
    const uint32_t MAX_BG1_SHMEM_SIZE = shmem_llr_buffer_size(ldpc2::max_variable_nodes<1>::value, // num shared memory nodes
                                                              CUPHY_LDPC_MAX_LIFTING_SIZE,         // lifting size
                                                              sizeof(__half));                     // element size
    const uint32_t MAX_BG2_SHMEM_SIZE = shmem_llr_buffer_size(ldpc2::max_variable_nodes<2>::value, // num shared memory nodes
                                                              CUPHY_LDPC_MAX_LIFTING_SIZE,         // lifting size
                                                              sizeof(__half));                     // element size
    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 8> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep,         MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep,         MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb,      MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb,      MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_fp32,    MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_fp32,    MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32, MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32, MAX_BG2_SHMEM_SIZE),
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_fp32);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_fp32);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb_fp32);
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_dp_desc_dyn_row_dep::get_launch_config()
cuphyStatus_t reg_index_fp_dp_desc_dyn_row_dep::get_launch_config(const ldpc::decoder&           dec,
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

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codewords(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;
    launchConfig.kernel_node_params_driver.sharedMemBytes = shmem_llr_buffer_size(NUM_VAR_NODES,   // num shared memory nodes
                                                                                  Z,               // lifting size
                                                                                  sizeof(__half)); // element size

    cudaFunction_t deviceFunction;
    cudaError_t    e = (BG == 1) ?  cudaGetFuncBySymbol(&deviceFunction, (void*)ldpc2_BG1_reg_index_fp_dp_desc_dyn_row_dep_tb): 
                                    cudaGetFuncBySymbol(&deviceFunction, (void*)ldpc2_BG2_reg_index_fp_dp_desc_dyn_row_dep_tb);
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
