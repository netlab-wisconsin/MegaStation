/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_C2V_CACHE_SPLIT_CUH_INCLUDED_)
#define LDPC2_C2V_CACHE_SPLIT_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// c2v_cache_split
// Check to variable (C2V) messages are stored in both shared memory
// and registers
// BG: Base graph (1 or 2)
// NUM_REG_C2V_NODES: Number of cC2V nodes stored in shared memory
// TC2V: Check to variable node class
template <int   BG_,
          int   NUM_REG_C2V_NODES,
          class TC2V,
          class TStorageCore,
          class TStorageNonCore,
          class TKernelParams>
struct c2v_cache_split
{
    //------------------------------------------------------------------
    typedef TC2V                  c2v_t;
    typedef typename c2v_t::app_t app_t;
    typedef TStorageCore          c2v_storage_core_t;
    typedef TStorageNonCore       c2v_storage_noncore_t;
    static const int BG = BG_;
    //------------------------------------------------------------------
    // c2v_cache_split()
    __device__
    c2v_cache_split(char*                     smem,
                    const LDPC_kernel_params& params)
    //: smem_(smem), Z_var_(params.Z_var)
    {
        init_reg_c2v();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the address of C2V data in shared memory (for this
        // thread).
        c2v_storage_shm_          = get_thread_c2v_address(smem,
                                                           params.Z_var * sizeof(app_t));
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize shared memory with "zero" C2V data
        init_shared_c2v(params.num_parity_nodes, params.Z);
    }
    //------------------------------------------------------------------
    // c2v_cache_split()
    __device__
    c2v_cache_split(char*                              smem,
                    const cuphyLDPCDecodeConfigDesc_t& params)

    {
        init_reg_c2v();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the address of C2V data in shared memory (for this
        // thread).
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the address of C2V data in shared memory (for this
        // thread).
        const int Kb              = (1 == BG_) ? 22 : 10;
        const int NUM_VAR_NODES   = Kb + params.num_parity_nodes;
        c2v_storage_shm_          = get_thread_c2v_address(smem,
                                                           params.Z * NUM_VAR_NODES * sizeof(app_t));
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize shared memory with "zero" C2V data
        init_shared_c2v(params.num_parity_nodes, params.Z);
    }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        // Nothing done here, since we zero-initialize shared memory in
        // the constructor.
    }
    //------------------------------------------------------------------
    template <int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        //thread0_dump_app(smem_, Z_var_);
        if(CHECK_IDX < 4)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Initialize a C2V row processor and process the row
            c2v_t c2v;
            c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_core_t>(params,
                                                                                   app,
                                                                                   app_addr,
                                                                                   c2v_storage_reg_core_[CHECK_IDX],
                                                                                   smem_offset);
        }
        else if(CHECK_IDX < NUM_REG_C2V_NODES)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Initialize a C2V row processor and process the row
            c2v_t c2v;
            c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_noncore_t>(params,
                                                                                      app,
                                                                                      app_addr,
                                                                                      c2v_storage_reg_[CHECK_IDX-4],
                                                                                      smem_offset);
        }
        else
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Load storage from shared memory
            c2v_storage_noncore_t st = c2v_storage_shm_[(CHECK_IDX  - NUM_REG_C2V_NODES) * params.Z];
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Initialize a C2V row processor and process the row
            c2v_t c2v;
            c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_noncore_t>(params,
                                                                                      app,
                                                                                      app_addr,
                                                                                      st,
                                                                                      smem_offset);
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Store C2V data back to shared memory, to be used during the
            // next iteration
            c2v_storage_shm_[(CHECK_IDX  - NUM_REG_C2V_NODES) * params.Z] = st;
        }
        //thread0_dump_app(smem_, Z_var_);
    }
private:
    //------------------------------------------------------------------
    // init_reg_c2v()
    __device__
    void init_reg_c2v()
    {
        #pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            c2v_storage_reg_core_[i].init();
        }
        #pragma unroll
        for(int i = 4; i < NUM_REG_C2V_NODES; ++i)
        {
            c2v_storage_reg_[i-4].init();
        }
    }
    //------------------------------------------------------------------
    // init_shared_c2v()
    __device__
    void init_shared_c2v(int num_parity_nodes, int Z)
    {
        const int             NUM_SHMEM_NODES = num_parity_nodes - NUM_REG_C2V_NODES;
        c2v_storage_noncore_t sZero;
        sZero.init();
        for(int i = 0; i < NUM_SHMEM_NODES; ++i)
        {
            c2v_storage_shm_[i * Z] = sZero;
        }
    }
    //------------------------------------------------------------------
    // get_thread_c2v_address()
    __device__
    c2v_storage_noncore_t* get_thread_c2v_address(char* smem,
                                                  int   app_size_bytes)
    {
        const int              APP_SIZE_PADDED = round_up_to_next(app_size_bytes,
                                                                  static_cast<int>(alignof(c2v_storage_noncore_t)));
        c2v_storage_noncore_t* c2v_base        = reinterpret_cast<c2v_storage_noncore_t*>(smem + APP_SIZE_PADDED);
        return (c2v_base + threadIdx.x);
    }
    //------------------------------------------------------------------
    // Data
    c2v_storage_core_t     c2v_storage_reg_core_[4];
    c2v_storage_noncore_t  c2v_storage_reg_[NUM_REG_C2V_NODES-4];
    c2v_storage_noncore_t* c2v_storage_shm_;
    //char* smem_;
    //int Z_var_;
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CACHE_SPLIT_CUH_INCLUDED_)