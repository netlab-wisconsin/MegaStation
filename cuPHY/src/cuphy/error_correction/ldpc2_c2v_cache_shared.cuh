/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_C2V_CACHE_SHARED_CUH_INCLUDED_)
#define LDPC2_C2V_CACHE_SHARED_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// c2v_cache_shared
// Check to variable (C2V) messages are stored in shared memory.
// Assumes that all APP variables are in shared memory.
// BG: Base graph (1 or 2)
// NUM_SMEM_VNODES: Number of APP variable nodes in shared memory
// TC2V: Check to variable node class
template <int          BG_,
          class        TC2V,
          class        TC2VStorage,
          class        TKernelParams>
struct c2v_cache_shared
{
    //------------------------------------------------------------------
    // C2V message type
    typedef TC2V                  c2v_t;
    typedef typename c2v_t::app_t app_t;
    typedef TC2VStorage           c2v_storage_t;
    static const int BG = BG_;
    //------------------------------------------------------------------
    // c2v_cache_shared()
    __device__
    c2v_cache_shared(char*                     smem,
                     const LDPC_kernel_params& params)

    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the address of C2V data in shared memory (for this
        // thread).
        c2v_storage_ = get_thread_c2v_address(smem, params.Z_var_szelem);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize shared memory with "zero" C2V data
        init_shared_c2v(params.num_parity_nodes, params.Z);
    }
    __device__
    c2v_cache_shared(char*                              smem,
                     const cuphyLDPCDecodeConfigDesc_t& params)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the address of C2V data in shared memory (for this
        // thread).
        const int Kb              = (1 == BG_) ? 22 : 10;
        const int NUM_VAR_NODES   = Kb + params.num_parity_nodes;
        c2v_storage_              = get_thread_c2v_address(smem,
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
        // No loop here, since we zero-initialize shared memory in the
        // constructor.
        //#pragma unroll
        //for(int i = 0; i < NUM_PARITY_NODES; ++i)
        //{
        //    c2v_[i].init();
        //}
    }
    //------------------------------------------------------------------
    // process_row()
    template <int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Load storage from shared memory
        c2v_storage_t st(c2v_storage_ + (CHECK_IDX * params.Z));
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize a C2V row processor and process the row
        c2v_t c2v;
        c2v.process_row<CHECK_IDX>(params, app, app_addr, st, smem_offset);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Store C2V data back to shared memory, to be used during the
        // next iteration
        st.store(c2v_storage_ + (CHECK_IDX * params.Z));
    }
private:
    //------------------------------------------------------------------
    // get_thread_c2v_address()
    __device__
    c2v_storage_t* get_thread_c2v_address(char* smem,
                                          int   app_size_bytes)
    {
        const int      APP_SIZE_PADDED = round_up_to_next(app_size_bytes,
                                                          static_cast<int>(alignof(c2v_storage_t)));
        c2v_storage_t* c2v_base        = reinterpret_cast<c2v_storage_t*>(smem + APP_SIZE_PADDED);
        return (c2v_base + threadIdx.x);
    }
    //------------------------------------------------------------------
    // init_shared_c2v()
    __device__
    void init_shared_c2v(int num_parity_nodes, int Z)
    {
        c2v_storage_t sZero;
        sZero.init();
        for(int i = 0; i < num_parity_nodes; ++i)
        {
            sZero.store(c2v_storage_ + (i * Z));
        }
    }
    //------------------------------------------------------------------
    // Data
    c2v_storage_t* c2v_storage_;
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CACHE_SHARED_CUH_INCLUDED_)