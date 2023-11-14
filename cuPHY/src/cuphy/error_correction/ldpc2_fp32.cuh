/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_FP32_CUH_INCLUDED_)
#define LDPC2_FP32_CUH_INCLUDED_

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// cC2V_storage<float>
template <> struct cC2V_storage_t<float>
{
    float     min0;
    float     min1;
    uint32_t  signs_index;
};

////////////////////////////////////////////////////////////////////////
// min1_policy_delta
// The difference between min0 and min1 is stored in the min1_or_delta
// field. When retrieving values from a compressed c2v structure, the
// min0 value is used - even for the index that provided min0. After
// updating with the min0 value, a scattered update with (min1 - min0)
// is performed. The advantage is that a per-output comparison with the
// min0 location is not required. However, the read-modify-write of the
// shared memory location for min0 may induce stalls.
template <typename T> struct min1_policy_delta;

template <> struct min1_policy_delta<float>
{
    //------------------------------------------------------------------
    // The min1_or_delta value is added via a write to shared memory,
    // so the min1_or_delta_value does not use the stored signs. We
    // indicate that no other class should set the sign of the min1
    // field.
    static const bool min1_sign_set = true;
    //------------------------------------------------------------------
    // finalize()
    // Update min1_or_delta to have the delta between min1 and min0,
    // taking the destination sign into account. (For a min1_delta
    // approach, the sign application will not be performed.)
    // sign_prod = 0 or 1, can be obtained via popc() & 0x1
    // Note that this function relies on min0 having the sign that it
    // should have AFTER the compressed c2V is expanded.
    static __device__ __inline__
    void finalize(word_t&  min0,
                  word_t&  min1_or_delta,
                  uint32_t sign_prod)
    {
        // We want to determine what to add to min0 to get min1,
        // taking the sign into account.
        // min0 + delta = min1
        // At this point, the sign of min0 is the "correct"
        // sign, i.e. the sign that we want when we expand the
        // compressed C2V value.
        // Note that at this point, min1 may be negative
        // or positive - it will be whatever the value was
        // when encountered during the update() function.
        // If min0 > 0:
        // <------------|---------|------|------->
        //              0       min0  abs(min1)
        //     delta = abs(min1) - min0
        // If min0 < 0:
        // <------------|---------|------|------->
        //        -abs(min1)     min0    0
        //     delta = -abs(min1) - min0
        //           = -(abs(min1) + min0)
        //           = -(abs(min1) - abs(min0))
        // So to handle both cases:
        // delta = [abs(min1) - abs(min0)] * sign(min0)
        //     where sign(min0) = +1 or -1
        min1_or_delta.f32 = fabsf(min1_or_delta.f32) - fabsf(min0.f32);
        min1_or_delta.u32 |= ((sign_prod << 31) ^ word_sign_mask(min0).u32);
    }
    //------------------------------------------------------------------
    // process_row_begin()
    static __device__ __inline__
    void process_row_begin(int min0_loc, const word_t& delta)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Scatter an update to the min0 value so that it is now min1
        smem_decrement(min0_loc, delta.f32);
    }
    //------------------------------------------------------------------
    // process_row_end()
    static __device__ __inline__
    void process_row_end(int min0_loc, const word_t& delta)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Scatter an update to the min0 value so that it is now min1
        smem_increment(min0_loc, delta.f32);
    }
    //------------------------------------------------------------------
    // get_update_value()
    static __device__ __inline__
    word_t get_update_value(const word_t& min0,
                            const word_t& min1_or_delta,
                            int           min0_loc,
                            int           index,
                            int           /*loc_unused*/)
    {
        // With min1 delta processing, we always use min0 for the update.
        // We will update the min0 value with the difference as a separate
        // step.
        return min0;
    }
};

////////////////////////////////////////////////////////////////////////
// min1_policy_default
// min1 value determined through normal min search is stored unmodified.
// When expanding the compressed c2V, the location is compared to the
// min0 location, and if they are equal min1 is used. (This approach
// requires comparison to the min0 location at each value.)
template <typename T> struct min1_policy_default;

template <> struct min1_policy_default<float>
{
    static const bool min1_sign_set = false;
    //------------------------------------------------------------------
    // finalize()
    static __device__ __inline__
    void finalize(word_t&  min0,
                  word_t&  min1_or_delta,
                  uint32_t sign_prod)
    {
    }
    //------------------------------------------------------------------
    // process_row_begin()
    static __device__ __inline__
    void process_row_begin(int min0_loc, const word_t& delta) {}
    //------------------------------------------------------------------
    // process_row_end()
    static __device__ __inline__
    void process_row_end(int min0_loc, const word_t& delta) {}
    //------------------------------------------------------------------
    // get_update_value()
    static __device__ __inline__
    word_t get_update_value(const word_t& min0,
                            const word_t& min1_or_delta,
                            int           min0_loc,
                            int           /*index*/,
                            int           loc)
    {
        return (loc == min0_loc) ? min1_or_delta : min0;
    }
};

template <int   BG,
          class TSignManager,
          class TMinSumUpdate>
struct cC2V_index<float, BG, TSignManager, TMinSumUpdate>
{
    typedef TSignManager               sign_mgr_t;
    typedef min1_policy_default<float> min1_t;
    typedef float                      app_t;
    //------------------------------------------------------------------
    // init_row()
    // Initialization function using first two input values
    __device__
    void init_row(word_t v0, int /*address0*/, word_t v1, int /*address1*/)
    {
        min0          = (fabsf(v0.f32) <= fabsf(v1.f32)) ? v0 : v1;
        min1_or_delta = (fabsf(v0.f32) <= fabsf(v1.f32)) ? v1 : v0;
        min0_loc      = (fabsf(v0.f32) <= fabsf(v1.f32)) ? 0  : 1;
        sign_mgr_t::init_row(signs, v0, v1);
    }
    //------------------------------------------------------------------
    // update()
    __device__
    void update(word_t v, int /*address*/, int index)
    {
        if(fabsf(v.f32) < fabsf(min0.f32))
        {
            // Note: storing values for min0 and min1, instead of
            // absolute values
            min1_or_delta = min0;
            min0          = v;
            min0_loc      = index;
        }
        else if(fabsf(v.f32) < fabsf(min1_or_delta.f32))
        {
            // Note: storing encountered value for min1, instead of
            // absolute value
            min1_or_delta = v;
        }
        sign_mgr_t::update(signs, v, index);
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize(const LDPC_kernel_params& params, int row_count)
    {
        // Apply normalization
        min0.f32          *= params.norm.f32;
        min1_or_delta.f32 *= params.norm.f32;
        uint32_t sign_prod = sign_mgr_t::sign_product(signs);
        sign_mgr_t::finalize(min0,
                             min1_or_delta,
                             signs,
                             sign_prod,
                             row_count,
                             !min1_t::min1_sign_set);
    }
    //------------------------------------------------------------------
    // process_row_init()
    // CHECK_IDX: Index of check (parity) node
    // NUM_SMEM_APP_CHECK_NODES: Number of check nodes for which APP
    //                           data is stored in shared memory. (For
    //                           check nodes >= this value, the APP data
    //                           will be fetched from global memory.
    template <int CHECK_IDX, int NUM_SMEM_APP_CHECK_NODES>
    __device__
    void process_row_init(const LDPC_kernel_params& params,
                          word_t                    (&app)[app_num_words<float, BG, CHECK_IDX>::value],
                          int                       (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Fetch app values and update the min0/min1/address fields
        #pragma unroll
        for(int i = 0; i < row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Load channel APP from given address
            if((CHECK_IDX < NUM_SMEM_APP_CHECK_NODES) ||
               (i < (row_degree<BG, CHECK_IDX>::value - 1)))
            {
                app[i]  = smem_address_as<word_t>(app_addr[i]);
            }
            else
            {
                app[i]  = gmem_address_as<word_t>(params.input_llr + (blockIdx.x * params.input_llr_stride_elements * sizeof(float)),
                                                  app_addr[i]);
            }
#if 0
            if((0 == threadIdx.x) && (i == 3))
            {
                printf("CHECK_IDX = %i, app[0] = %f, app[1] = %f, app[2] = %f\n", CHECK_IDX, app[0].f32, app[1].f32, app[2].f32);
            }
#endif

            if(1 == i)
            {
                // Initialize with first two values [0, 1]
                init_row(app[0], app_addr[0], app[1], app_addr[1]);
            }
            else if(i > 1)
            {
                // Update with subsequent values
                update(app[i], app_addr[i], i);
            }
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Post-process min0/min1/address fields
        finalize(params, row_degree<BG, CHECK_IDX>::value);
#if 0
        if(0 == threadIdx.x)
        {
            printf("CHECK_IDX = %i, min0 = %f, min1 = %f, signs = 0x%X, index = %i\n", CHECK_IDX, min0.f32, min1_or_delta.f32, signs, min0_loc);
        }
#endif
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // The upper loop bound is the update_row_degree - we don't need
        // to update extension APP values that are used by only one node
        #pragma unroll
        for(int i = 0; i < update_row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Note: passing the index for both args to get_update_value(),
            // since indexed C2Vs use the index for min0 comparisons
            word_t inc = sign_mgr_t::apply_sign(signs,
                                                i,
                                                min1_t::get_update_value(min0,
                                                                         min1_or_delta,
                                                                         min0_loc,
                                                                         i,
                                                                         i));
            app[i].f32 += inc.f32;
            write_shared_word(app[i], app_addr[i]);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // min1 policy processing
        min1_t::process_row_end(min0_loc, min1_or_delta);
    }
    //------------------------------------------------------------------
    // process_row()
    template <int CHECK_IDX, int NUM_SMEM_CHECK_NODES>
    __device__
    void process_row(const LDPC_kernel_params& params,
                     word_t                    (&app)[app_num_words<float, BG, CHECK_IDX>::value],
                     int                       (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // min1 policy processing
        min1_t::process_row_begin(min0_loc, min1_or_delta);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Retrieve the APP value and subtract the value from the
        // previous iteration. (We don't subtract the previous value
        // for isolated edges since they are not modified.)
        #pragma unroll
        for(int i = 0; i < update_row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Load from shared memory
            app[i]  = smem_address_as<word_t>(app_addr[i]);
            // Subtract the contribution from this check node, from the
            // previous iteration
            // Note: passing the index for both args to get_update_value(),
            // since indexed C2Vs use the index for min0 comparisons
            word_t dec = sign_mgr_t::apply_sign(signs,
                                                i,
                                                min1_t::get_update_value(min0,
                                                                         min1_or_delta,
                                                                         min0_loc,
                                                                         i,
                                                                         i));
            app[i].f32 -= dec.f32;
        }
        for(int i = update_row_degree<BG, CHECK_IDX>::value; i < row_degree<BG, CHECK_IDX>::value; ++i)
        {
            if(CHECK_IDX < NUM_SMEM_CHECK_NODES)
            {
                // Load from shared memory
                app[i]  = smem_address_as<word_t>(app_addr[i]);
            }
            else
            {
                // Load from global memory
                app[i]  = gmem_address_as<word_t>(params.input_llr + (blockIdx.x * params.input_llr_stride_elements * sizeof(float)),
                                                  app_addr[i]);
            }
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update the C2V structure with the sequence of values
        #pragma unroll
        for(int i = 0; i < row_degree<BG, CHECK_IDX>::value; ++i)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Use the APP value to update the new C2V
            if(1 == i)
            {
                // Initialize with first two values [0, 1]
                init_row(app[0], app_addr[0], app[1], app_addr[1]);
            }
            else if(i > 1)
            {
                // Update with subsequent values
                update(app[i], app_addr[i], i);
            }
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Post-process min0/min1/address fields
        finalize(params, row_degree<BG, CHECK_IDX>::value);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // The upper loop bound is the update_row_degree - we don't need
        // to update extension APP values that are used by only one node
        #pragma unroll
        for(int i = 0; i < update_row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Note: passing the index for both args to get_update_value(),
            // since indexed C2Vs use the index for min0 comparisons
            word_t inc = sign_mgr_t::apply_sign(signs,
                                                i,
                                                min1_t::get_update_value(min0,
                                                                         min1_or_delta,
                                                                         min0_loc,
                                                                         i,
                                                                         i));
            app[i].f32 += inc.f32;
            write_shared_word(app[i], app_addr[i]);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // min1 policy processing
        min1_t::process_row_end(min0_loc, min1_or_delta);
    }
    //------------------------------------------------------------------
    // load_global()
    __device__
    void load_global(const LDPC_kernel_params& params, int checkIndex)
    {
        static_assert(sizeof(*this) == sizeof(int4), "c2v load assuming int4 size");
        // Each block decodes a different codeword
        // Number of stored c2v messages for each block: mb * Z
        // Size of each c2v message: 16
        const int gmem_offset    = (blockIdx.x * params.mbz16) + (checkIndex * params.z16) + (threadIdx.x * sizeof(int4));
        char*     load_address   = static_cast<char*>(params.workspace) + gmem_offset;
        int4      load_value     = *(reinterpret_cast<const int4*>(load_address));
        min0.i32                 = load_value.x;
        min1_or_delta.i32        = load_value.y;
        signs                    = static_cast<uint32_t>(load_value.z);
        min0_loc                 = load_value.w;
    }
    //------------------------------------------------------------------
    // store_global()
    __device__
    void store_global(const LDPC_kernel_params& params, int checkIndex)
    {
        static_assert(sizeof(*this) == sizeof(int4), "c2v load assuming int4 size");
        // Each block decodes a different codeword
        // Number of stored c2v messages for each block: mb * Z
        // Size of each c2v message: 16
        const int gmem_offset    = (blockIdx.x * params.mbz16) + (checkIndex * params.z16) + (threadIdx.x * sizeof(int4));
        char*     store_address  = static_cast<char*>(params.workspace) + gmem_offset;
        int4      store_value    = {min0.i32, min1_or_delta.i32, static_cast<int32_t>(signs), min0_loc};
        *(reinterpret_cast<int4*>(store_address)) = store_value;
    }
    //------------------------------------------------------------------
    // Data
    word_t    min0;
    word_t    min1_or_delta;
    uint32_t  signs;
    int       min0_loc; // location is an index in this case (not an address)
};

////////////////////////////////////////////////////////////////////////
// cC2V_address
// Compressed "check to variable" representation using APP addresses
// instead of index values.
// The APP address for the min0 value is used to write an increment
// to shared memory, instead of checking whether each index matches the
// min0 index.
template <typename                  T,
          int                       BG,
          template <typename> class TMin1Policy,
          class                     TSignManager> struct cC2V_address;

template <int                       BG,
          template <typename> class TMin1Policy,
          class                     TSignManager>
struct cC2V_address<float, BG, TMin1Policy, TSignManager>
{
    typedef TSignManager            sign_mgr_t;
    typedef TMin1Policy<float>      min1_t;
    typedef float                   app_t;
    //------------------------------------------------------------------
    // init_row()
    // Initialization function using first two input values
    __device__
    void init_row(word_t v0, int address0, word_t v1, int address1)
    {
        min0          = (fabsf(v0.f32) <= fabsf(v1.f32)) ? v0       : v1;
        min1_or_delta = (fabsf(v0.f32) <= fabsf(v1.f32)) ? v1       : v0;
        min0_loc      = (fabsf(v0.f32) <= fabsf(v1.f32)) ? address0 : address1;
        sign_mgr_t::init_row(signs, v0, v1);
    }
    //------------------------------------------------------------------
    // init()
    // Initialization function for no-op first iteration 
    __device__
    void init()
    {
        min0.f32          = 0.0f;
        min1_or_delta.f32 = 0.0f;
        min0_loc          = 0;
        sign_mgr_t::init(signs);
    }
    //------------------------------------------------------------------
    // update()
    __device__
    void update(word_t v, int address, int index)
    {
        if(fabsf(v.f32) < fabsf(min0.f32))
        {
            // Note: storing values for min0 and min1, instead of
            // absolute values
            min1_or_delta = min0;
            min0          = v;
            min0_loc      = address;
        }
        else if(fabsf(v.f32) < fabsf(min1_or_delta.f32))
        {
            // Note: storing value for min1, instead of
            // absolute value
            min1_or_delta = v;
        }
        sign_mgr_t::update(signs, v, index);
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize(const LDPC_kernel_params& params, int row_count)
    {
        // Apply normalization
        min0.f32 *= params.norm.f32;
        min1_or_delta.f32 *= params.norm.f32;
        // Calculate the product of the signs
        uint32_t sign_prod = sign_mgr_t::sign_product(signs);
        // Note: sign_mgr_t::finalize() may change sign of min0, so
        // we call the min1 finalizer first.
        min1_t::finalize(min0, min1_or_delta, sign_prod);
        sign_mgr_t::finalize(min0,
                             min1_or_delta,
                             signs,
                             sign_prod,
                             row_count,
                             !min1_t::min1_sign_set);
    }
    //------------------------------------------------------------------
    // process_row_init()
    template <int CHECK_IDX, int NUM_SMEM_APP_CHECK_NODES>
    __device__
    void process_row_init(const LDPC_kernel_params& params,
                          word_t                    (&app)[app_num_words<float, BG, CHECK_IDX>::value],
                          int                       (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Fetch app values and update the min0/min1/address fields
        #pragma unroll
        for(int i = 0; i < row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Load channel APP from given address
            if((CHECK_IDX < NUM_SMEM_APP_CHECK_NODES)  ||
               (i < (row_degree<BG, CHECK_IDX>::value - 1)))
            {
                app[i]  = smem_address_as<word_t>(app_addr[i]);
            }
            else
            {
                app[i]  = gmem_address_as<word_t>(params.input_llr + (blockIdx.x * params.input_llr_stride_elements * sizeof(float)),
                                                  app_addr[i]);
            }
            if(1 == i)
            {
                // Initialize with first two values [0, 1]
                init_row(app[0], app_addr[0], app[1], app_addr[1]);
            }
            else if(i > 1)
            {
                // Update with subsequent values
                update(app[i], app_addr[i], i);
            }
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Post-process min0/min1/address fields
        finalize(params, row_degree<BG, CHECK_IDX>::value);
#if 0
        if(0 == threadIdx.x)
        {
            printf("CHECK_IDX = %i, min0 = %f, min1 = %f, signs = 0x%X, index = %i\n", CHECK_IDX, min0.f32, min1_or_delta.f32, signs, min0_loc);
        }
#endif
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #pragma unroll
        for(int i = 0; i < row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Get the value, assuming that this isn't the min0 value.
            // (We'll account for that case below.)
            word_t inc = sign_mgr_t::apply_sign(signs,
                                                i,
                                                min1_t::get_update_value(min0,
                                                                         min1_or_delta,
                                                                         min0_loc,
                                                                         i,
                                                                         app_addr[i]));
            app[i].f32 += inc.f32;
            write_shared_word(app[i], app_addr[i]);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // min1 policy processing
        min1_t::process_row_end(min0_loc, min1_or_delta);
    }
    //------------------------------------------------------------------
    // process_row()
    template <int CHECK_IDX, int NUM_SMEM_APP_CHECK_NODES>
    __device__
    void process_row(const LDPC_kernel_params& params,
                     word_t                    (&app)[app_num_words<float, BG, CHECK_IDX>::value],
                     int                       (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // min1 policy processing
        min1_t::process_row_begin(min0_loc, min1_or_delta);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #pragma unroll
        for(int i = 0; i < row_degree<BG, CHECK_IDX>::value; ++i)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Load channel APP from given address
            if(CHECK_IDX <= NUM_SMEM_APP_CHECK_NODES)
            {
                app[i]  = smem_address_as<word_t>(app_addr[i]);
            }
            else
            {
                app[i]  = gmem_address_as<word_t>(params.input_llr + (blockIdx.x * params.input_llr_stride_elements * sizeof(float)),
                                                  app_addr[i]);
            }
            // Subtract the contribution from this check node, from the
            // previous iteration
            word_t dec = sign_mgr_t::apply_sign(signs,
                                                i,
                                                min1_t::get_update_value(min0,
                                                                         min1_or_delta,
                                                                         min0_loc,
                                                                         i,
                                                                         app_addr[i]));
            app[i].f32 -= dec.f32;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #pragma unroll
        for(int i = 0; i < row_degree<BG, CHECK_IDX>::value; ++i)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Use the APP value to update the new C2V
            if(1 == i)
            {
                // Initialize with first two values [0, 1]
                init_row(app[0], app_addr[0], app[1], app_addr[1]);
            }
            else if(i > 1)
            {
                // Update with subsequent values
                update(app[i], app_addr[i], i);
            }
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Post-process min0/min1/address fields
        finalize(params, row_degree<BG, CHECK_IDX>::value);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #pragma unroll
        for(int i = 0; i < row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Get the value, assuming that this isn't the min0 value.
            // (We'll account for that case below.)
            word_t inc = sign_mgr_t::apply_sign(signs,
                                                i,
                                                min1_t::get_update_value(min0,
                                                                         min1_or_delta,
                                                                         min0_loc,
                                                                         i,
                                                                         app_addr[i]));
            app[i].f32 += inc.f32;
            write_shared_word(app[i], app_addr[i]);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // min1 policy processing
        min1_t::process_row_end(min0_loc, min1_or_delta);
    }
    //------------------------------------------------------------------
    // load_global()
    __device__
    void load_global(const LDPC_kernel_params& params, int checkIdx)
    {
        static_assert(sizeof(*this) == sizeof(int4), "c2v load assuming int4 size");
        // Each block decodes a different codeword
        // Number of stored c2v messages for each block: mb * Z
        // Size of each c2v message: 16
        const int gmem_offset    = (blockIdx.x * params.mbz16) + (checkIdx * params.z16) + (threadIdx.x * sizeof(int4));
        char*     load_address   = static_cast<char*>(params.workspace) + gmem_offset;
        int4      load_value     = *(reinterpret_cast<const int4*>(load_address));
        min0.i32                 = load_value.x;
        min1_or_delta.i32        = load_value.y;
        signs                    = static_cast<uint32_t>(load_value.z);
        min0_loc                 = load_value.w;
    }
    //------------------------------------------------------------------
    // store_global()
    __device__
    void store_global(const LDPC_kernel_params& params, int checkIdx)
    {
        static_assert(sizeof(*this) == sizeof(int4), "c2v load assuming int4 size");
        // Each block decodes a different codeword
        // Number of stored c2v messages for each block: mb * Z
        // Size of each c2v message: 16
        const int gmem_offset    = (blockIdx.x * params.mbz16) + (checkIdx * params.z16) + (threadIdx.x * sizeof(int4));
        char*     store_address  = static_cast<char*>(params.workspace) + gmem_offset;
        int4      store_value    = {min0.i32, min1_or_delta.i32, static_cast<int32_t>(signs), min0_loc};
        *(reinterpret_cast<int4*>(store_address)) = store_value;
    }
    //------------------------------------------------------------------
    // Data
    word_t    min0;
    word_t    min1_or_delta;
    uint32_t  signs;
    int       min0_loc; // location refers to shmem address here
};

} // namespace ldpc2

#endif // !defined(LDPC2_FP32_CUH_INCLUDED_)
