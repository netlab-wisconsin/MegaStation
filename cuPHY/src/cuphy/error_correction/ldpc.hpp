/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CUPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_)
#define CUPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_

#include <vector>
#include "tensor_desc.hpp"
#include "cuphy_context.hpp"

typedef tensor_ref_contig_2D<CUPHY_R_32U> LDPC_output_t;

////////////////////////////////////////////////////////////////////////
// cuphyLDPCDecoder
// Empty base class for internal context class, used by forward
// declaration in public-facing cuphy.h.
struct cuphyLDPCDecoder
{
};

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{

class decoder;
    
////////////////////////////////////////////////////////////////////////
// ldpc::decode_algo
// Base class for different LDPC decoder implementations
class decode_algo
{
public:
    //------------------------------------------------------------------
    // destructor
    virtual ~decode_algo() {}
    //------------------------------------------------------------------
    // decode()
    // Decode the input codewords ("legacy" tensor interface)
    virtual cuphyStatus_t decode(decoder&                           dec,
                                 LDPC_output_t&                     tDst,
                                 const_tensor_pair&                 tLLR,
                                 const cuphyLDPCDecodeConfigDesc_t& config,
                                 cudaStream_t                       strm) = 0;
    //------------------------------------------------------------------
    // decode_tb()
    // Decode the input codewords (transport block interface)
    virtual cuphyStatus_t decode_tb(decoder&                     dec,
                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    cudaStream_t                 strm) = 0;
    //------------------------------------------------------------------
    // init()
    // One-time (per cuPHY context) initialization for the algorithm
    virtual cuphyStatus_t init(decoder& desc)  { return CUPHY_STATUS_SUCCESS; }
    //------------------------------------------------------------------
    // get_workspace_size()
    // Return the workspace size required for a given configuration
    virtual std::pair<bool, size_t> get_workspace_size(const decoder&                     dec,
                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                       int                                num_cw) = 0;
    //------------------------------------------------------------------
    // can_decode_config()
    virtual bool can_decode_config(const decoder&                     dec,
                                   const cuphyLDPCDecodeConfigDesc_t& config)
    {
        return true;
    }
    //------------------------------------------------------------------
    // get_launch_config()
    virtual cuphyStatus_t get_launch_config(const decoder&                 dec,
                                            cuphyLDPCDecodeLaunchConfig_t& launchConfig) = 0;
        
protected:
    //------------------------------------------------------------------
    // constructor
    decode_algo() {}
private:
};

////////////////////////////////////////////////////////////////////////
// decoder
class decoder : public cuphyLDPCDecoder
{
public:
    //------------------------------------------------------------------
    // decoder()
    decoder(const cuphy_i::context& ctx);
    //------------------------------------------------------------------
    // decode()
    // LDPC decode using the tensor descriptor interface
    cuphyStatus_t decode(tensor_pair&                       tDst,
                         const_tensor_pair&                 tLLR,
                         const cuphyLDPCDecodeConfigDesc_t& config,
                         cudaStream_t                       strm);
    //------------------------------------------------------------------
    // decode_tb()
    // LDPC decode using the transport block interface
    cuphyStatus_t decode_tb(const cuphyLDPCDecodeDesc_t&  decodeDesc,
                            cudaStream_t                  strm);
    //------------------------------------------------------------------
    // workspace_size()
    std::pair<bool, size_t> workspace_size(const cuphyLDPCDecodeConfigDesc_t& config,
                                           int                                numCodeWords);
    //------------------------------------------------------------------
    // choose_algo()
    int choose_algo(const cuphyLDPCDecodeConfigDesc_t& config) const;
    //------------------------------------------------------------------
    // device index
    int index() const { return deviceIndex_; }
    //------------------------------------------------------------------
    // compute capability
    uint64_t compute_cap() const { return cc_; }
    //------------------------------------------------------------------
    // maximum shared mem per block (optin)
    int max_shmem_per_block_optin() const { return sharedMemPerBlockOptin_; }
    //------------------------------------------------------------------
    // SM count
    int sm_count() const { return multiProcessorCount_; }
    //------------------------------------------------------------------
    // get_total_num_codewords()
    static int get_total_num_codewords(const cuphyLDPCDecodeDesc_t& decodeDesc)
    {
        int num = 0;
        for(int i = 0; i < decodeDesc.num_tbs; ++i)
        {
            num += decodeDesc.llr_input[i].num_codewords;
        }
        return num;
    }
    //------------------------------------------------------------------
    // get_total_num_codeword_pairs()
    static int get_total_num_codeword_pairs(const cuphyLDPCDecodeDesc_t& decodeDesc)
    {
        // Determine the number of codeword pairs. Note that if there
        // are multiple transport blocks, and if any of the transport
        // blocks have an ODD number of codewords, we may have one or
        // more non-full "interior" pairs.
        int num = 0;
        for(int i = 0; i < decodeDesc.num_tbs; ++i)
        {
            num += (decodeDesc.llr_input[i].num_codewords + 1) / 2;
        }
        return num;
    }
    //------------------------------------------------------------------
    // set_normalization()
    cuphyStatus_t set_normalization(cuphyLDPCDecodeConfigDesc_t& config);
    //------------------------------------------------------------------
    // get_launch_config()
    cuphyStatus_t get_launch_config(cuphyLDPCDecodeLaunchConfig_t& launchConfig);
    //------------------------------------------------------------------
    static int get_num_variable_nodes(int BG, int num_parity)
    {
        return num_parity + ((1 == BG) ? 22 : 10);
    }
private:
    int choose_algo_sm70(const cuphyLDPCDecodeConfigDesc_t& config) const;
    int choose_algo_sm75(const cuphyLDPCDecodeConfigDesc_t& config) const;
    int choose_algo_sm80(const cuphyLDPCDecodeConfigDesc_t& config) const;
    int choose_algo_sm86(const cuphyLDPCDecodeConfigDesc_t& config) const;
    int choose_algo_sm89(const cuphyLDPCDecodeConfigDesc_t& config) const;
    int choose_algo_sm90(const cuphyLDPCDecodeConfigDesc_t& config) const;

    typedef std::unique_ptr<decode_algo> decode_algo_ptr_t;
    //------------------------------------------------------------------
    // Data
    // Note: copying required device data from cuPHY context to avoid
    //       issues if cuphy context is destroyed before decoder object.
    int      deviceIndex_;            // index of device associated with context
    uint64_t cc_;                     // compute capability (major << 32) | minor
    int      sharedMemPerBlockOptin_; // maximum shared memory per block usable by option
    int      multiProcessorCount_;    // number of multiprocessors on device

    // Pointers to different LDPC decoder implementations
    std::vector<decode_algo_ptr_t> algos_;
};

} // namespace ldpc

// ldpcEncodeDescr is passed as a __grid_constant__ and is currently 32 bytes.
// If adding fields, note any size increases. Some of the fields below could
// have their widths shortened if necessary.
struct ldpcEncodeDescr
{
    // Swapped old overprovisioned arrays of PDSCH_MAX_UES_PER_CELL_GROUP with pointers to buffers. Note that
    // these would need to be allocated by the caller; OK for PDSCH not as user-friendly for component level API.
    LDPC_output_t* input;
    LDPC_output_t* output;
    int num_TBs; // Number of input or output elements that are valid for each buffer. The size of the buffers themselves can be different.
    uint16_t BG;
    uint16_t Kb;
    uint16_t Z;
    uint16_t num_rows;
    char H_type;
    bool puncture;
};
typedef struct ldpcEncodeDescr ldpcEncodeDescr_t;
typedef struct ldpcEncodeDescr ldpcEncodeDescr_t_array[PDSCH_MAX_N_TBS_SUPPORTED];

#endif // !defined(CUPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_)
