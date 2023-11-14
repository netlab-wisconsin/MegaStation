/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(BFC_HPP_INCLUDED_)
#define BFC_HPP_INCLUDED_

#include "tensor_desc.hpp"
#include "cuphy.h"

// Implementation of the beamforming coefficient compute interface exposed as an opaque data type 
// to abstract out implementation details (beamforming coefficient compute C++ class). The beamforming 
// coefficient compute is implemented as a C++ class which inherits from this interface structure defined
// as an empty shell (opaque type is a struct since the interface is C compatible). Pointer to the opaque 
// type is also exposed in the interface as a handle to the underlying implementation
struct cuphyBfwCoefComp
{};

namespace bfw_coefComp
{
void bfcCoefCompute(uint32_t           nBSAnts,
                    uint32_t           nLayers,
                    uint32_t           Nprb,
                    const_tensor_pair& tH,
                    const_tensor_pair& tLambda,
                    tensor_pair&       tCoef,
                    tensor_pair&       tDbg,
                    cudaStream_t       strm);


typedef struct _bfwCoefCompKernelBfLayerPrm
{
    uint8_t             ueLayerIdx;       // Layer index of a UE
    uint16_t            startPrbGrpOffset;// Start PRB group offset within SRS ChEst buffer
    cuphyTensorInfo3_t  tInfoSrsChEst;    // SRS channel estimation tensor for a UE
} bfwCoefCompKernelBfLayerPrm_t;

typedef struct _bfwCoefCompKernelUeGrpPrm
{
    uint16_t                       nPrbGrp;         // number of beamforming weights in frequency    
    uint16_t                       nRxAnt;          // number of gNB receiving antennas
    uint8_t                        nBfLayers;       // number of layers being beamformed
    bfwCoefCompKernelBfLayerPrm_t* pBfLayerPrmCpu;  // Per layer parameters CPU pointer
    bfwCoefCompKernelBfLayerPrm_t* pBfLayerPrmGpu;  // Per layer parameters GPU pointer
#ifdef BFW_BOTH_COMP_FLOAT
    cuphyTensorInfo3_t             tInfoBfwCoefs;   // BFW coefficient tensor
#endif
    uint8_t*                       pBfwCompCoef;    // BFW compressed coefficient buffer
} bfwCoefCompKernelUeGrpPrm_t;

// BFW coef compute static descriptor
typedef struct _bfwCoefCompStatDescr
{
    float                          lambda;
    uint8_t                        compressBitwidth;
    bfwCoefCompKernelUeGrpPrm_t*   pKernelUeGrpPrms;
    // bfwCoefCompKernelBfLayerPrm_t* pKernelBfLayerPrms;      
    uint16_t*                      pHetCfgUeGrpMap[CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS]; // Mapping of Heterogenous config to UE group
} bfwCoefCompStatDescr_t;

// BFW coef compute dynamic descriptor
typedef struct _bfwCoefCompDynDescr
{
    uint16_t hetCfgIdx;
} bfwCoefCompDynDescr_t;
using bfwCoefCompDynDescrArr_t = std::array<bfwCoefCompDynDescr_t,CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS>;

// Class implementation of the BFW coefficient compute component
class bfwCoefComp : public cuphyBfwCoefComp 
{
public:
    bfwCoefComp(uint16_t nMaxUeGrps, uint16_t nMaxTotalLayers)
    : m_nMaxUeGrps(nMaxUeGrps),
      m_nMaxTotalLayers(nMaxTotalLayers)
    {
    }

    ~bfwCoefComp()                  = default;
    bfwCoefComp(bfwCoefComp const&) = delete;
    bfwCoefComp& operator=(bfwCoefComp const&) = delete;

    static void getDescrInfo(uint16_t nMaxUeGrps,
                             uint16_t nMaxTotalLayers,
                             size_t&  statDescrSizeBytes,
                             size_t&  statDescrAlignBytes,
                             size_t&  dynDescrSizeBytes,
                             size_t&  dynDescrAlignBytes,
                             size_t&  hetCfgUeGrpMapSizeBytes,
                             size_t&  hetCfgUeGrpMapAlignBytes,
                             size_t&  ueGrpPrmsSizeBytes,
                             size_t&  ueGrpPrmsAlignBytes,
                             size_t&  bfLayerPrmsSizeBytes,
                             size_t&  bfLayerPrmsAlignBytes);

    // initialize channel equalizer object and static component descriptor
    cuphyStatus_t init(bool         enableCpuToGpuDescrAsyncCpy,
                       uint8_t      compressBitwidth,
                       float        lambda,
                       void*        pStatDescrCpu,
                       void*        pStatDescrGpu,
                       void*        pDynDescrsCpu,
                       void*        pDynDescrsGpu,
                       void*        pHetCfgUeGrpMapCpu,
                       void*        pHetCfgUeGrpMapGpu,
                       void*        pUeGrpPrmsCpu,
                       void*        pUeGrpPrmsGpu,
                       void*        pBfLayerPrmsCpu,
                       void*        pBfLayerPrmsGpu,
                       cudaStream_t strm);

    // setup object state and dynamic component descriptor in prepration towards execution
    cuphyStatus_t setupCoefComp(uint16_t                      nUeGrps,
                                cuphyBfwUeGrpPrm_t const*     pUeGrpPrms,
                                bool                          enableCpuToGpuDescrAsyncCpy,
                                cuphySrsChEstBuffInfo_t*      pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                                cuphyTensorPrm_t*             pTBfwCoef,
#endif
                                uint8_t**                     pBfwCompCoef,
                                cuphyBfwCoefCompLaunchCfgs_t* pLaunchCfgs,
                                cudaStream_t                  strm);                              

private:
    void setupAndBatchCoefComp(uint16_t                  nUeGrps,
                               cuphyBfwUeGrpPrm_t const* pUeGrpPrms,
                               uint32_t&                 nHetCfgs,
                               cuphySrsChEstBuffInfo_t*  pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                               cuphyTensorPrm_t*         pTBfwCoef,
#endif
                               uint8_t**                 pBfwCompCoef);

    void setupUeGrpDynDescr(cuphyBfwUeGrpPrm_t const&      ueGrpPrm,
                            bfwCoefCompKernelUeGrpPrm_t&   kernelUeGrpPrm,
                            bfwCoefCompKernelBfLayerPrm_t* pKernelLayerPrmCpu,
                            bfwCoefCompKernelBfLayerPrm_t* pKernelLayerPrmGpu,
                            cuphySrsChEstBuffInfo_t*       pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                            cuphyTensorPrm_t*              pTBfwCoef,
#endif
                            uint8_t**                      pBfwCompCoef);


    void bfwCoefCompKernelSelL1(bool                         getKernelFuncOnly,
                                uint16_t                     nMaxPrbGrp,
                                uint16_t                     nUeGrps,
                                uint16_t                     nRxAnts,
                                uint8_t                      nLayers,
                                cuphyDataType_t              srsChEstType,
                                cuphyDataType_t              lambdaType,
#ifdef BFW_BOTH_COMP_FLOAT
                                cuphyDataType_t              coefType,
#endif
                                cuphyBfwCoefCompLaunchCfg_t& launchCfg);

    template <typename TStorageIn, typename TStorageOut, typename TCompute>
    void bfwCoefCompKernelSelL0(bool                         getKernelFuncOnly,
                                uint16_t                     nMaxPrbGrp,
                                uint16_t                     nUeGrps,
                                uint16_t                     nRxAnts,
                                uint8_t                      nLayers,                                
                                cuphyBfwCoefCompLaunchCfg_t& launchCfg);
                              
    template <typename TStorageIn,
              typename TStorageOut,
              typename TCompute,
              uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
              uint32_t N_LAYERS>  // # of layers (# of cols in H matrix)
    void bfwMmseCoefComp(bool                         getKernelFuncOnly,
                         uint16_t                     nMaxPrbGrp,
                         uint16_t                     nUeGrps,
                         cuphyBfwCoefCompLaunchCfg_t& launchCfg);

    template <uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
              uint32_t N_LAYERS,  // # of layers (# of cols in H matrix)
              uint32_t N_THRD_GRPS_PER_THRD_BLK,
              uint32_t N_THRDS_PER_GRP>
    void bfwMmseCoefCompKernelLaunchGeo(uint16_t nMaxPrbGrp,
                                        uint16_t nUeGrps,
                                        dim3&    gridDim,
                                        dim3&    blockDim);

    typedef struct _bfwCoefCompHetCfg
    {
        CUfunction func;
        uint16_t   nMaxPrbGrp; // Maximum number of PRB groups across all UE groups corresponding to this heterogenous config
        uint16_t   nUeGrps;    // Number of user groups corresponding to this heterogenous config
    } bfwCoefCompHetCfg_t;

    // Class state modified by setup saved in data members

    using bfwCoefCompHetCfgArr_t = std::array<bfwCoefCompHetCfg_t, CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS>;
    bfwCoefCompHetCfgArr_t m_coefCompHetCfgsArr{};
    uint16_t *m_pHetCfgUeGrpMapCpu = nullptr, *m_pHetCfgUeGrpMapGpu = nullptr;
    std::array<uint16_t*, CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS> m_pHetCfgUeGrpMapArr{};

    uint16_t m_nMaxTotalLayers = 0;
    uint16_t m_nMaxUeGrps = 0;

    // Kernel descriptor pointers
    bfwCoefCompStatDescr_t *m_pStatDescrCpu = nullptr, *m_pStatDescrGpu = nullptr; 
    bfwCoefCompDynDescr_t *m_pDynDescrCpu = nullptr, *m_pDynDescrGpu = nullptr;

    bfwCoefCompKernelUeGrpPrm_t *m_pKernelUeGrpPrmCpu = nullptr, *m_pKernelUeGrpPrmGpu = nullptr;
    bfwCoefCompKernelBfLayerPrm_t *m_pKernelBfLayerPrmCpu = nullptr, *m_pKernelBfLayerPrmGpu = nullptr;

    // Channel estimator kernel arguments (supplied via descriptors)
    typedef struct _bfwCoefCompKernelArgs
    {
        bfwCoefCompStatDescr_t* pStatDescr; // pointer to static descriptor
        bfwCoefCompDynDescr_t*  pDynDescr;  // pointer to dynamic descriptor
    } bfwCoefCompKernelArgs_t;

    using bfwCoefCompKernelArgsArr_t  = std::array<bfwCoefCompKernelArgs_t, CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS>;
    bfwCoefCompKernelArgsArr_t m_coefCompKernelArgsArr{};
};

} // namespace bfw_coefComp

#endif // !defined(BFC_HPP_INCLUDED_)
