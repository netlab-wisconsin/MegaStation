/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuphy.h"
#include "tensor_desc.hpp"
#include "cuphy_api.h"

#define MAX_N_ANT_PORTS (4)
#define MAX_N_SYM (4)
#define MAX_N_HOPS (4)
#define MAX_N_REPS (4)
#define MAX_N_COMB_PER_UE (2)


#define MAX_N_SRS_CELL (24) 
#define MAX_N_SRS_UE (MAX_N_SRS_CELL * 8) // MAX_N_SRS_CELL * nUEs_perCell (e.g., 8 as in CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP)
#define MAX_N_COMP_BLOCKS (MAX_N_SRS_UE * 68) // MAX_N_SRS_UE * nComputeBlocks_perUe (e.g. 272/4)
// nComputeBlocks_perUe = nHops * nPrbsPerHop / 4 * nCombs

#define N_SYM_PER_SLOT 14
#define N_SC_PER_PRB 12
#define N_PRIMES 303
#define MAX_N_SC 24
#define MAX_N_ANT_PORT 4
#define FOCC_LENGTH 4
#define N_PRB_PER_COMP_BLK 4
#define PRB_GRP_SIZE 2
#define N_GRP_PER_COMP_BLK 2

#define POINT_ONE_PERCENT 0.001

struct cuphySrsChEst0
{};


// Channel estimator static descriptor
struct srsChEst0StatDescr_t
{
    tensor_ref_any<CUPHY_C_16F>  tFocc_table;

    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts1_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts2_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts4_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts1_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts2_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts4_wide;

    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts1_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts2_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts4_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts1_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts2_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts4_narrow;

    float noisEstDebias_comb2_nPorts1;
    float noisEstDebias_comb2_nPorts2;
    float noisEstDebias_comb2_nPorts4;
    float noisEstDebias_comb4_nPorts1;
    float noisEstDebias_comb4_nPorts2;
    float noisEstDebias_comb4_nPorts4;
};


struct cellDescr_t{
    uint8_t                      mu;
    uint16_t                     nRxAntSrs;
    tensor_ref_any<CUPHY_C_16F>  tDataRx;
};

struct __align__(32) compBlockDescr_t{
    uint16_t    ueIdx;
    uint8_t     combIdx;
    uint8_t     hopIdx;
    uint16_t    blockStartPrb;
};

struct ueDescr_t{
        // compute parameters:
        uint8_t  repSymIdxs[MAX_N_HOPS][MAX_N_REPS];
        uint16_t hopStartPrbs[MAX_N_HOPS];
        uint8_t  nRepPerHop[MAX_N_HOPS]; 
        uint16_t nPrbsPerHop;
        uint8_t  u[MAX_N_SYM];
        float    q[MAX_N_SYM];
        float    alphaCommon;
        uint8_t  lowPaprTableIdx;
        uint16_t lowPaprPrime;
        uint8_t  nPortsPerComb;
        uint8_t  portToFoccMap[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS];
        uint8_t  combSize; 
        uint8_t  combOffsets[MAX_N_COMB_PER_UE];
        uint8_t  nCombScPerPrb; 
        uint8_t  portToUeAntMap[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS];
        uint8_t  portToL2OutUeAntMap[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS];
        uint8_t  cellIdx;

        // ue group parameters:
        uint32_t ueBlockCntr;
        uint32_t ueNumBlocks;

        // temp wideband report
        float   tmpWidebandNoiseEnergy  = 0;
        float   tmpWidebandSignalEnergy = 0;
        __half2 tmpWidebandScCorr       = __floats2half2_rn(0.f, 0.f);

        // output buffers:
        float*                       pUeRbSnr;
        cuphySrsReport_t*            pUeSrsReport;
        tensor_ref_any<CUPHY_C_32F>  tChEstBuff;        //ToDo considering it stores half float output, should we change it to CUPHY_C_16F?
        tensor_ref_any<CUPHY_C_32F>  tChEstToL2;        //ToDo considering it stores half float output, should we change it to CUPHY_C_16F?
        uint16_t                     chEstBuffStartPrbGrp;
};



struct srsChEst0DynDescr_t{
    ueDescr_t        ueDescrs[MAX_N_SRS_UE];
    cellDescr_t      cellDescrs[MAX_N_SRS_CELL];
    compBlockDescr_t compBlockDescrs[MAX_N_COMP_BLOCKS];
    int              nSrsUes;
};


//  srsChEst kernel arguments (Supplied via descriptors)
struct srsChEst0KernelArgs_t
{
    srsChEst0StatDescr_t* pStatDescr; 
    srsChEst0DynDescr_t*  pDynDescr;
};

class srsChEst0 : public cuphySrsChEst0
{
public:
    srsChEst0();
    ~srsChEst0()                           = default;
    srsChEst0(srsChEst0 const&)            = delete;
    srsChEst0& operator=(srsChEst0 const&) = delete;

    void init(cuphySrsFilterPrms_t* pSrsFilterPrms,
              bool                  enableCpuToGpuDescrAsyncCpy,
              srsChEst0StatDescr_t* pCpuStatDesc,
              void*                 pGpuStatDesc,
              cudaStream_t          strm);

    

    cuphyStatus_t setup( uint16_t                      nSrsUes,
                        cuphyUeSrsPrm_t*              h_srsUePrms,
                        uint16_t                      nCells,
                        cuphyTensorPrm_t*             pTDataRx, 
                        cuphySrsCellPrms_t*           h_srsCellPrms,
                        float*                        d_rbSnrBuff,
                        uint32_t*                     h_rbSnrBuffOffsets,
                        cuphySrsReport_t*             d_pSrsReports,
                        cuphySrsChEstBuffInfo_t*      h_chEstBuffInfo,
                        void**                        d_addrsChEstToL2Buff,
                        cuphySrsChEstToL2_t*          h_chEstToL2,
                        bool                          enableCpuToGpuDescrAsyncCpy,
                        srsChEst0DynDescr_t*          pCpuDynDesc,
                        void*                         pGpuDynDesc,
                        cuphySrsChEst0LaunchCfg_t*    pLaunchCfg,
                        cudaStream_t                  strm);

    void kernelSelect(srsChEst0DynDescr_t* pCpuDynDesc, uint16_t nSrsUes, uint16_t nCompBlocks, cuphySrsChEst0LaunchCfg_t* pLaunchCfg);

    static void getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    srsChEst0KernelArgs_t m_kernelArgs;

};
