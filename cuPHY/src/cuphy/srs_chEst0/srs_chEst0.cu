/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "srs_chEst0.hpp"
#include "goldSequenceHostSrs.cpp"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "math_utils.cuh"
#include "cuda_fp16.h"
#include <assert.h>

namespace cg = cooperative_groups;
#define USE_SRS_ATOMIC_REDUCTION 1 // flag used to choose between 2-stage cg::reduction vs using atomic combined with cg::reduction

static constexpr uint32_t LOWER_BYTE_BMSK = 255;

constexpr uint16_t PRIMES_TABLE[303] =
   {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999};

constexpr uint16_t SRS_BW_TABLE[64][8] =
   {{4,1,4,1,4,1,4,1},
    {8,1,4,2,4,1,4,1},
    {12,1,4,3,4,1,4,1},
    {16,1,4,4,4,1,4,1},
    {16,1,8,2,4,2,4,1},
    {20,1,4,5,4,1,4,1},
    {24,1,4,6,4,1,4,1},
    {24,1,12,2,4,3,4,1},
    {28,1,4,7,4,1,4,1},
    {32,1,16,2,8,2,4,2},
    {36,1,12,3,4,3,4,1},
    {40,1,20,2,4,5,4,1},
    {48,1,16,3,8,2,4,2},
    {48,1,24,2,12,2,4,3},
    {52,1,4,13,4,1,4,1},
    {56,1,28,2,4,7,4,1},
    {60,1,20,3,4,5,4,1},
    {64,1,32,2,16,2,4,4},
    {72,1,24,3,12,2,4,3},
    {72,1,36,2,12,3,4,3},
    {76,1,4,19,4,1,4,1},
    {80,1,40,2,20,2,4,5},
    {88,1,44,2,4,11,4,1},
    {96,1,32,3,16,2,4,4},
    {96,1,48,2,24,2,4,6},
    {104,1,52,2,4,13,4,1},
    {112,1,56,2,28,2,4,7},
    {120,1,60,2,20,3,4,5},
    {120,1,40,3,8,5,4,2},
    {120,1,24,5,12,2,4,3},
    {128,1,64,2,32,2,4,8},
    {128,1,64,2,16,4,4,4},
    {128,1,16,8,8,2,4,2},
    {132,1,44,3,4,11,4,1},
    {136,1,68,2,4,17,4,1},
    {144,1,72,2,36,2,4,9},
    {144,1,48,3,24,2,12,2},
    {144,1,48,3,16,3,4,4},
    {144,1,16,9,8,2,4,2},
    {152,1,76,2,4,19,4,1},
    {160,1,80,2,40,2,4,10},
    {160,1,80,2,20,4,4,5},
    {160,1,32,5,16,2,4,4},
    {168,1,84,2,28,3,4,7},
    {176,1,88,2,44,2,4,11},
    {184,1,92,2,4,23,4,1},
    {192,1,96,2,48,2,4,12},
    {192,1,96,2,24,4,4,6},
    {192,1,64,3,16,4,4,4},
    {192,1,24,8,8,3,4,2},
    {208,1,104,2,52,2,4,13},
    {216,1,108,2,36,3,4,9},
    {224,1,112,2,56,2,4,14},
    {240,1,120,2,60,2,4,15},
    {240,1,80,3,20,4,4,5},
    {240,1,48,5,16,3,8,2},
    {240,1,24,10,12,2,4,3},
    {256,1,128,2,64,2,4,16},
    {256,1,128,2,32,4,4,8},
    {256,1,16,16,8,2,4,2},
    {264,1,132,2,44,3,4,11},
    {272,1,136,2,68,2,4,17},
    {272,1,68,4,4,17,4,1},
    {272,1,16,17,8,2,4,2}};

#define LOW_PAPR_TABLE_0_N_ROWS 30
#define LOW_PAPR_TABLE_0_N_COLS 12
static __device__ __constant__  int8_t LOW_PAPR_TABLE_0[LOW_PAPR_TABLE_0_N_ROWS][LOW_PAPR_TABLE_0_N_COLS] =
   {{-3,1,-3,-3,-3,3,-3,-1,1,1,1,-3},
    {-3,3,1,-3,1,3,-1,-1,1,3,3,3},
    {-3,3,3,1,-3,3,-1,1,3,-3,3,-3},
    {-3,-3,-1,3,3,3,-3,3,-3,1,-1,-3},
    {-3,-1,-1,1,3,1,1,-1,1,-1,-3,1},
    {-3,-3,3,1,-3,-3,-3,-1,3,-1,1,3},
    {1,-1,3,-1,-1,-1,-3,-1,1,1,1,-3},
    {-1,-3,3,-1,-3,-3,-3,-1,1,-1,1,-3},
    {-3,-1,3,1,-3,-1,-3,3,1,3,3,1},
    {-3,-1,-1,-3,-3,-1,-3,3,1,3,-1,-3},
    {-3,3,-3,3,3,-3,-1,-1,3,3,1,-3},
    {-3,-1,-3,-1,-1,-3,3,3,-1,-1,1,-3},
    {-3,-1,3,-3,-3,-1,-3,1,-1,-3,3,3},
    {-3,1,-1,-1,3,3,-3,-1,-1,-3,-1,-3},
    {1,3,-3,1,3,3,3,1,-1,1,-1,3},
    {-3,1,3,-1,-1,-3,-3,-1,-1,3,1,-3},
    {-1,-1,-1,-1,1,-3,-1,3,3,-1,-3,1},
    {-1,1,1,-1,1,3,3,-1,-1,-3,1,-3},
    {-3,1,3,3,-1,-1,-3,3,3,-3,3,-3},
    {-3,-3,3,-3,-1,3,3,3,-1,-3,1,-3},
    {3,1,3,1,3,-3,-1,1,3,1,-1,-3},
    {-3,3,1,3,-3,1,1,1,1,3,-3,3},
    {-3,3,3,3,-1,-3,-3,-1,-3,1,3,-3},
    {3,-1,-3,3,-3,-1,3,3,3,-3,-1,-3},
    {-3,-1,1,-3,1,3,3,3,-1,-3,3,3},
    {-3,3,1,-1,3,3,-3,1,-1,1,-1,1},
    {-1,1,3,-3,1,-1,1,-1,-1,-3,1,-1},
    {-3,-3,3,3,3,-3,-1,1,-3,3,1,-3},
    {1,-1,3,1,1,-1,-1,-1,1,3,-3,1},
    {-3,3,-3,3,-3,-3,3,-1,-1,1,3,-3}};

#define LOW_PAPR_TABLE_1_N_ROWS 30
#define LOW_PAPR_TABLE_1_N_COLS 24
static __device__ __constant__  int8_t LOW_PAPR_TABLE_1[LOW_PAPR_TABLE_1_N_ROWS][LOW_PAPR_TABLE_1_N_COLS] =
   {{-1,-3,3,-1,3,1,3,-1,1,-3,-1,-3,-1,1,3,-3,-1,-3,3,3,3,-3,-3,-3},
    {-1,-3,3,1,1,-3,1,-3,-3,1,-3,-1,-1,3,-3,3,3,3,-3,1,3,3,-3,-3},
    {-1,-3,-3,1,-1,-1,-3,1,3,-1,-3,-1,-1,-3,1,1,3,1,-3,-1,-1,3,-3,-3},
    {1,-3,3,-1,-3,-1,3,3,1,-1,1,1,3,-3,-1,-3,-3,-3,-1,3,-3,-1,-3,-3},
    {-1,3,-3,-3,-1,3,-1,-1,1,3,1,3,-1,-1,-3,1,3,1,-1,-3,1,-1,-3,-3},
    {-3,-1,1,-3,-3,1,1,-3,3,-1,-1,-3,1,3,1,-1,-3,-1,-3,1,-3,-3,-3,-3},
    {-3,3,1,3,-1,1,-3,1,-3,1,-1,-3,-1,-3,-3,-3,-3,-1,-1,-1,1,1,-3,-3},
    {-3,1,3,-1,1,-1,3,-3,3,-1,-3,-1,-3,3,-1,-1,-1,-3,-1,-1,-3,3,3,-3},
    {-3,1,-3,3,-1,-1,-1,-3,3,1,-1,-3,-1,1,3,-1,1,-1,1,-3,-3,-3,-3,-3},
    {1,1,-1,-3,-1,1,1,-3,1,-1,1,-3,3,-3,-3,3,-1,-3,1,3,-3,1,-3,-3},
    {-3,-3,-3,-1,3,-3,3,1,3,1,-3,-1,-1,-3,1,1,3,1,-1,-3,3,1,3,-3},
    {-3,3,-1,3,1,-1,-1,-1,3,3,1,1,1,3,3,1,-3,-3,-1,1,-3,1,3,-3},
    {3,-3,3,-1,-3,1,3,1,-1,-1,-3,-1,3,-3,3,-1,-1,3,3,-3,-3,3,-3,-3},
    {-3,3,-1,3,-1,3,3,1,1,-3,1,3,-3,3,-3,-3,-1,1,3,-3,-1,-1,-3,-3},
    {-3,1,-3,-1,-1,3,1,3,-3,1,-1,3,3,-1,-3,3,-3,-1,-1,-3,-3,-3,3,-3},
    {-3,-1,-1,-3,1,-3,-3,-1,-1,3,-1,1,-1,3,1,-3,-1,3,1,1,-1,-1,-3,-3},
    {-3,-3,1,-1,3,3,-3,-1,1,-1,-1,1,1,-1,-1,3,-3,1,-3,1,-1,-1,-1,-3},
    {3,-1,3,-1,1,-3,1,1,-3,-3,3,-3,-1,-1,-1,-1,-1,-3,-3,-1,1,1,-3,-3},
    {-3,1,-3,1,-3,-3,1,-3,1,-3,-3,-3,-3,-3,1,-3,-3,1,1,-3,1,1,-3,-3},
    {-3,-3,3,3,1,-1,-1,-1,1,-3,-1,1,-1,3,-3,-1,-3,-1,-1,1,-3,3,-1,-3},
    {-3,-3,-1,-1,-1,-3,1,-1,-3,-1,3,-3,1,-3,3,-3,3,3,1,-1,-1,1,-3,-3},
    {3,-1,1,-1,3,-3,1,1,3,-1,-3,3,1,-3,3,-1,-1,-1,-1,1,-3,-3,-3,-3},
    {-3,1,-3,3,-3,1,-3,3,1,-1,-3,-1,-3,-3,-3,-3,1,3,-1,1,3,3,3,-3},
    {-3,-1,1,-3,-1,-1,1,1,1,3,3,-1,1,-1,1,-1,-1,-3,-3,-3,3,1,-1,-3},
    {-3,3,-1,-3,-1,-1,-1,3,-1,-1,3,-3,-1,3,-3,3,-3,-1,3,1,1,-1,-3,-3},
    {-3,1,-1,-3,-3,-1,1,-3,-1,-3,1,1,-1,1,1,3,3,3,-1,1,-1,1,-1,-3},
    {-1,3,-1,-1,3,3,-1,-1,-1,3,-1,-3,1,3,1,1,-3,-3,-3,-1,-3,-1,-3,-3},
    {3,-3,-3,-1,3,3,-3,-1,3,1,1,1,3,-1,3,-3,-1,3,-1,3,1,-1,-3,-3},
    {-3,1,-3,1,-3,1,1,3,1,-3,-3,-1,1,3,-1,-3,3,1,-1,-3,-3,-3,-3,-3},
    {3,-3,-1,1,3,-1,-1,-3,-1,3,-1,-3,-1,-3,3,-1,3,1,1,-3,3,-3,-3,-3}};

// NOTE: not sure how to make C_FP16 constant table, so for now fOCC_table read in static descriptors. (Probably better as a constant table)
// static __device__ __constant__  __half2 FOCC_TABLE[4][4] =
// {{{1.000000,0.000000},{1.000000,0.000000},{1.000000,0.000000},{1.000000,0.000000}},
// {{1.000000,0.000000},{0.000000,-1.000000},{-1.000000,0.000000},{0.000000,1.000000}},
// {{1.000000,0.000000},{-1.000000,0.000000},{1.000000,0.000000},{-1.000000,0.000000}},
// {{1.000000,0.000000},{0.000000,1.000000},{-1.000000,0.000000},{0.000000,-1.000000}}};

 //ToDo: we may consider defining number of antennas as a template parameter and adjust launch bound parameters
 // according to CTA size to avoid excessive register spills when it could be avoided
 template <int combSize, int nPortsPerComb>
__device__ void srsChEst0KernelInner(srsChEst0StatDescr_t* pStatDescr, srsChEst0DynDescr_t* pDynDescr)
{
   cg::thread_block thisThrdBlk = cg::this_thread_block();
   int              tid         = thisThrdBlk.thread_rank();
   cg::thread_block_tile<32> tile = cg::tiled_partition<32>(thisThrdBlk);

   // filter parameters:
   tensor_ref_any<CUPHY_C_16F>& tFocc_table = pStatDescr->tFocc_table;

   tensor_ref_any<CUPHY_C_16F>& tW_comb2_nPorts1_wide = pStatDescr->tW_comb2_nPorts1_wide;
   tensor_ref_any<CUPHY_C_16F>& tW_comb2_nPorts2_wide = pStatDescr->tW_comb2_nPorts2_wide;
   tensor_ref_any<CUPHY_C_16F>& tW_comb2_nPorts4_wide = pStatDescr->tW_comb2_nPorts4_wide;
   tensor_ref_any<CUPHY_C_16F>& tW_comb4_nPorts1_wide = pStatDescr->tW_comb4_nPorts1_wide;
   tensor_ref_any<CUPHY_C_16F>& tW_comb4_nPorts2_wide = pStatDescr->tW_comb4_nPorts2_wide;
   tensor_ref_any<CUPHY_C_16F>& tW_comb4_nPorts4_wide = pStatDescr->tW_comb4_nPorts4_wide;

   tensor_ref_any<CUPHY_C_16F>& tW_comb2_nPorts1_narrow = pStatDescr->tW_comb2_nPorts1_narrow;
   tensor_ref_any<CUPHY_C_16F>& tW_comb2_nPorts2_narrow = pStatDescr->tW_comb2_nPorts2_narrow;
   tensor_ref_any<CUPHY_C_16F>& tW_comb2_nPorts4_narrow = pStatDescr->tW_comb2_nPorts4_narrow;
   tensor_ref_any<CUPHY_C_16F>& tW_comb4_nPorts1_narrow = pStatDescr->tW_comb4_nPorts1_narrow;
   tensor_ref_any<CUPHY_C_16F>& tW_comb4_nPorts2_narrow = pStatDescr->tW_comb4_nPorts2_narrow;
   tensor_ref_any<CUPHY_C_16F>& tW_comb4_nPorts4_narrow = pStatDescr->tW_comb4_nPorts4_narrow;

   float& noisEstDebias_comb2_nPorts1 = pStatDescr->noisEstDebias_comb2_nPorts1;
   float& noisEstDebias_comb2_nPorts2 = pStatDescr->noisEstDebias_comb2_nPorts2;
   float& noisEstDebias_comb2_nPorts4 = pStatDescr->noisEstDebias_comb2_nPorts4;
   float& noisEstDebias_comb4_nPorts1 = pStatDescr->noisEstDebias_comb4_nPorts1;
   float& noisEstDebias_comb4_nPorts2 = pStatDescr->noisEstDebias_comb4_nPorts2;
   float& noisEstDebias_comb4_nPorts4 = pStatDescr->noisEstDebias_comb4_nPorts4;

   // comp block parameters:
   const uint32_t    compBlockIdx   = blockIdx.x;
   compBlockDescr_t& compBlockDescr = pDynDescr->compBlockDescrs[compBlockIdx];
   uint16_t          ueIdx          = compBlockDescr.ueIdx;
   uint8_t           combIdx        = compBlockDescr.combIdx;
   uint8_t           hopIdx         = compBlockDescr.hopIdx;
   uint16_t          blockStartPrb  = compBlockDescr.blockStartPrb;


   // user parameters:
   ueDescr_t& ueDescr                                = pDynDescr->ueDescrs[ueIdx];
   uint8_t(&repSymIdxs)[MAX_N_REPS]                  = ueDescr.repSymIdxs[hopIdx];
   uint8_t ueStartSym                                = ueDescr.repSymIdxs[0][0];
   uint16_t hopStartPrb                              = ueDescr.hopStartPrbs[hopIdx];
   uint8_t  nRepPerHop                               = ueDescr.nRepPerHop[hopIdx];
   uint16_t nPrbsPerHop                              = ueDescr.nPrbsPerHop;
   uint8_t(&u)[MAX_N_SYM]                            = ueDescr.u;
   float(&q)[MAX_N_SYM]                              = ueDescr.q;
   float    alphaCommon                              = ueDescr.alphaCommon;
   uint8_t  lowPaprTableIdx                          = ueDescr.lowPaprTableIdx;
   uint16_t lowPaprPrime                             = ueDescr.lowPaprPrime;
   uint8_t(&portToFoccMap)[MAX_N_ANT_PORTS]          = ueDescr.portToFoccMap[combIdx];
   uint8_t combOffset                                = ueDescr.combOffsets[combIdx];
   constexpr uint8_t nCombScPerPrb                   = (combSize == 2) ? 6 : 3;
   uint8_t(&portToUeAntMap)[MAX_N_ANT_PORTS]         = ueDescr.portToUeAntMap[combIdx];
   uint8_t(&portToL2OutUeAntMap)[MAX_N_ANT_PORTS]    = ueDescr.portToL2OutUeAntMap[combIdx];
   float*                       pUeRbSnr             = ueDescr.pUeRbSnr;
   uint8_t                      cellIdx              = ueDescr.cellIdx;
   cuphySrsReport_t*&           pUeSrsReport         = ueDescr.pUeSrsReport;
   float&                       widebandSnr          = pUeSrsReport->widebandSnr;
   float&                       toEstMicroSec        = pUeSrsReport->toEstMicroSec;
   float&                       widebandNoiseEnergy  = pUeSrsReport->widebandNoiseEnergy;
   float&                       widebandSignalEnergy = pUeSrsReport->widebandSignalEnergy;
   __half2&                     widebandScCorr       = pUeSrsReport->widebandScCorr;
   volatile float&      tmpWidebandNoiseEnergy       = ueDescr.tmpWidebandNoiseEnergy;
   volatile float&      tmpWidebandSignalEnergy      = ueDescr.tmpWidebandSignalEnergy;
   volatile __half2&    tmpWidebandScCorr            = ueDescr.tmpWidebandScCorr;
   tensor_ref_any<CUPHY_C_32F>& tChEstBuff           = ueDescr.tChEstBuff;
   uint16_t                     chEstBuffStartPrbGrp = ueDescr.chEstBuffStartPrbGrp;
   tensor_ref_any<CUPHY_C_32F>& tChEstToL2           = ueDescr.tChEstToL2;


   // ue grouping params
   uint32_t& ueBlockCntr   = ueDescr.ueBlockCntr;
   const int ueNumBlocks   = ueDescr.ueNumBlocks;

   // cell parameters:
   cellDescr_t&                cellDescr = pDynDescr->cellDescrs[cellIdx];
   uint8_t                     mu        = cellDescr.mu;
   uint16_t                    nRxAntSrs = cellDescr.nRxAntSrs;
   tensor_ref_any<CUPHY_C_16F> tDataRx   = cellDescr.tDataRx;

   // Setup: pick ChEst filter
   tensor_ref_any<CUPHY_C_16F>* pW_wide        = nullptr;
   tensor_ref_any<CUPHY_C_16F>* pW_narrow      = nullptr;
   float                        noiseEstDebias = 0;

   if constexpr (combSize == 2)
   {
       if constexpr (nPortsPerComb == 1)
       {
           pW_wide        = &tW_comb2_nPorts1_wide;
           pW_narrow      = &tW_comb2_nPorts1_narrow;
           noiseEstDebias = noisEstDebias_comb2_nPorts1;
       }
       else if constexpr (nPortsPerComb == 2)
       {
           pW_wide        = &tW_comb2_nPorts2_wide;
           pW_narrow      = &tW_comb2_nPorts2_narrow;
           noiseEstDebias = noisEstDebias_comb2_nPorts2;
       }
       else if constexpr (nPortsPerComb == 4)
       {
           pW_wide        = &tW_comb2_nPorts4_wide;
           pW_narrow      = &tW_comb2_nPorts4_narrow;
           noiseEstDebias = noisEstDebias_comb2_nPorts4;
       }
   }
   else if constexpr (combSize == 4)
   {
       if constexpr (nPortsPerComb == 1)
       {
           pW_wide        = &tW_comb4_nPorts1_wide;
           pW_narrow      = &tW_comb4_nPorts1_narrow;
           noiseEstDebias = noisEstDebias_comb4_nPorts1;
       }
       else if constexpr (nPortsPerComb == 2)
       {
           pW_wide        = &tW_comb4_nPorts2_wide;
           pW_narrow      = &tW_comb4_nPorts2_narrow;
           noiseEstDebias = noisEstDebias_comb4_nPorts2;
       }
       else if constexpr (nPortsPerComb == 4)
       {
           pW_wide        = &tW_comb4_nPorts4_wide;
           pW_narrow      = &tW_comb4_nPorts4_narrow;
           noiseEstDebias = noisEstDebias_comb4_nPorts4;
       }
   }

   constexpr uint8_t nSrsScBlock = combSize == 4 ? 12 : 24;

   // shared memory assignments: temporary storage for srs computation
   __shared__ extern __half2 sh_buff[];

   __half2* sh_rxSrs   = sh_buff;                                            //size nRxAntSrs * nSrsScBlock
   __half2* sh_Hest    = &sh_rxSrs[nRxAntSrs * nSrsScBlock];                 //size nRxAntSrs * nSrsScBlock * nPortsPerComb
   __half2* sh_avgHest = &sh_Hest[nRxAntSrs * nSrsScBlock * nPortsPerComb];  //size nRxAntSrs * N_GRP_PER_COMP_BLK * nPortsPerComb
   __half2* sh_W_matrix = &sh_avgHest[nRxAntSrs * N_GRP_PER_COMP_BLK * nPortsPerComb]; // size nSrsScBlock * nSrsScBlock

   // shared memory for step 3
   __shared__ half2 sh_avgScCorr;
   sh_avgScCorr = half2(0, 0);
   // shared memory for step 6
   __shared__ float avgSignalEnergyPrb[N_PRB_PER_COMP_BLK];
   __shared__ float avgSignalEnergySc[nSrsScBlock];
   __shared__ float avgNoiseEnergy;
   __shared__ float avgSignalEnergy;
   __shared__ __half2 sh_focc_table[FOCC_LENGTH*FOCC_LENGTH];
   assert(thisThrdBlk.size() > N_PRB_PER_COMP_BLK);
   for (int i = tid; i < nSrsScBlock; i += thisThrdBlk.size()) {
      avgSignalEnergySc[i] = 0.0f;
      // avgSignalEnergyPrb is initialized below by the thread
      // responsible for accumulating avgSignalEnergySc entries into avgSignalEnergyPrb
   }

   if (tid == 0) {
      avgNoiseEnergy = 0.0f;
      avgSignalEnergy = 0.0f;
   }

   for (int i = thisThrdBlk.size()-1-tid; i < FOCC_LENGTH*FOCC_LENGTH; i += thisThrdBlk.size()) {
      const int row = i/FOCC_LENGTH;
      const int col = i - row * FOCC_LENGTH;
      sh_focc_table[i] = tFocc_table({row, col});
   }
   //thisThrdBlk.sync(); // this can be commented since before using the initialized shared mem above, there is another block sync barrier
   //------------------------------------------------------------------------------------------------------------------------

   // STEP 1: Load Rx SRS subcarriers, remove ZC cover-code, average repetitions
   // flatten nested loop over nRxAntSrs -> nSrsScBlocks
   int max_loop_iters = nRxAntSrs * nSrsScBlock;
   for(int i = tid; i < max_loop_iters; i += thisThrdBlk.size())
   {
       int scIdx  = i % nSrsScBlock;
       int antIdx = i / nSrsScBlock;
       __half2 srs = half2(0,0);

       int ZcScIdx   = scIdx + nCombScPerPrb * (blockStartPrb - hopStartPrb);
       int loadScIdx = N_SC_PER_PRB * blockStartPrb + scIdx * combSize + combOffset;

       for(int repIdx = 0; repIdx < nRepPerHop; repIdx++)
       {
           int symIdx = repSymIdxs[repIdx];

           // extract subcarrier for this repetition:
           __half2 y = tDataRx({loadScIdx, symIdx, antIdx});

           // compute subcarrier ZC coverCode for this repetition:
           float2 r = {0, 0}; //ToDo: check impact of sincospif instead of __sincosf on perf, if not significant use that instead
           if(lowPaprTableIdx == 0)
           {
               auto u_repIdx = u[symIdx - ueStartSym];
               __sincosf(((M_PI * LOW_PAPR_TABLE_0[u_repIdx][ZcScIdx]) / 4.0f + alphaCommon * scIdx), &r.y, &r.x);
           }
           else if(lowPaprTableIdx == 1)
           {
               auto u_repIdx = u[symIdx - ueStartSym];
               __sincosf(((M_PI * LOW_PAPR_TABLE_1[u_repIdx][ZcScIdx]) / 4.0f + alphaCommon * scIdx), &r.y, &r.x);
           }
           else
           {
               // to improve precision with large args, we use the following block to compute:
               //__sincosf(((-M_PI * q_repIdx * m * (m + 1)) / static_cast<float>(lowPaprPrime) + alphaCommon * scIdx), &r.y, &r.x);
               uint32_t q_repIdx = static_cast<uint32_t>(q[symIdx - ueStartSym]);
               uint32_t m        =  ZcScIdx % lowPaprPrime;

               uint32_t primeRemainder = (q_repIdx * m * (m + 1)) % lowPaprPrime;
               uint32_t primeDivisor   = (q_repIdx * m * (m + 1)) / lowPaprPrime;

               float halfCycleFlag = 0;
               if((primeDivisor % 2) == 1)
               {
                   halfCycleFlag = 1;
               }

               __sincosf(-M_PI * (static_cast<float>(primeRemainder) / static_cast<float>(lowPaprPrime) + halfCycleFlag) + alphaCommon * scIdx, &r.y, &r.x);
           }

           // remove ZC coverCode and add :
           srs = __hadd2(srs, complex_conjmul(y, __float22half2_rn(r)));
       }
       // normalize :
       sh_rxSrs[i].x = srs.x / static_cast<__half>(nRepPerHop);
       sh_rxSrs[i].y = srs.y / static_cast<__half>(nRepPerHop);
   }

   for (int i = thisThrdBlk.size()-1-tid; i < nSrsScBlock*nSrsScBlock; i += thisThrdBlk.size()) {
       const int row = i / nSrsScBlock;
       const int col = i - row*nSrsScBlock;
       sh_W_matrix[i] = (*pW_wide)({row, col});
   }

   thisThrdBlk.sync();        //=============================================================================

   // STEP 2: remove cyclic shifts and apply wide filter to estimate channel
   // flatten nested loop over nRxAntSrs -> nSrsScBlocks -> nPortsPerComb
   // note: intermediate computations in different steps are performed in half precision, if not good enough, may consider using float
   max_loop_iters = nRxAntSrs * nSrsScBlock * nPortsPerComb;
   for(int i = tid; i < max_loop_iters; i += thisThrdBlk.size())
   {
       int portIdx = i % nPortsPerComb;
       int scIdx   = (i / nPortsPerComb) % nSrsScBlock;
       int antIdx  = i / (nPortsPerComb * nSrsScBlock);
       int foccIdx = portToFoccMap[portIdx];

       const __half2 *srs = sh_rxSrs + antIdx * nSrsScBlock;
       const __half2 *w = sh_W_matrix + scIdx*nSrsScBlock;
       const __half2 *foccBase = sh_focc_table + foccIdx*FOCC_LENGTH;

       auto est  = half2{0, 0};
       for(int inputScIdx = 0; inputScIdx < nSrsScBlock; inputScIdx++)
       {
           const auto focc = foccBase[inputScIdx % FOCC_LENGTH];
           est        = __hadd2(est, complex_mul(complex_conjmul(*w, focc), *srs));
           w++;
           srs++;
       }
       sh_Hest[i] = est;
   }
   thisThrdBlk.sync();        //=============================================================================

   // STEP 3: estimate delay phase ramp
   __half2 sumScCorr = {0, 0};

   max_loop_iters = nRxAntSrs * (nSrsScBlock - 1) * nPortsPerComb;
   for(int i = tid; i < max_loop_iters; i += thisThrdBlk.size())
   {
       int  portIdx = i % nPortsPerComb;
       int  scIdx   = (i / nPortsPerComb) % (nSrsScBlock - 1);
       int  antIdx  = i / (nPortsPerComb * (nSrsScBlock - 1));
       auto est0    = sh_Hest[portIdx + nPortsPerComb * scIdx + nPortsPerComb * nSrsScBlock * antIdx];
       auto est1    = sh_Hest[portIdx + nPortsPerComb * (scIdx + 1) + nPortsPerComb * nSrsScBlock * antIdx];
       sumScCorr    = __hadd2(sumScCorr, complex_conjmul(est1, est0));
   }

   __half2 tile_avgScCorr = cg::reduce(tile, sumScCorr, cg::plus<__half2>());
   tile_avgScCorr.x /= nRxAntSrs * (nSrsScBlock - 1) * nPortsPerComb;
   tile_avgScCorr.y /= nRxAntSrs * (nSrsScBlock - 1) * nPortsPerComb;
#if USE_SRS_ATOMIC_REDUCTION
   if (tile.thread_rank() == 0) {
       atomicAdd(&sh_avgScCorr, tile_avgScCorr);
   }
   thisThrdBlk.sync();
#else
   __shared__ half2 sh_avgScCorrPerTile[32]; // for reduce sum
   if (tile.thread_rank() == 0) {
       sh_avgScCorrPerTile[tile.meta_group_rank()] = tile_avgScCorr;
   }
   thisThrdBlk.sync();
   if (tile.meta_group_rank()==0)
   {
       tile_avgScCorr = tile.thread_rank() < tile.meta_group_size() ? sh_avgScCorrPerTile[tile.thread_rank()] : __half2{0, 0};
       sh_avgScCorr   = cg::reduce(tile, tile_avgScCorr, cg::plus<__half2>());
   }
   thisThrdBlk.sync();
#endif

   const float phaseRamp = atanf(__half2float(sh_avgScCorr.y) / __half2float(sh_avgScCorr.x)) / combSize;

   // STEP 4: remove delay phase ramp from rx signal
   max_loop_iters = nRxAntSrs * nSrsScBlock;
   for(int i = tid; i < max_loop_iters; i += thisThrdBlk.size())
   {
       int    scIdx        = i % nSrsScBlock;
       int    scIdx_global = scIdx * combSize;
       float2 phase_conj;
       __sincosf(-phaseRamp * scIdx_global, &phase_conj.y, &phase_conj.x);
       sh_rxSrs[i] = complex_mul(__float22half2_rn(phase_conj), sh_rxSrs[i]);
   }

   for (int i = thisThrdBlk.size()-1-tid; i < nSrsScBlock*nSrsScBlock; i += thisThrdBlk.size()) {
       const int row = i / nSrsScBlock;
       const int col = i - row*nSrsScBlock;
       sh_W_matrix[i] = (*pW_narrow)({row, col});
   }

   thisThrdBlk.sync();        //=============================================================================

   // STEP 5: remove cyclic shifts and apply narrow filter to estimate channel
   max_loop_iters   = nRxAntSrs * nSrsScBlock * nPortsPerComb;
   int avgHest_size = nRxAntSrs * N_GRP_PER_COMP_BLK * nPortsPerComb;
   for(int i = tid; i < max_loop_iters; i += thisThrdBlk.size())
   {
       int portIdx = i % nPortsPerComb;
       int scIdx   = (i / nPortsPerComb) % nSrsScBlock;
       int antIdx  = i / (nPortsPerComb * nSrsScBlock);
       int foccIdx = portToFoccMap[portIdx];

       const __half2 *w = sh_W_matrix + scIdx * nSrsScBlock;
       const __half2 *srs = sh_rxSrs + antIdx * nSrsScBlock;
       const __half2 *foccBase = sh_focc_table + foccIdx*FOCC_LENGTH;

       auto est = half2{0, 0};
       for(int inputScIdx = 0; inputScIdx < nSrsScBlock; inputScIdx++)
       {
           const auto focc = foccBase[inputScIdx % FOCC_LENGTH];
           est        = __hadd2(est, complex_mul(complex_conjmul(*w, focc), *srs));
           w++;
           srs++;
       }

       sh_Hest[i] = est;
   }

   for (int i = thisThrdBlk.size()-1-tid; i < avgHest_size; i += thisThrdBlk.size()) {
      sh_avgHest[i] = __half2{0,0};
   }

   thisThrdBlk.sync();

   // STEP 6: Average estimates. Estimate energy and noise
   __half2 noise_signal{0,0};

   max_loop_iters = nRxAntSrs * nSrsScBlock;
   for(int i = tid; i < max_loop_iters; i += thisThrdBlk.size())
   {
       int scIdx  = i % nSrsScBlock;
       int antIdx = i / nSrsScBlock;
       //------------------------------
       int     grpIdx = scIdx / (PRB_GRP_SIZE * nCombScPerPrb);

       const int scIdxModFoccLength = scIdx % FOCC_LENGTH;
       __half2 scRxEst{0, 0};
       for(int portIdx = 0; portIdx < nPortsPerComb; portIdx++)
       {
           // apply fOCC to sh_Hest and update scRxEst:
           auto foccIdx = portToFoccMap[portIdx];
           const auto focc = sh_focc_table[foccIdx*FOCC_LENGTH + scIdxModFoccLength];
           auto est     = sh_Hest[portIdx + nPortsPerComb * scIdx + nPortsPerComb * nSrsScBlock * antIdx];
           scRxEst      = __hadd2(scRxEst, complex_mul(focc, est));

           // update average Ests:
           __half2& sh_avgH = sh_avgHest[portIdx + nPortsPerComb * grpIdx + N_GRP_PER_COMP_BLK * nPortsPerComb * antIdx];
           atomicAdd(&sh_avgH, est);
       }

       // update block averaged signal/noise energy:
       __half2 noise  = __hsub2(scRxEst, sh_rxSrs[i]);
       noise          = __hmul2(noise, noise);
       __half2 signal = __hmul2(scRxEst, scRxEst);
       noise_signal   = __hadd2(noise_signal, __half2(__hadd(noise.x, noise.y), __hadd(signal.x, signal.y)));
   }

   for (int i = thisThrdBlk.size()-1-tid; i < nSrsScBlock; i += thisThrdBlk.size()) {
      float energyAccum = 0.0f;
      for (int j = 0; j < nRxAntSrs; j++) {
        for (int k = 0; k < nPortsPerComb; k++) {
            const __half2 est = sh_Hest[k + nPortsPerComb * i + nPortsPerComb * nSrsScBlock * j];
            auto est2 = __half22float2(__hmul2(est, est));
            energyAccum += est2.x + est2.y;
        }
      }
      avgSignalEnergySc[i] = energyAccum;
   }

   thisThrdBlk.sync();

   constexpr int numPrbs = nSrsScBlock / nCombScPerPrb;
   // The reduction below uses the first thread in each warp and, for the warp-based
   // reduction, the first full warp. Thus, start from the end of the last warp for
   // this energy reduction.
   for (int i = thisThrdBlk.size()-1-tid; i < numPrbs; i += thisThrdBlk.size()) {
      float accum = avgSignalEnergySc[i*nCombScPerPrb];
      for (int j = 1; j < nCombScPerPrb; j++) {
         accum += avgSignalEnergySc[i*nCombScPerPrb+j];
      }
      avgSignalEnergyPrb[i] = accum;
   }

   // No explicit block sync here because there is another block sync below
   // before avgSignalEnergyPrb is used

   // since sum of signal energies in rare occasions might exceed the range covered by FP16, we use float instead of __half
   float2 sum_noise_signal = __half22float2(noise_signal);
   sum_noise_signal.x = cg::reduce(tile, sum_noise_signal.x, cg::plus<float>());
   sum_noise_signal.y = cg::reduce(tile, sum_noise_signal.y, cg::plus<float>());

#if USE_SRS_ATOMIC_REDUCTION
   if (tile.thread_rank() == 0) {
       atomicAdd(&avgNoiseEnergy, sum_noise_signal.x);
       atomicAdd(&avgSignalEnergy, sum_noise_signal.y);
   }
   thisThrdBlk.sync();
#else
   __shared__ float2 sum_noise_signalPerTile[32]; // for reduce sum
   if (tile.thread_rank() == 0) {
       sum_noise_signalPerTile[tile.meta_group_rank()] = sum_noise_signal;
   }
   thisThrdBlk.sync();
   if (tile.meta_group_rank()==0)
   {
       sum_noise_signal = tile.thread_rank() < tile.meta_group_size() ? sum_noise_signalPerTile[tile.thread_rank()] : float2{0, 0};
       avgNoiseEnergy   = cg::reduce(tile, sum_noise_signal.x, cg::plus<float>());
       avgSignalEnergy  = cg::reduce(tile, sum_noise_signal.y, cg::plus<float>());
   }
   thisThrdBlk.sync();
#endif



   if(tid == 0)
   {
       avgNoiseEnergy  = (noiseEstDebias * nRepPerHop * avgNoiseEnergy) / (nRxAntSrs * nSrsScBlock);
       avgSignalEnergy = avgSignalEnergy / (nRxAntSrs * nPortsPerComb * nSrsScBlock);
       if(((nPortsPerComb != 1) && (combSize == 4)) || ((nPortsPerComb == 4) && (combSize == 2)))
       {
           avgNoiseEnergy = avgNoiseEnergy - avgSignalEnergy * POINT_ONE_PERCENT;
       }
   }
   thisThrdBlk.sync();

   //=============================================================================

   // STEP 7: save output to buffers
   // save avg ests
   auto chEstBuffOffset   = blockStartPrb / 2 - chEstBuffStartPrbGrp;
   auto chEstToL2Offset   = (blockStartPrb - hopStartPrb) / 2 + (nPrbsPerHop / 2) * hopIdx;
   float estNormalizerInv = static_cast<float>(nCombScPerPrb * PRB_GRP_SIZE);

   max_loop_iters = nRxAntSrs * nPortsPerComb * N_GRP_PER_COMP_BLK;
   for(int i = tid; i < max_loop_iters; i += thisThrdBlk.size())
   {
       int grpIdx  = i % N_GRP_PER_COMP_BLK;
       int portIdx = (i / N_GRP_PER_COMP_BLK) % nPortsPerComb;
       int antIdx  = i / (nPortsPerComb * N_GRP_PER_COMP_BLK);
       //ToDo considering avgHest is half float, should we change tChEstBuff to CUPHY_C_16F?
       float2 avgH{__half22float2(sh_avgHest[portIdx + nPortsPerComb * grpIdx + nPortsPerComb * N_GRP_PER_COMP_BLK * antIdx])};
       tChEstBuff({chEstBuffOffset + grpIdx, antIdx, portToUeAntMap[portIdx]}) = float2{avgH.x / estNormalizerInv, avgH.y / estNormalizerInv};
       tChEstToL2({chEstToL2Offset + grpIdx, antIdx, portToL2OutUeAntMap[portIdx]}) = float2{avgH.x / estNormalizerInv, avgH.y / estNormalizerInv};
   }

   // Compute and save per-prb SNR
   if(combIdx == 0)
   {
       auto signalEnergyNormalizer = 1.f/(nRxAntSrs * nCombScPerPrb * nPortsPerComb);
       for (int i = thisThrdBlk.size()-1-tid; i < N_PRB_PER_COMP_BLK; i += thisThrdBlk.size())
       {
           avgSignalEnergyPrb[i] = avgSignalEnergyPrb[i] * signalEnergyNormalizer;
           auto rbSnr            = 10 * log10f(avgSignalEnergyPrb[i] / avgNoiseEnergy);
           pUeRbSnr[nPrbsPerHop * hopIdx + blockStartPrb - hopStartPrb + i] = rbSnr;
       }
   }

   __shared__ uint32_t sh_ueBlockCntr;
   if(tid == 0)
   {
       atomicAdd((float*)(&tmpWidebandNoiseEnergy) , avgNoiseEnergy);
       atomicAdd((float*)(&tmpWidebandSignalEnergy), avgSignalEnergy);
       atomicAdd((__half2*)(&tmpWidebandScCorr)    , sh_avgScCorr);
       __threadfence();
       // for finalization step
       sh_ueBlockCntr = atomicAdd(&ueBlockCntr, 1) + 1;
   }
   thisThrdBlk.sync();

   //=============================================================================

   // FINALIZATION
   // in kernelSelect, we set grid dimension gridDim.x equal to nCompBlocks
   // the last thread-block to reach here will perform the finalization step
   if(sh_ueBlockCntr == ueNumBlocks)
   {
       for(int i = tid; i < pDynDescr->nSrsUes; i += thisThrdBlk.size())
       {
           widebandSnr = 10 * log10(tmpWidebandSignalEnergy / tmpWidebandNoiseEnergy);
           widebandSignalEnergy = tmpWidebandSignalEnergy;
           widebandNoiseEnergy = tmpWidebandNoiseEnergy;
           // timing advance
           uint32_t scs  = (1 << mu) * 15000; //2^mu * 15*10^3
           toEstMicroSec = float(-1.0e6) * atanf(__half2float(tmpWidebandScCorr.y) / __half2float(tmpWidebandScCorr.x)) / static_cast<float>(2 * M_PI * scs * combSize);

           widebandScCorr.x = tmpWidebandScCorr.x;
           widebandScCorr.y = tmpWidebandScCorr.y;
       }
   }
}

__launch_bounds__(1024, 1)
__global__ void srsChEst0Kernel(srsChEst0StatDescr_t* pStatDescr, srsChEst0DynDescr_t* pDynDescr)
{
   const uint32_t    compBlockIdx   = blockIdx.x;
   compBlockDescr_t& compBlockDescr = pDynDescr->compBlockDescrs[compBlockIdx];
   uint16_t          ueIdx          = compBlockDescr.ueIdx;
   ueDescr_t&        ueDescr        = pDynDescr->ueDescrs[ueIdx];
   uint8_t           nPortsPerComb  = ueDescr.nPortsPerComb;
   uint8_t           combSize       = ueDescr.combSize;

   if (combSize == 2) {
      if (nPortsPerComb == 1) {
         srsChEst0KernelInner<2,1>(pStatDescr, pDynDescr);
      } else if (nPortsPerComb == 2) {
         srsChEst0KernelInner<2,2>(pStatDescr, pDynDescr);
      } else if (nPortsPerComb == 4) {
         srsChEst0KernelInner<2,4>(pStatDescr, pDynDescr);
      } else {
         return;
      }
   } else { // combSize == 4
      if (nPortsPerComb == 1) {
         srsChEst0KernelInner<4,1>(pStatDescr, pDynDescr);
      } else if (nPortsPerComb == 2) {
         srsChEst0KernelInner<4,2>(pStatDescr, pDynDescr);
      } else if (nPortsPerComb == 4) {
         srsChEst0KernelInner<4,4>(pStatDescr, pDynDescr);
      } else {
         return;
      }
   }
}

srsChEst0::srsChEst0()
{}



void srsChEst0::getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
   statDescrSizeBytes  = sizeof(srsChEst0StatDescr_t);
   statDescrAlignBytes = alignof(srsChEst0StatDescr_t);

   dynDescrSizeBytes  = sizeof(srsChEst0DynDescr_t);
   dynDescrAlignBytes = alignof(srsChEst0DynDescr_t);
}

void  srsChEst0::kernelSelect(srsChEst0DynDescr_t*       pCpuDynDesc,
                            uint16_t                   nSrsUes,
                            uint16_t                   nCompBlocks,
                            cuphySrsChEst0LaunchCfg_t* pLaunchCfg)
{
   // thread block geometry:
   uint16_t max_nRxAnts = 0;
   uint8_t  max_nPorts  = 0;
   uint8_t  min_nCombs  = 4;
   int      max_loop_iters = 0;
   for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
   {
       ueDescr_t ueDescr       = pCpuDynDesc->ueDescrs[ueIdx];
       uint16_t  nRxAntSrs     = pCpuDynDesc->cellDescrs[ueDescr.cellIdx].nRxAntSrs;
       uint8_t   nPortsPerComb = ueDescr.nPortsPerComb;
       uint8_t   combSize      = ueDescr.combSize;
       //
       max_nRxAnts = (nRxAntSrs > max_nRxAnts) ? nRxAntSrs : max_nRxAnts;
       max_nPorts  = (nPortsPerComb > max_nPorts) ? nPortsPerComb : max_nPorts;
       min_nCombs  = (combSize < min_nCombs) ? combSize : min_nCombs;
       const int nSrsScBlock = combSize == 4 ? 12 : 24;
       int loop_iters = static_cast<int>(nRxAntSrs) * nSrsScBlock * nPortsPerComb;
       max_loop_iters = (loop_iters > max_loop_iters) ? loop_iters : max_loop_iters;
   }
   uint16_t max_nSrsSc = min_nCombs == 4 ? 12 : 24;

   // launch geometry (can change!)
   dim3 gridDim(nCompBlocks);
   const int MAX_THREADS_PER_BLOCK = 1024;
   const int nthreads = (max_loop_iters < MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK/2 : MAX_THREADS_PER_BLOCK;
   dim3 blockDim(nthreads,1);

   // kernel (only one kernel option for now)
   void* kernelFunc = reinterpret_cast<void*>(srsChEst0Kernel);
   cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc);

   // populate kernel parameters
   CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

   kernelNodeParamsDriver.blockDimX = blockDim.x;
   kernelNodeParamsDriver.blockDimY = blockDim.y;
   kernelNodeParamsDriver.blockDimZ = blockDim.z;

   kernelNodeParamsDriver.gridDimX = gridDim.x;
   kernelNodeParamsDriver.gridDimY = gridDim.y;
   kernelNodeParamsDriver.gridDimZ = gridDim.z;

   kernelNodeParamsDriver.extra          = nullptr;
   // shared memory buffer used in computing srs
   kernelNodeParamsDriver.sharedMemBytes = max_nRxAnts * max_nSrsSc * sizeof(__half2);                       // for sh_rxSrs
   kernelNodeParamsDriver.sharedMemBytes += max_nRxAnts * max_nSrsSc * max_nPorts * sizeof(__half2);         // for sh_Hest
   kernelNodeParamsDriver.sharedMemBytes += max_nRxAnts * N_GRP_PER_COMP_BLK * max_nPorts * sizeof(__half2); // for sh_avgHest
   kernelNodeParamsDriver.sharedMemBytes += max_nSrsSc * max_nSrsSc * sizeof(__half2);                       // for sh_W_{wide,narrow}
}




cuphyStatus_t srsChEst0::setup(  uint16_t                      nSrsUes,
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
                              cudaStream_t                  strm)
{
   if((nCells > MAX_N_SRS_CELL) || (nSrsUes > MAX_N_SRS_UE)) return CUPHY_STATUS_INVALID_ARGUMENT;
   uint16_t           nCompBlocks          = 0;
   constexpr uint16_t nPrbsPerComputeBlock = 4;
   pCpuDynDesc->nSrsUes                    = nSrsUes;

   for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
   {
       ueDescr_t&            ueDescr  = pCpuDynDesc->ueDescrs[ueIdx];
       cuphyUeSrsPrm_t&      ueSrsPrm = h_srsUePrms[ueIdx];
       cuphySrsCellPrms_t&   cellPrm  = h_srsCellPrms[ueSrsPrm.cellIdx];

       // ue descriptor parmaters to be populated:
       uint8_t  (&repSymIdxs)[MAX_N_HOPS][MAX_N_REPS]                  = ueDescr.repSymIdxs;
       uint16_t (&hopStartPrbs)[MAX_N_HOPS]                            = ueDescr.hopStartPrbs;
       uint8_t  (&nRepPerHop)[MAX_N_HOPS]                              = ueDescr.nRepPerHop;
       uint16_t& nPrbsPerHop                                           = ueDescr.nPrbsPerHop;
       uint8_t  (&u)[MAX_N_SYM]                                        = ueDescr.u;
       float    (&q)[MAX_N_SYM]                                        = ueDescr.q;
       float&    alphaCommon                                           = ueDescr.alphaCommon;
       uint8_t&  lowPaprTableIdx                                       = ueDescr.lowPaprTableIdx;
       uint16_t& lowPaprPrime                                          = ueDescr.lowPaprPrime;
       uint8_t&  nPortsPerComb                                         = ueDescr.nPortsPerComb;
       uint8_t  (&portToFoccMap)[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS]   = ueDescr.portToFoccMap;
       uint8_t&  combSize                                              = ueDescr.combSize;
       uint8_t  (&combOffsets)[MAX_N_COMB_PER_UE]                      = ueDescr.combOffsets;
       uint8_t&  nCombScPerPrb                                         = ueDescr.nCombScPerPrb;
       uint8_t  (&portToUeAntMap)[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS]  = ueDescr.portToUeAntMap;
         uint8_t  (&portToL2OutUeAntMap)[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS]  = ueDescr.portToL2OutUeAntMap;
       float*&                      pUeRbSnr                           = ueDescr.pUeRbSnr;
       uint8_t&                     cellIdx                            = ueDescr.cellIdx;
       cuphySrsReport_t*&           pUeSrsReport                       = ueDescr.pUeSrsReport;
       tensor_ref_any<CUPHY_C_32F>& tChEstBuff                         = ueDescr.tChEstBuff;
      tensor_ref_any<CUPHY_C_32F>&  tChEstToL2                         = ueDescr.tChEstToL2;
       uint16_t&                    chEstBuffStartPrbGrp               = ueDescr.chEstBuffStartPrbGrp;
       uint32_t&                    ueBlockCntr                        = ueDescr.ueBlockCntr;
       uint32_t&                    ueNumBlocks                        = ueDescr.ueNumBlocks;

       ueNumBlocks = 0;
       ueBlockCntr = 0;

       // some parameters can be directly copied:
       cellIdx  = ueSrsPrm.cellIdx;
       combSize = ueSrsPrm.combSize;

       // comb parameters:
       nCombScPerPrb         = (ueSrsPrm.combSize == 2) ? 6 : 3;
       uint8_t n_SRS_cs_max  = (ueSrsPrm.combSize == 2) ? 8 : 12;

       // hop prb size:
       nPrbsPerHop         = SRS_BW_TABLE[ueSrsPrm.configIdx][2*ueSrsPrm.bandwidthIdx];
       uint16_t M_sc_b_SRS = nPrbsPerHop * nCombScPerPrb;

       // compute SRS sequence group u and sequence number v:
       uint8_t  v[MAX_N_SYM];
       for(int symIdx = 0; symIdx < ueSrsPrm.nSyms; ++symIdx)
       {
           v[symIdx] = 0;
           u[symIdx] = ueSrsPrm.sequenceId % 30;
       }
       if(ueSrsPrm.groupOrSequenceHopping == 1)
       {
           uint32_t c  = seqCpuSrs::gold32n_CPU(ueSrsPrm.sequenceId, 8 * (cellPrm.slotNum * N_SYM_PER_SLOT + ueSrsPrm.startSym));
           for(int symIdx = 0; symIdx < ueSrsPrm.nSyms; ++symIdx)
           {
               uint8_t f_gh = ((c >> 8*symIdx) & LOWER_BYTE_BMSK) % 30;
               u[symIdx]    = (f_gh + ueSrsPrm.sequenceId) % 30;
           }
       }
       if((ueSrsPrm.groupOrSequenceHopping == 2) && (M_sc_b_SRS >= (N_SC_PER_PRB * 6)))
       {
           uint32_t c = seqCpuSrs::gold32n_CPU(ueSrsPrm.sequenceId, cellPrm.slotNum * N_SYM_PER_SLOT + ueSrsPrm.startSym);
           for(int symIdx = 0; symIdx < ueSrsPrm.nSyms; ++symIdx)
           {
               v[symIdx] = static_cast<uint8_t>((c >> symIdx) & 1);
           }
       }

       // determine if user gets multiple combs:
       uint8_t nCombs = 1;
       nPortsPerComb  = ueSrsPrm.nAntPorts;
       if ((ueSrsPrm.cyclicShift >= n_SRS_cs_max / 2) && (ueSrsPrm.nAntPorts == 4))
       {
           nCombs        = 2;
           nPortsPerComb = 2;
       }

       // cyclic shifts:
       alphaCommon = 2 * M_PI  * static_cast<float>(ueSrsPrm.cyclicShift % n_SRS_cs_max) / static_cast<float>(n_SRS_cs_max);
       for(int combIdx = 0; combIdx < nCombs; ++combIdx)
       {
           for(int portIdx = 0; portIdx < nPortsPerComb; ++portIdx)
           {
               portToFoccMap[combIdx][portIdx] = portIdx * MAX_N_ANT_PORTS / ueSrsPrm.nAntPorts * nCombs + combIdx;
           }
       }

       // port to antenna port
       for(int combIdx = 0; combIdx < nCombs; ++combIdx)
       {
           for(int portIdx = 0; portIdx < nPortsPerComb; ++portIdx)
           {
               portToUeAntMap[combIdx][portIdx] = ueSrsPrm.srsAntPortToUeAntMap[portIdx*nCombs + combIdx];
               portToL2OutUeAntMap[combIdx][portIdx] = portIdx*nCombs + combIdx;
           }
       }


       // comb offsets:
       for(int combIdx = 0; combIdx < nCombs; ++combIdx)
       {
           combOffsets[combIdx] = (ueSrsPrm.combOffset + combIdx * combSize / 2) % combSize;
       }

       // hop starting Prbs
       uint16_t hopStartPrbs0[MAX_N_SYM];
       uint8_t  nHops0         = ueSrsPrm.nSyms / ueSrsPrm.nRepetitions;
       uint16_t nSlotsPerFrame = 10 * (1 << cellPrm.mu);
       uint16_t m_SRS_b;

       for(int hopIdx = 0; hopIdx < nHops0; hopIdx++)
       {
           hopStartPrbs0[hopIdx] = ueSrsPrm.frequencyShift;
           uint16_t Nb, nb;
           for(int b = 0; b <= ueSrsPrm.bandwidthIdx; ++b)
           {
               Nb      = SRS_BW_TABLE[ueSrsPrm.configIdx][2*b + 1];
               m_SRS_b = SRS_BW_TABLE[ueSrsPrm.configIdx][2*b];
               nb      = ((4 * ueSrsPrm.frequencyPosition) / m_SRS_b) % Nb;

               if((b > ueSrsPrm.frequencyHopping) && (ueSrsPrm.frequencyHopping < ueSrsPrm.bandwidthIdx))
               {
                   uint16_t n_SRS;
                   if(ueSrsPrm.resourceType == 0)
                   {
                       n_SRS = hopIdx;
                   }else
                   {
                       uint16_t slotIdx = nSlotsPerFrame * cellPrm.frameNum + cellPrm.slotNum - ueSrsPrm.Toffset;
                       if((slotIdx % ueSrsPrm.Tsrs) == 0)
                       {
                           n_SRS = (slotIdx / ueSrsPrm.Tsrs) * (ueSrsPrm.nSyms / ueSrsPrm.nRepetitions) + hopIdx;
                       }

                   }

                   uint16_t PI_bm1 = 1;
                   for(int b_prime = ueSrsPrm.frequencyHopping + 1; b_prime < b; ++b_prime)
                   {
                       PI_bm1 = PI_bm1 * SRS_BW_TABLE[ueSrsPrm.configIdx][2*b_prime + 1];
                   }
                   uint16_t PI_b = PI_bm1 * Nb;
                   uint16_t Fb;
                   if((Nb % 2) == 0)
                   {
                       Fb = (Nb / 2) * ((n_SRS % PI_b) / PI_bm1) + (n_SRS % PI_b) / (2 * PI_bm1);
                   }else
                   {
                       Fb = (Nb / 2) * (n_SRS / PI_bm1);
                   }
                   nb = (Fb + 4 * ueSrsPrm.frequencyPosition / m_SRS_b) % Nb;
               }
               hopStartPrbs0[hopIdx] += m_SRS_b * nb;
           }
       }

       // low PAPR sequence parameters:
       uint16_t nSubcarriers = m_SRS_b * nCombScPerPrb;
       lowPaprTableIdx = 255;
       lowPaprPrime    = 0;

       if(nSubcarriers == 12)
       {
           lowPaprTableIdx = 0;
       }else
       {
           if(nSubcarriers == 24)
           {
               lowPaprTableIdx = 1;
           }else
           {
               for(int primeIdx = 1; primeIdx < N_PRIMES; ++primeIdx)
               {
                   if(PRIMES_TABLE[primeIdx] > nSubcarriers)
                   {
                       lowPaprPrime = PRIMES_TABLE[primeIdx - 1];
                       break;
                   }
               }
           }
       }

       for(int symIdx = 0; symIdx < ueSrsPrm.nSyms; ++symIdx)
       {
           float qBar = static_cast<float>(lowPaprPrime) * static_cast<float>(u[symIdx] + 1) / static_cast<float>(31);
           if((static_cast<uint32_t>(floor(2*qBar)) % 2) == 0)
           {
               q[symIdx] = floor(qBar + 0.5) + v[symIdx];
           }else
           {
               q[symIdx] = floor(qBar + 0.5) - v[symIdx];
           }
       }

       // combine hops with the same start PRB
       uint8_t nHops   = 1;
       nRepPerHop[0]   = ueSrsPrm.nRepetitions;
       hopStartPrbs[0] = hopStartPrbs0[0];
       for(int repIdx = 0; repIdx < ueSrsPrm.nRepetitions; ++repIdx)
       {
           repSymIdxs[0][repIdx] = ueSrsPrm.startSym - cellPrm.srsStartSym + repIdx;
       }

       for(int hopIdx0 = 1; hopIdx0 < nHops0; hopIdx0++)
       {
           bool newHopFlag = true;
           for(int hopIdx = 0; hopIdx < nHops; ++hopIdx)
           {
               if(hopStartPrbs0[hopIdx0] == hopStartPrbs[hopIdx])
               {
                   for(int repIdx = 0; repIdx < ueSrsPrm.nRepetitions; ++repIdx)
                   {
                       repSymIdxs[hopIdx][nRepPerHop[hopIdx] + repIdx] = ueSrsPrm.startSym - cellPrm.srsStartSym + hopIdx0 * ueSrsPrm.nRepetitions + repIdx;
                   }
                   nRepPerHop[hopIdx] += ueSrsPrm.nRepetitions;
                   newHopFlag          = false;
                   break;
               }
           }

           if(newHopFlag)
           {
               for(int repIdx = 0; repIdx <  ueSrsPrm.nRepetitions; ++repIdx)
               {
                   repSymIdxs[nHops][repIdx] = ueSrsPrm.startSym - cellPrm.srsStartSym + hopIdx0 * ueSrsPrm.nRepetitions + repIdx;
               }
               nRepPerHop[nHops]   = ueSrsPrm.nRepetitions;
               hopStartPrbs[nHops] = hopStartPrbs0[nHops];
               nHops += 1;
           }
       }

       // ues output parameters:
       uint16_t chEstBuffIdx                    = ueSrsPrm.chEstBuffIdx;
       cuphySrsChEstBuffInfo_t& ueChEstBuffInfo = h_chEstBuffInfo[chEstBuffIdx];

       tensor_pair tPairUeChEstBuffer (static_cast<const tensor_desc&>(*(ueChEstBuffInfo.tChEstBuffer.desc)), ueChEstBuffInfo.tChEstBuffer.pAddr);
       const tensor_layout_any& tUeChEstBufferLayout = tPairUeChEstBuffer.first.get().layout();
       void*                    tUeChEstBufferAddr   = tPairUeChEstBuffer.second;
       tChEstBuff = std::move(tensor_ref_any<CUPHY_C_32F>(tUeChEstBufferAddr, tUeChEstBufferLayout.dimensions.begin(), tUeChEstBufferLayout.strides.begin()));

       chEstBuffStartPrbGrp = ueChEstBuffInfo.startPrbGrp;
       pUeSrsReport         = d_pSrsReports + ueIdx;
       pUeRbSnr             = d_rbSnrBuff + h_rbSnrBuffOffsets[ueIdx];

         h_chEstToL2[ueIdx].prbGrpSize = 2; // Hard coded to two for now TODO: uncomment once driver populatesnew API
         h_chEstToL2[ueIdx].nPrbGrps   = nHops * nPrbsPerHop / h_chEstToL2[ueIdx].prbGrpSize;
         int prbGrpSize = 2;
         int nPrbGrps   = nHops * nPrbsPerHop / prbGrpSize;

         std::array<int, CUPHY_DIM_MAX> srsChEstToL2Dim;
         srsChEstToL2Dim.fill(1);
         srsChEstToL2Dim[0] = nPrbGrps;
         srsChEstToL2Dim[1] = cellPrm.nRxAntSrs;
         srsChEstToL2Dim[2] = ueSrsPrm.nAntPorts;

         std::array<int, CUPHY_DIM_MAX> srsChEstToL2Str;
         srsChEstToL2Str.fill(nPrbGrps * cellPrm.nRxAntSrs * ueSrsPrm.nAntPorts);
         srsChEstToL2Str[0] = 1;
         srsChEstToL2Str[1] = nPrbGrps;
         srsChEstToL2Str[2] = nPrbGrps * cellPrm.nRxAntSrs;

         tChEstToL2 = std::move(tensor_ref_any<CUPHY_C_32F>(d_addrsChEstToL2Buff[ueIdx], srsChEstToL2Dim.begin(), srsChEstToL2Str.begin()));

       // allocate compute blocks for the user:
       uint16_t nBlocksFreq = nPrbsPerHop / nPrbsPerComputeBlock;
       for(int freqBlockIdx = 0; freqBlockIdx < nBlocksFreq; ++freqBlockIdx)
       {
           for(int hopIdx = 0; hopIdx < nHops; ++hopIdx)
           {
               for(int combIdx = 0; combIdx < nCombs; ++combIdx)
               {
                   pCpuDynDesc->compBlockDescrs[nCompBlocks].ueIdx         = ueIdx;
                   pCpuDynDesc->compBlockDescrs[nCompBlocks].hopIdx        = hopIdx;
                   pCpuDynDesc->compBlockDescrs[nCompBlocks].combIdx       = combIdx;
                   pCpuDynDesc->compBlockDescrs[nCompBlocks].blockStartPrb = hopStartPrbs[hopIdx] + freqBlockIdx * nPrbsPerComputeBlock;
                   nCompBlocks += 1;
                   ueNumBlocks += 1;
               }
           }
       }
   }

   // populate cell descriptors:
   for(int cellIdx = 0; cellIdx < nCells; ++cellIdx)
   {
       pCpuDynDesc->cellDescrs[cellIdx].mu     = h_srsCellPrms[cellIdx].mu;
       pCpuDynDesc->cellDescrs[cellIdx].nRxAntSrs = h_srsCellPrms[cellIdx].nRxAntSrs;

       tensor_pair tPairDataRx (static_cast<const tensor_desc&>(*(pTDataRx[cellIdx].desc)), pTDataRx[cellIdx].pAddr);
       const tensor_layout_any& tDataRxLayout = tPairDataRx.first.get().layout();
       void*                    tDataRxAddr   = tPairDataRx.second;
       pCpuDynDesc->cellDescrs[cellIdx].tDataRx = std::move(tensor_ref_any<CUPHY_C_16F>(tDataRxAddr, tDataRxLayout.dimensions.begin(), tDataRxLayout.strides.begin()));
   }

   srsChEst0KernelArgs_t& kernelArgs = m_kernelArgs;
   kernelArgs.pDynDescr = reinterpret_cast<srsChEst0DynDescr_t*>(pGpuDynDesc);

   // optional descriptor copy to GPU memory
   if(enableCpuToGpuDescrAsyncCpy)
   {
       cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(srsChEst0DynDescr_t), cudaMemcpyHostToDevice, strm);
   }

   // select kernel (includes launch geometry). Populate launchCfg.
   kernelSelect(pCpuDynDesc, nSrsUes, nCompBlocks, pLaunchCfg);
   pLaunchCfg->kernelArgs[0] = &kernelArgs.pStatDescr;
   pLaunchCfg->kernelArgs[1] = &kernelArgs.pDynDescr;

   pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
   return CUPHY_STATUS_SUCCESS;

}

void tensorPrm_to_tensorRef(cuphyTensorPrm_t& tensorPrm, tensor_ref_any<CUPHY_C_16F>& tRef)
{
   tensor_pair tPair(static_cast<const tensor_desc&>(*(tensorPrm.desc)), tensorPrm.pAddr);
   const tensor_layout_any& tLayout = tPair.first.get().layout();
   void*                    tAddr   = tPair.second;
   tRef = std::move(tensor_ref_any<CUPHY_C_16F>(tAddr, tLayout.dimensions.begin(), tLayout.strides.begin()));
}

void srsChEst0::init(cuphySrsFilterPrms_t* pSrsFilterPrms,
                    bool                  enableCpuToGpuDescrAsyncCpy,
                    srsChEst0StatDescr_t* pCpuStatDesc,
                    void*                 pGpuStatDesc,
                    cudaStream_t          strm)
{
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmFocc_table, pCpuStatDesc->tFocc_table);

   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb2_nPorts1_wide, pCpuStatDesc->tW_comb2_nPorts1_wide);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb2_nPorts2_wide, pCpuStatDesc->tW_comb2_nPorts2_wide);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb2_nPorts4_wide, pCpuStatDesc->tW_comb2_nPorts4_wide);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb4_nPorts1_wide, pCpuStatDesc->tW_comb4_nPorts1_wide);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb4_nPorts2_wide, pCpuStatDesc->tW_comb4_nPorts2_wide);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb4_nPorts4_wide, pCpuStatDesc->tW_comb4_nPorts4_wide);

   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb2_nPorts1_narrow, pCpuStatDesc->tW_comb2_nPorts1_narrow);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb2_nPorts2_narrow, pCpuStatDesc->tW_comb2_nPorts2_narrow);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb2_nPorts4_narrow, pCpuStatDesc->tW_comb2_nPorts4_narrow);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb4_nPorts1_narrow, pCpuStatDesc->tW_comb4_nPorts1_narrow);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb4_nPorts2_narrow, pCpuStatDesc->tW_comb4_nPorts2_narrow);
   tensorPrm_to_tensorRef(pSrsFilterPrms->tPrmW_comb4_nPorts4_narrow, pCpuStatDesc->tW_comb4_nPorts4_narrow);

   pCpuStatDesc->noisEstDebias_comb2_nPorts1 = pSrsFilterPrms->noisEstDebias_comb2_nPorts1;
   pCpuStatDesc->noisEstDebias_comb2_nPorts2 = pSrsFilterPrms->noisEstDebias_comb2_nPorts2;
   pCpuStatDesc->noisEstDebias_comb2_nPorts4 = pSrsFilterPrms->noisEstDebias_comb2_nPorts4;
   pCpuStatDesc->noisEstDebias_comb4_nPorts1 = pSrsFilterPrms->noisEstDebias_comb4_nPorts1;
   pCpuStatDesc->noisEstDebias_comb4_nPorts2 = pSrsFilterPrms->noisEstDebias_comb4_nPorts2;
   pCpuStatDesc->noisEstDebias_comb4_nPorts4 = pSrsFilterPrms->noisEstDebias_comb4_nPorts4;

   srsChEst0KernelArgs_t& kernelArgs = m_kernelArgs;
   kernelArgs.pStatDescr = reinterpret_cast<srsChEst0StatDescr_t*>(pGpuStatDesc);

   if(enableCpuToGpuDescrAsyncCpy)
   {
       cudaMemcpyAsync(pGpuStatDesc, pCpuStatDesc, sizeof(srsChEst0StatDescr_t), cudaMemcpyHostToDevice, strm);
   }
}
