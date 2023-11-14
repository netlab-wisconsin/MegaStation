/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuphy.h"
#include "cuphy_api.h"
#include "tensor_desc.hpp"
#include <limits>
#include <mutex>


#pragma once

static constexpr uint32_t N_TONES_PER_PRB     = 12;
static constexpr uint32_t LOWER_BYTE_BMSK     = 255;
static constexpr uint32_t MAX_SYMS_F0         = 2;
static constexpr uint32_t MAX_RX_ANTENNA      = 16;
static constexpr uint32_t SYM_PER_SLOT        = 14;
static constexpr uint32_t F0_CG_SIZE          = 32;
static constexpr uint32_t F0_GROUPS_PER_BLOCK = 1;  // previous to cell group, it was 8; set to 1 since the kernel assumes nRxAnt is common
                                                    // across all UE groups which is not necessary valid in multi-cell scenarios

static constexpr float confidenceThrF0 = 0.1; // threshold for determining confidence levels of SR and HARQ values

 // Find the largest correlation score and the accompanying index into the mcs array
 template <int n_mcs>
 inline __device__ void UpdateCsArray(const uint16_t cs_common[2], 
                                             const uint8_t num_sym,
                                             const uint8_t cs0, 
                                             float *group_corr,
                                             uint32_t *index,
                                             float *largest)
{
   *largest = 0;
   uint8_t cs;
   float corr_tmp;

   //if constexpr (n_mcs == 1) {
   if (n_mcs == 1) {
      // m_csArray[] = {0}
      corr_tmp = 0;
      for (int s = 0; s < num_sym; s++) {
         cs = (cs_common[s] + cs0 + 0) % N_TONES_PER_PRB;
         corr_tmp += group_corr[s*N_TONES_PER_PRB + cs];
      }

      *largest = corr_tmp;
      *index = 0;
   }
   //else if constexpr (n_mcs == 2) {
   else if (n_mcs == 2) {
      // m_csArray[] = {0, 6}
      #pragma unroll
      for (uint8_t m = 0; m < 2; m++) {
         corr_tmp = 0;
         for (int s = 0; s < num_sym; s++) {
            cs = (cs_common[s] + cs0 + m*6) % N_TONES_PER_PRB;
            corr_tmp += group_corr[s*N_TONES_PER_PRB + cs];
         }

         if (corr_tmp > *largest) {
            *largest = corr_tmp;
            *index = m;
         }
      }

      return;
   }
   //else if constexpr (n_mcs == 4) {
   else if (n_mcs == 4) {
      static constexpr uint8_t m_cs[] = {0, 9, 3, 6};
      #pragma unroll
      for (uint8_t m = 0; m < 4; m++) {
         corr_tmp = 0;
         for (int s = 0; s < num_sym; s++) {
            cs = (cs_common[s] + cs0 + m_cs[m]) % N_TONES_PER_PRB;
            corr_tmp += group_corr[s*N_TONES_PER_PRB + cs];
         }

         if (corr_tmp > *largest) {
            *largest = corr_tmp;
            *index = m;        
         }         
      }      
   }
   else {
      static constexpr uint8_t m_cs[] = {0, 1, 9, 10, 3, 4, 6, 7};      
      #pragma unroll
      for (uint8_t m = 0; m < 8; m++) {
         corr_tmp = 0;
         for (int s = 0; s < num_sym; s++) {
            cs = (cs_common[s] + cs0 + m_cs[m]) % N_TONES_PER_PRB;
            corr_tmp += group_corr[s*N_TONES_PER_PRB + cs];
         }

         if (corr_tmp > *largest) {
            *largest = corr_tmp;
            *index = m;
         }               
      }         
   }
}

// Implementation of the PUCCH F0 reciever interface exposed as an opaque data type to abstract out implementation
// details (PUCCH F0 reciever  C++ class). The PUCCH F0 reciever  is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPucchF0Rx
{};


// parameters for a single PUCCH F0 Group
struct pucchF0UciGrpPrms
{
    uint8_t nUciInGrp;

    // parameters shared by all UCI in group:
    uint8_t  freqHopFlag;
    uint16_t bwpStart;
    uint8_t  startSym;
    uint16_t startPrb;
    uint8_t  nSym;
    uint8_t  groupHopFlag;
    uint16_t secondHopPrb;
    uint8_t  u[2];
    uint16_t csCommon[2];
    uint16_t cellIdx;

    // UCI specific parameters:
    uint8_t     bitLenHarq    [CUPHY_PUCCH_F0_MAX_UCI_PER_GRP];
    uint8_t     srFlag        [CUPHY_PUCCH_F0_MAX_UCI_PER_GRP];
    uint8_t     cs0           [CUPHY_PUCCH_F0_MAX_UCI_PER_GRP];
    uint16_t    uciOutputIdx  [CUPHY_PUCCH_F0_MAX_UCI_PER_GRP];
    __half      DTXthreshold  [CUPHY_PUCCH_F0_MAX_UCI_PER_GRP];
};
typedef struct pucchF0UciGrpPrms pucchF0UciGrpPrms_t;

// Pucch F0 reciever dynamic descriptor
struct pucchF0RxDynDescr
{
    pucchF0UciGrpPrms_t         uciGrpPrms[CUPHY_PUCCH_F0_MAX_GRPS]; // parameters
    cuphyPucchF0F1UciOut_t*     pF0UcisOut;                          // output uci buffer
    cuphyPucchCellPrm_t*        pCellPrms;                           // RX Antennas, slot num, hopping id and input slot buffer
    uint16_t                    numUciGrps;
};
typedef struct pucchF0RxDynDescr pucchF0RxDynDescr_t;

// PUCCH format 0 kernel arguments (supplied via descriptors)
typedef struct
{
    pucchF0RxDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_RSSI_MEAS_N_HOM_CFG dynamic descriptors
} pucchF0KernelArgs_t;


// Class implementation of PUCCH F0 reciver
// class puschRxChEst : public cuphyPuschRxChEst
class pucchF0Rx : public cuphyPucchF0Rx
{
public:
    pucchF0Rx(cudaStream_t strm);
    ~pucchF0Rx()                           = default;
    pucchF0Rx(pucchF0Rx const&)            = delete;
    pucchF0Rx& operator=(pucchF0Rx const&) = delete;

    void InitConstantMem(cudaStream_t strm);

    // setup object state and dynamic component descriptor in prepration towards execution
    void setup(cuphyTensorPrm_t*          pDataRx,
               cuphyPucchF0F1UciOut_t*    pF0UcisOut,
               uint16_t                   nCells,
               uint16_t                   nF0Ucis,
               cuphyPucchUciPrm_t*        pF0UciPrms,
               cuphyPucchCellPrm_t*       pCmnCellPrms,                // number of antennas, slot number and hopping idx
               bool                       enableCpuToGpuDescrAsyncCpy,
               pucchF0RxDynDescr_t*       pCpuDynDesc,                     // pointer to descriptor in cpu
               void*                      pGpuDynDesc,                     // pointer to descriptor in gpu
               cuphyPucchF0RxLaunchCfg_t* pLaunchCfg,                      // pointer to launch configuration
               cudaStream_t               strm);                           // stream to perform copy


    void kernelSelect(uint16_t                   nUciGrps,
                      cuphyPucchF0RxLaunchCfg_t* pLaunchCfg);


    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    pucchF0KernelArgs_t m_kernelArgs;

   private:
      static bool isConstMemInited;
      static std::mutex m_mutexConstMemInit;

};

