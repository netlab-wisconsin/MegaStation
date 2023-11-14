/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 #include "uciOnPusch_segLLRs0.hpp"
 #include "descrambling.cuh"

using namespace cuphy_i;

__host__ __device__ __forceinline__ void assignReGrid0(reGrid0_t& reGrid,
                                          uint16_t  nUnassignedResInSymbol, 
                                          uint32_t  nUnassignedBitsInSymbol,
                                          uint32_t& nAssignedRmBits, 
                                          uint32_t  G, 
                                          uint8_t   nBitsPerRe)
{
   reGrid.rmBufferOffset = nAssignedRmBits;

   uint32_t nUnassignedRmBits = G - nAssignedRmBits;
   if(nUnassignedRmBits > nUnassignedBitsInSymbol){
      reGrid.nRes     = nUnassignedResInSymbol;
      reGrid.ReStride = 1;
   }else{
      reGrid.nRes     = div_round_up(nUnassignedRmBits, static_cast<uint32_t>(nBitsPerRe));
      reGrid.ReStride = nUnassignedBitsInSymbol / nUnassignedRmBits;
   }

   nAssignedRmBits = nAssignedRmBits + reGrid.nRes * nBitsPerRe;
}
 


__host__ __device__ __forceinline__ void updateNumUnassigned0(reGrid0_t&  reGrid,
                                             uint16_t&  nUnassignedResInSymbol,
                                             uint32_t&  nUnassignedBitsInSymbol,
                                             uint8_t    nBitsPerRe)
{
   nUnassignedResInSymbol  = nUnassignedResInSymbol  - reGrid.nRes;
   nUnassignedBitsInSymbol = nUnassignedBitsInSymbol - reGrid.nRes * nBitsPerRe;
}

__device__ __forceinline__ void checkIfReAssigned0(reGrid0_t&  reGrid,
                                   uint16_t   virtualReIdx,
                                   bool&      assignFlag,
                                   uint16_t&  cumltNumAssignedRes)
{
   uint16_t r = virtualReIdx % reGrid.ReStride;
   uint16_t d = virtualReIdx / reGrid.ReStride;

   // Check if RE is assigned to the RM buffer
   if(r != reGrid.gridOffset)
   {
      assignFlag          = false;
      cumltNumAssignedRes = ((d + 1) >= reGrid.nRes) ? reGrid.nRes : (d + 1);
   }else if(d >= reGrid.nRes)
   {
      assignFlag          = false;
      cumltNumAssignedRes = reGrid.nRes;
   }else
   {
      assignFlag          = true;
      cumltNumAssignedRes = d;
   } 
}

__device__ __forceinline__ void standardDescramble0(__half*    pReLLRs,
                                    uint32_t   scramSeq,
                                    uint8_t    nBitsPerRe)
{
   for(uint8_t bitIdx = 0; bitIdx < nBitsPerRe; ++bitIdx)
   {
      if((scramSeq >> bitIdx) & 1)
      {
         pReLLRs[bitIdx] = -pReLLRs[bitIdx];
      }
   }
}

__device__ __forceinline__ void oneBitSpxDescramble0(__half*    pReLLRs,
                                     uint32_t   scramSeq,
                                     uint8_t    nLayers,
                                     uint8_t    nBitsPerQam)
{
   for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
   {
      uint8_t bitIdx = layerIdx * nBitsPerQam;
      if((scramSeq >> bitIdx) & 1)
      {
         pReLLRs[bitIdx]      =  -pReLLRs[bitIdx];
         pReLLRs[bitIdx + 1]  =  -pReLLRs[bitIdx + 1];
      }
   }
}

__device__ __forceinline__ void descramAndStoreLLRs0(uint32_t reDescSeq, uint8_t spx1Flag, __half* pRmLLRBuff, uint8_t nBitsPerRe, __half* pReLLRs)
{
   
   // Scramble even bit LLRs:
   for(uint8_t bitIdx = 0; bitIdx < nBitsPerRe; bitIdx+=2)
   {
      if((reDescSeq >> bitIdx) & 1)
      {
         pReLLRs[bitIdx] = -pReLLRs[bitIdx];
      }
   }

   // Scramble odd bit LLRs:
   for(uint8_t bitIdx = 1; bitIdx < nBitsPerRe; bitIdx+=2)
   {
      if((reDescSeq >> (bitIdx - spx1Flag)) & 1)
      {
         pReLLRs[bitIdx] = -pReLLRs[bitIdx];
      }
   }

   // Store RE LLRs into RM buffer:
   if(nBitsPerRe % 4 == 0)
   {
      half4* pRmBuffer4 = reinterpret_cast<half4*>(pRmLLRBuff);
      half4* pReLLRs4   = reinterpret_cast<half4*>(pReLLRs);
      for(int bitIdx = 0; bitIdx < nBitsPerRe / 4; ++bitIdx)
      {
         // 64-bit store
         pRmBuffer4[bitIdx].dbl = pReLLRs4[bitIdx].dbl;
      }
   }
   else if(nBitsPerRe % 2 == 0)
   {
      __half2* pRmBuffer2 = reinterpret_cast<__half2*>(pRmLLRBuff);
      __half2* pReLLRs2    = reinterpret_cast<__half2*>(pReLLRs);
      for(int bitIdx = 0; bitIdx < nBitsPerRe / 2; ++bitIdx)
      {
         // 32-bit store
         pRmBuffer2[bitIdx] = pReLLRs2[bitIdx];
      }
   }
   else
   {
      for(int bitIdx = 0; bitIdx < nBitsPerRe; ++bitIdx)
      {
         // 16-bit store
         pRmLLRBuff[bitIdx] = pReLLRs[bitIdx];
      }
   }
}

 __global__ void
 __launch_bounds__(168, 6) // 168 = 12 * 14 is maximum CTA size
 uciOnPuschSegLLRs0Kernel(uciOnPuschSegLLRs0DynDescr_t* pDesc)
{
   const uint16_t uciIdx  = blockIdx.y;

   // segmentation options:
   cuphyUciToSeg_t uciToSeg = pDesc->uciToSeg;

   // Indicies of RE being processed:
   const uint16_t prbIdx     = blockIdx.x;
   const uint16_t reIdx      = 12 * prbIdx + threadIdx.x;
   const uint8_t  symIdx     = threadIdx.y;

   // Indicies of Ue and UeGrp
   uint16_t ueGrpIdx = pDesc->uciToUserMap[uciIdx].ueGrpIdx;
   uint16_t ueIdx    = pDesc->uciToUserMap[uciIdx].ueIdx;
   
   // User parameters:
   uint8_t   nBitsPerQam      = pDesc->pUePrmsGpu[ueIdx].Qm;
   uint8_t   nLayers          = pDesc->pUePrmsGpu[ueIdx].Nl;
   uint32_t  cinit            = pDesc->pUePrmsGpu[ueIdx].cinit;
   uint32_t* pLayerMap        = pDesc->pUePrmsGpu[ueIdx].layer_map_array;
   uint8_t   csi2Flag         = pDesc->pUePrmsGpu[ueIdx].csi2Flag;
   uint16_t  nPrb             = pDesc->pUeGrpPrmsGpu[ueGrpIdx].nPrb;
   uint8_t   isDataPresent    = pDesc->pUePrmsGpu[ueIdx].isDataPresent;

   // Per UCI parameters:
   perUciPrms0_t& perUciPrms    =  pDesc->perUciPrmsArray[uciIdx];
   reGrid0_t& rvdHarqReGrid     =  perUciPrms.rvdHarqReGrids[symIdx];
   reGrid0_t& harqReGrid        =  perUciPrms.harqReGrids[symIdx];
   reGrid0_t& csi1ReGrid        =  perUciPrms.csi1ReGrids[symIdx];
   bool       dmrsFlag          =  perUciPrms.dmrsFlags[symIdx];
   uint32_t   schRmBuffOffset   =  perUciPrms.schRmBuffOffsets[symIdx];
   uint8_t    nBitsPerRe        =  perUciPrms.nBitsPerRe;
   bool       harqPunctFlag     =  perUciPrms.harqPunctFlag;
   uint8_t    harqSpx1Flag      =  perUciPrms.harqSpx1Flag;
   uint8_t    nSym              =  perUciPrms.nSym;
   uint32_t*  pDescramOffsets   =  perUciPrms.descramOffsets;

   // Input Buffer
   tensor_ref_any<CUPHY_R_16F>& tEqOutLLRs  =  pDesc->tEqOutLLRs[ueGrpIdx];

   // Output buffers
   __half* pHarqLLRs = pDesc->pUePrmsGpu[ueIdx].d_harqLLrs;
   __half* pCsi1LLRs = pDesc->pUePrmsGpu[ueIdx].d_csi1LLRs;
   __half* pSchLLRs  = pDesc->pUePrmsGpu[ueIdx].d_schAndCsi2LLRs;

   // check for early exit
   if((symIdx >= nSym) || (prbIdx >= nPrb)){
      return;
   }
   if(dmrsFlag && ((reIdx % 2 == 0) || (isDataPresent == 0))){
      return;
   }

   // Loads LLRs to be assigned HARQ, CSI-P2, or SCH
   __half reLLRs[MAX_BITS_PER_RE];

   auto tEqOutLLRsAdr = tEqOutLLRs.addr();
   auto layout        = tEqOutLLRs.layout();
   int  s0            = layout.strides[0];
   int  s1            = layout.strides[1];
   int  idx0          = layout.strides[2] * reIdx + layout.strides[3] * symIdx;

   if (s0 == 1 && (s1 % 4) == 0 && (idx0 % 4) == 0 && (nBitsPerQam % 4) == 0) {
      // use 64-bit LD/ST
      half4* tEqOutLLRsAdr4 = reinterpret_cast<half4*>(tEqOutLLRs.addr());
      half4* reLLRs4        = reinterpret_cast<half4*>(reLLRs);
      for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
      {
            int idx1 = (pLayerMap[layerIdx] * s1 + idx0) / 4;
            for(uint8_t bitIdx = 0; bitIdx < nBitsPerQam / 4; ++bitIdx)
            {
                reLLRs4[bitIdx + layerIdx * nBitsPerQam / 4].dbl = tEqOutLLRsAdr4[bitIdx + idx1].dbl;
            }
      }
   } else {
      for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
      {
            int idx1 = pLayerMap[layerIdx] * s1 + idx0;
            for(uint8_t bitIdx = 0; bitIdx < nBitsPerQam; ++bitIdx)
            {
                reLLRs[bitIdx + layerIdx * nBitsPerQam] = tEqOutLLRsAdr[bitIdx * s0 + idx1];
            }
      }
   }

   // RE descrambling sequence
   uint32_t reDescSeq = 0;
   if(dmrsFlag){
      reDescSeq = descrambling::gold32n(cinit, pDescramOffsets[symIdx] + reIdx/2*nBitsPerRe);
   }else{
      reDescSeq = descrambling::gold32n(cinit, pDescramOffsets[symIdx] + reIdx*nBitsPerRe);
   }

   // If DMRS symbol, descramble and store the LLR
   if(dmrsFlag){
      __half* pRmLLRBuff = pSchLLRs + schRmBuffOffset + reIdx / 2 * nBitsPerRe;
      descramAndStoreLLRs0(reDescSeq, 0, pRmLLRBuff, nBitsPerRe, reLLRs);
      return;
   }

   // initialize variables:
   uint16_t virtualReIdx        = reIdx;
   bool     rvdHarqReFlag       = false;
   bool     assignFlag          = false;
   uint16_t cumltNumAssignedRes = 0;
   bool     thisReIsPunctFlag   = false;

   if(rvdHarqReGrid.nRes > 0)
   {
      // check if RE reserved to HARQ. Compute the number of REs
      // reserved to HARQ < reIdx
      uint16_t cumltNumRvdRes = 0;
      checkIfReAssigned0(rvdHarqReGrid,
                        reIdx,
                        rvdHarqReFlag,
                        cumltNumRvdRes);

      if(rvdHarqReFlag)
      {
         // If reserved, check if RE assigend to HARQ. Compute the number of
         // REs assigned to HARQ < reIdx.
         checkIfReAssigned0(harqReGrid,
                           cumltNumRvdRes,
                           assignFlag,
                           cumltNumAssignedRes);
         if(assignFlag)
         {
            __half* pRmLLRBuff = pHarqLLRs + harqReGrid.rmBufferOffset + cumltNumAssignedRes * nBitsPerRe;
            descramAndStoreLLRs0(reDescSeq, harqSpx1Flag, pRmLLRBuff, nBitsPerRe, reLLRs);

            if(harqPunctFlag)
            {
               thisReIsPunctFlag = true;
            }else
            {
               return;
            }
         }
      }else
      {
         // Otherwise update virtualReIdx (RE index after all RVD HARQ REs removed)
         virtualReIdx = virtualReIdx - cumltNumRvdRes;
      }
   }

   if(uciToSeg == SEG_ONLY_EARLY_UCI)
   {
      return;
   }

   if((csi1ReGrid.nRes > 0) && (!rvdHarqReFlag))
   {
      // Check if RE assigend to CSI-P1. Compute the number of CSI-P1 REs
      // assigned < reIdx.
      checkIfReAssigned0(csi1ReGrid,
                        virtualReIdx,
                        assignFlag,
                        cumltNumAssignedRes);
      if(assignFlag)
      {         
         __half* pRmLLRBuff = pCsi1LLRs + csi1ReGrid.rmBufferOffset + cumltNumAssignedRes * nBitsPerRe;
         descramAndStoreLLRs0(reDescSeq, 0, pRmLLRBuff, nBitsPerRe, reLLRs);
         return;
      }else
      {
         // otherwise update virtualReIdx
         if(harqPunctFlag)
         {
            virtualReIdx = reIdx - cumltNumAssignedRes;        // RE index after all CSI-P1 REs removed (HARQ puncturing means REs assigned to both HARQ and SCH/CSI-P2)
         }else
         {
            virtualReIdx = virtualReIdx - cumltNumAssignedRes; // RE index after all HARQ + CSI-P1 REs removed
         }
      }
   }else
   {
      if(harqPunctFlag) 
      {
         virtualReIdx = reIdx; 
      }
   }

   // RE assigned to SCH 
   if((csi2Flag == 0) && (isDataPresent == 1)) // if CSI2 enabled, LLRs saved during seg2 kernel
   {
      if(thisReIsPunctFlag) // If punctured the bits belong to HARQ, set RE LLRs to zero
      {
         for(uint8_t bitIdx = 0; bitIdx < nBitsPerRe; ++bitIdx)
         {
            reLLRs[bitIdx] = 0;
         }
      }

      __half* pRmBuffer = pSchLLRs + schRmBuffOffset + virtualReIdx * nBitsPerRe;
      descramAndStoreLLRs0(reDescSeq, 0, pRmBuffer, nBitsPerRe, reLLRs);
   }
}


void kernelSelect(uint16_t nUciUes, uciToUserMap_t* pUciToUserMap, cuphyPuschRxUeGrpPrms_t*  pUeGrpPrmsCpu, cuphyUciOnPuschSegLLRs0LaunchCfg_t* pLaunchCfg)
{
   // determine max number of PRBs and OFDM symbols
   uint16_t MAX_N_PRBS = 0;
   uint8_t  MAX_N_SYMS = 0;
   for(int uciIdx = 0; uciIdx < nUciUes; ++uciIdx)
   {
      uint16_t ueGrpIdx = pUciToUserMap[uciIdx].ueGrpIdx;

      if(pUeGrpPrmsCpu[ueGrpIdx].nPrb > MAX_N_PRBS)
         MAX_N_PRBS = pUeGrpPrmsCpu[ueGrpIdx].nPrb;

      uint8_t nDmrsCdmGrpsNoData = pUeGrpPrmsCpu[ueGrpIdx].nDmrsCdmGrpsNoData;
      uint8_t nDataSym           = pUeGrpPrmsCpu[ueGrpIdx].nDataSym;
      uint8_t nDmrsSyms          = pUeGrpPrmsCpu[ueGrpIdx].nDmrsSyms;
      uint8_t nSym               = 0;
      if(nDmrsCdmGrpsNoData==1){
         nSym = nDataSym + nDmrsSyms;
      }else{
         nSym = nDataSym;
      }
      if(nSym > MAX_N_SYMS)
      MAX_N_SYMS = nSym;
   }
   //printf("MAX_N_DATA_SYMS[%d]\n", MAX_N_DATA_SYMS);
   // launch geometry
   dim3 blockDim(12, MAX_N_SYMS);  // One thread block covers one entire PRB: 12 subcarriers x MAX_N_DATA_SYMS
   dim3 gridDim(MAX_N_PRBS, nUciUes);  

   // kernel (only one kernel option for now)
   void* kernelFunc = reinterpret_cast<void*>(uciOnPuschSegLLRs0Kernel);
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
   kernelNodeParamsDriver.sharedMemBytes = 0;
}


void computePerUciPrms0(perUciPrms0_t& perUciPrms, PerTbParams& uePrms, cuphyPuschRxUeGrpPrms_t& ueGrpPrms)
{

   // User parameters:
   uint32_t G_harq_rvd  = uePrms.G_harq_rvd;
   uint32_t G_harq      = uePrms.G_harq;
   uint32_t G_csi1      = uePrms.G_csi1;
   uint32_t nLayers     = uePrms.Nl;
   uint32_t nBitsPerQam = uePrms.Qm;
   uint16_t nBitsHarq   = uePrms.nBitsHarq;
   uint8_t  nBitsPerRe  = nLayers * nBitsPerQam;
   
   // User group parameters:
   uint8_t  nDataSym    =  ueGrpPrms.nDataSym;
   uint8_t  nDmrsSyms   =  ueGrpPrms.nDmrsSyms;
   uint8_t* pDataSymLoc =  ueGrpPrms.dataSymLoc;
   uint8_t* pDmrsSymLoc =  ueGrpPrms.dmrsSymLoc;
   uint16_t nPrb        =  ueGrpPrms.nPrb;
   uint8_t nDmrsCdmGrpsNoData = ueGrpPrms.nDmrsCdmGrpsNoData;

   // Allocation parameters:
   uint32_t   numUnassignedBitsInSymbols[nDataSym];
   uint16_t   numUnassignedResInSymbols[nDataSym];
   reGrid0_t* pRvdHarqReGrids    = perUciPrms.rvdHarqReGrids;
   reGrid0_t* pHarqReGrids       = perUciPrms.harqReGrids;
   reGrid0_t* pCsi1ReGrids       = perUciPrms.csi1ReGrids;
   uint32_t*  pDescramOffsets    = perUciPrms.descramOffsets;
   uint8_t&   nSym               = perUciPrms.nSym;
   bool*      pDmrsFlags         = perUciPrms.dmrsFlags;
   uint32_t*  pSchRmBuffOffsets  = perUciPrms.schRmBuffOffsets;

   // Determine number of symbols containing SCH and/or UCI
   if(nDmrsCdmGrpsNoData == 1){
      nSym = nDataSym + nDmrsSyms;
   }else{
      nSym = nDataSym;
   }

   // Determine first symbol in slot containing SCH and/or UCI
   uint8_t startSymInSlot = 0;
   if(nDmrsCdmGrpsNoData == 1){
      startSymInSlot = std::min(pDataSymLoc[0], pDmrsSymLoc[0]);
   }else{
      startSymInSlot = pDataSymLoc[0];
   }

   // Determine which symbols contain DMRS
   for(int symIdx = 0; symIdx < nSym; ++symIdx){
      pDmrsFlags[symIdx] = false;
   }

   if(nDmrsCdmGrpsNoData == 1)
   {
      for(int i = 0; i < nDmrsSyms; ++i)
      {
         uint8_t dmrsSymIdx_within_schAndUciSymbols    = pDmrsSymLoc[i] - startSymInSlot;
         pDmrsFlags[dmrsSymIdx_within_schAndUciSymbols] = true;
      }
   }

   // Determine maxLength
   uint8_t maxLength = 1;
   if(nDmrsSyms > 1){
      if(pDmrsSymLoc[1] == (pDmrsSymLoc[0] + 1)){
         maxLength = 2;
      }
   }

   // Determine starting HARQ symbol
   uint8_t startSymHarq = pDmrsSymLoc[0] - startSymInSlot;
   if(nDmrsCdmGrpsNoData == 1){
      startSymHarq = pDmrsSymLoc[0] - startSymInSlot + maxLength;
   }

   //printf("startDataSymHarq[%d]\n", startDataSymHarq);
   // Determine if HARQ puncturing used
   if((0 < nBitsHarq) && (nBitsHarq < 3)){
      perUciPrms.harqPunctFlag = true;
   }else{
      perUciPrms.harqPunctFlag = false;
      G_harq_rvd               = G_harq;
   }

   // Determine if one bit simplex:
   uint8_t harqSpx1Flag = (nBitsHarq == 1) ? 1 : 0;

   // Determine starting CSI data symbol (always 0 now)
   uint8_t startSymCsi1 = 0;

   // Determine starting SCH data symbol (always 0 now)
   uint8_t startSymSch = 0;

   // Initialize number of unassigned Res and bits
   uint16_t nResPerSym  = nPrb * 12;
   uint32_t nBitsPerSym = nPrb * 12 * nBitsPerRe;
   for(int symIdx = 0; symIdx < nSym; ++symIdx)
   {
      if(pDmrsFlags[symIdx]){
         numUnassignedResInSymbols[symIdx]  = 0;
         numUnassignedBitsInSymbols[symIdx] = 0;
      }else{
         numUnassignedResInSymbols[symIdx]  = nResPerSym;
         numUnassignedBitsInSymbols[symIdx] = nBitsPerSym;
      }
   }

   // Initalize grid values
   for(int symIdx = 0; symIdx < nSym; ++symIdx)
   {
      pRvdHarqReGrids[symIdx] = std::move(reGrid0_t());
      pHarqReGrids[symIdx]    = std::move(reGrid0_t());
      pCsi1ReGrids[symIdx]    = std::move(reGrid0_t());
   }

   // Determine RE grids reserved for HARQ
   uint32_t nAssignedHarqRvdRmBits = 0;
   for(int symIdx = startSymHarq; symIdx < nSym; ++symIdx)
   {
      if(nAssignedHarqRvdRmBits >= G_harq_rvd)
         break;
      if(numUnassignedResInSymbols[symIdx]  == 0)
         continue;

      assignReGrid0(pRvdHarqReGrids[symIdx], 
                    numUnassignedResInSymbols[symIdx], 
                    numUnassignedBitsInSymbols[symIdx],
                    nAssignedHarqRvdRmBits,
                    G_harq_rvd,
                    nBitsPerRe);

      updateNumUnassigned0(pRvdHarqReGrids[symIdx],
                           numUnassignedResInSymbols[symIdx],
                           numUnassignedBitsInSymbols[symIdx],
                           nBitsPerRe);
   }

   // HARQ rateMatched bits assigned to sub-grid of reserved HARQ grid
   uint32_t nAssignedHarqRmBits = 0;
   for(int symIdx = startSymHarq; symIdx < nSym; ++symIdx)
   {
      if(nAssignedHarqRmBits >= G_harq)
         break;

      uint32_t nRvdRes  = pRvdHarqReGrids[symIdx].nRes;
      uint32_t nRvdBits = nRvdRes * nBitsPerRe; 
      if(nRvdRes == 0)
         continue;

      assignReGrid0(pHarqReGrids[symIdx], 
                   nRvdRes, 
                   nRvdBits,
                   nAssignedHarqRmBits,
                   G_harq,
                   nBitsPerRe);
   }

   // Determine CSI-P1 RE grid
   uint32_t nAssignedCsi1RmBits = 0;
   for(int symIdx = startSymCsi1; symIdx < nSym; ++symIdx)
   {
      if(nAssignedCsi1RmBits >= G_csi1)
         break;
      if(numUnassignedResInSymbols[symIdx]  == 0)
         continue;

      assignReGrid0(pCsi1ReGrids[symIdx], 
                     numUnassignedResInSymbols[symIdx], 
                     numUnassignedBitsInSymbols[symIdx],
                     nAssignedCsi1RmBits,
                     G_csi1,
                     nBitsPerRe);
   }

   // update number of RE avaliable to SCH
   for(int symIdx = 0; symIdx < nSym; ++symIdx)
   {
      if(perUciPrms.harqPunctFlag) // If HARQ puncturing enabled, REs assigned to HARQ are also "assigned" to SCH or CSI-P2.
      {
         numUnassignedResInSymbols[symIdx]  = nResPerSym;
         numUnassignedBitsInSymbols[symIdx] = nBitsPerSym;
      }
      
      updateNumUnassigned0(pCsi1ReGrids[symIdx],
                          numUnassignedResInSymbols[symIdx],
                          numUnassignedBitsInSymbols[symIdx],
                          nBitsPerRe);
   }
   
   // Determine SCH rm bufffer offsets:
   uint32_t nAssignedSchRmBits = 0;
   for(int symIdx = startSymSch; symIdx < nSym; ++symIdx)
   {
      pSchRmBuffOffsets[symIdx] = nAssignedSchRmBits;
      if(pDmrsFlags[symIdx])
      {
         nAssignedSchRmBits += nBitsPerSym >> 1;
      }else
      {
         nAssignedSchRmBits += numUnassignedBitsInSymbols[symIdx];
      }
   }

   // Determine descrambling offsets:
   uint32_t totalNumSchAndUciBits = 0;
   for(int symIdx = 0; symIdx < nSym; ++symIdx) 
   {
      pDescramOffsets[symIdx] = totalNumSchAndUciBits;
      if(pDmrsFlags[symIdx])
      {
         totalNumSchAndUciBits += nBitsPerSym >> 1;
      }else
      {
         totalNumSchAndUciBits += nBitsPerSym;
      }
   }



   
   //printf("nAssignedSchRmBits[%d]\n", nAssignedSchRmBits);

   // Wrap:
   perUciPrms.nBitsPerRe   = nBitsPerRe;
   perUciPrms.harqSpx1Flag = harqSpx1Flag;
}


void  uciOnPuschSegLLRs0::setup( uint16_t                             nUciUes,
                                 uint16_t*                            pUciUeIdxs,
                                 PerTbParams*                         pTbPrmsCpu,
                                 PerTbParams*                         pTbPrmsGpu,
                                 uint16_t                             nUeGrps,
                                 cuphyTensorPrm_t*                    pTensorPrmsEqOutLLRs,
                                 cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsCpu, 
                                 cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsGpu,
                                 cuphyUciToSeg_t                      uciToSeg,               
                                 uciOnPuschSegLLRs0DynDescr_t*        pCpuDynDesc,
                                 void*                                pGpuDynDesc,
                                 uint8_t                              enableCpuToGpuDescrAsyncCpy,
                                 cuphyUciOnPuschSegLLRs0LaunchCfg_t*  pLaunchCfg,
                                 cudaStream_t                         strm)
{
   // pipeline parameters
   pCpuDynDesc->pUePrmsGpu    = pTbPrmsGpu;
   pCpuDynDesc->pUeGrpPrmsGpu = pUeGrpPrmsGpu;
   pCpuDynDesc->uciToSeg      = uciToSeg;

   // input buffer addrs
   for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
   {
      auto                     tensorDesc       = static_cast<tensor_desc&> (*pTensorPrmsEqOutLLRs[ueGrpIdx].desc);
      const tensor_layout_any& tEqOutLLRsLayout = tensorDesc.layout();
      void*                    tEqOutLLRsAddr   = pTensorPrmsEqOutLLRs[ueGrpIdx].pAddr;

      pCpuDynDesc->tEqOutLLRs[ueGrpIdx] = std::move(tensor_ref_any<CUPHY_R_16F>(tEqOutLLRsAddr, tEqOutLLRsLayout.dimensions.begin(), tEqOutLLRsLayout.strides.begin()));
   }
   

   uint16_t nSelectedUciUes = 0;

   for(int uciIdx = 0; uciIdx < nUciUes; ++uciIdx)
   {
      uint16_t ueIdx = pUciUeIdxs[uciIdx];
      if(((uciToSeg == SEG_ONLY_EARLY_UCI) && (pTbPrmsCpu[ueIdx].isEarlyHarq)) || (uciToSeg == SEG_ALL_UCI))
      {
         // Indicies:
         uint16_t ueGrpIdx = pTbPrmsCpu[ueIdx].userGroupIndex;
         pCpuDynDesc->uciToUserMap[nSelectedUciUes].ueIdx    = ueIdx;
         pCpuDynDesc->uciToUserMap[nSelectedUciUes].ueGrpIdx = ueGrpIdx;

         // Compute per UCI parameters:
         computePerUciPrms0(pCpuDynDesc->perUciPrmsArray[nSelectedUciUes], pTbPrmsCpu[ueIdx], pUeGrpPrmsCpu[ueGrpIdx]);

         nSelectedUciUes += 1;
      }
   }

   // save pointer to GPU descriptor
   uciOnPuschSegLLRs0KernelArgs_t& kernelArgs = m_kernelArgs;
   kernelArgs.pDynDescr = reinterpret_cast<uciOnPuschSegLLRs0DynDescr_t*>(pGpuDynDesc);

   // Optional descriptor copy to GPU memory
   if(enableCpuToGpuDescrAsyncCpy)
   {
      cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(uciOnPuschSegLLRs0DynDescr_t), cudaMemcpyHostToDevice, strm);
   }

   // select kernel (includes launch geometry). Populate launchCfg.
   kernelSelect(nSelectedUciUes, pCpuDynDesc->uciToUserMap, pUeGrpPrmsCpu, pLaunchCfg);
   

   pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
   pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);

   // reGrid0_t* pHarqRvdReGrid = pCpuDynDesc->perUciPrmsArray[0].rvdHarqReGrids;
   // for(int symIdx = 0; symIdx < 13; ++symIdx)
   // {
   //    printf("\n symIdx                         = %ld", symIdx);
   //    printf("\n reGrid_harq_rvd.nRes           = %ld", pHarqRvdReGrid[symIdx].nRes);
   //    printf("\n reGrid_harq_rvd.ReStride       = %ld", pHarqRvdReGrid[symIdx].ReStride);
   //    printf("\n reGrid_harq_rvd.rmBufferOffset = %ld", pHarqRvdReGrid[symIdx].rmBufferOffset);
   //    printf("\n");
   // }
   
   // reGrid0_t* pHarqReGrid = pCpuDynDesc->perUciPrmsArray[0].harqReGrids;
   // for(int symIdx = 0; symIdx < 13; ++symIdx)
   // {
   //    printf("\n symIdx                         = %ld", symIdx);
   //    printf("\n reGrid_harq.nRes           = %ld", pHarqReGrid[symIdx].nRes);
   //    printf("\n reGrid_harq.ReStride       = %ld", pHarqReGrid[symIdx].ReStride);
   //    printf("\n reGrid_harq.rmBufferOffset = %ld", pHarqReGrid[symIdx].rmBufferOffset);
   //    printf("\n");
   // }

   // reGrid0_t* pCsi1ReGrid = pCpuDynDesc->perUciPrmsArray[0].csi1ReGrids;
   // for(int symIdx = 0; symIdx < 13; ++symIdx)
   // {
   //    printf("\n symIdx                     =  %ld", symIdx);
   //    printf("\n reGrid_csi1.nRes           =  %ld", pCsi1ReGrid[symIdx].nRes);
   //    printf("\n reGrid_csi1.ReStride       =  %ld", pCsi1ReGrid[symIdx].ReStride);
   //    printf("\n reGrid_csi1.rmBufferOffset =  %ld", pCsi1ReGrid[symIdx].rmBufferOffset);
   //    printf("\n");
   // }
}


void uciOnPuschSegLLRs0::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
   dynDescrSizeBytes  = sizeof(uciOnPuschSegLLRs0DynDescr_t);
   dynDescrAlignBytes = alignof(uciOnPuschSegLLRs0DynDescr_t);
}