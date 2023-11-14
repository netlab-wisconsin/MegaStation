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
#include "descrambling.cuh"
#include "uciOnPusch_segLLRs1.hpp"

#include <cuda_fp16.h>

#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>

#include <stdio.h>

#include <functional>

#include <cuda/std/tuple>

#define _DEV_INLINE __device__ __forceinline__

namespace cg = cooperative_groups;

//// UL-SCH and UCI mapping grid
constexpr uint32_t kSymbolMax = 14;

template <typename _FloatType>
struct float_to_integral;

template <>
struct float_to_integral<double> {
    using type = uint64_t;
    static constexpr type sign_bit = 0x8000000000000000;
};
template <>
struct float_to_integral<float> {
    using type = uint32_t;
    static constexpr type sign_bit = 0x80000000;
};
template <>
struct float_to_integral<__half> {
    using type = uint16_t;
    static constexpr type sign_bit = 0x8000;
};

template <typename _FloatType>
using float_to_integral_t = typename float_to_integral<_FloatType>::type;

template <typename _FloatType>
__device__ inline _FloatType descramble_toggle_sign(_FloatType llr_input) {
   // sign bit of float/half/double
   constexpr auto sign_bit = float_to_integral<_FloatType>::sign_bit;
   float_to_integral_t<_FloatType> float_pattern;

   memcpy(&float_pattern, &llr_input, sizeof(_FloatType));

   float_pattern ^= sign_bit;

   memcpy(&llr_input, &float_pattern, sizeof(_FloatType));

   return llr_input;
}

constexpr _DEV_INLINE uint32_t getLLrAckOffset(const uciRvdStride* stride, uint32_t symbolIdx) {
   uint32_t count = 0;

   for (uint32_t i = 0; i < symbolIdx; i++) {
      count += stride[i].rvdCount;
   }

   return count;
}


constexpr _DEV_INLINE uint8_t countDmrsSymOffset(const uint8_t* symbolArray, uint8_t symbolCount, uint8_t symbolIdx) {
   uint8_t count = 0;

   for (uint32_t i = 0; i < symbolCount; i++) {
      if (symbolIdx > symbolArray[i]) {
         count++;
      }
   }

   return count;
}

constexpr _DEV_INLINE bool isDmrsSym(const uint8_t* symbolArray, uint8_t symbolCount, uint8_t symbolIdx) {
   for (uint32_t i = 0; i < symbolCount; i++) {
      if (symbolIdx == symbolArray[i]) {
         return true;
      }
   }

   return false;
}

template <typename _Tp>
constexpr _Tp cuphy_gcd(_Tp a, _Tp b) {
   while (true) {
      if (a == 0) return b;
      b %= a;
      if (b == 0) return a;
      a %= b;
   }
}

template <typename _Tp>
constexpr _Tp cuphy_lcm(_Tp a, _Tp b) {
   _Tp gcd = cuphy_gcd(a, b);
   if (gcd) {
      return a / gcd * b;
   }
   return 0;
}

__launch_bounds__(12*14)
__global__ void uciOnPuschSegLLRs1Kernel(uciOnPuschSegLLRs1DynDescr_t* pDesc)
{
   // Expected thread mapping
   // threadIdx.x
   /*
   auto cta = cg::this_thread_block();
   auto w32 = cg::tiled_partition<32>(cta);
   auto w16 = cg::tiled_partition<16>(cta);
   */

   // Expected Launch Config:

   // CTA indices
   const uint8_t  subCarrierIdx  = threadIdx.x;
   const uint8_t  puschSymIdx    = threadIdx.y;

   // PRB offset
   const auto  groupIndex     = blockIdx.y;
   const auto  prbIdx         = blockIdx.x;

   const auto  ueIdx          = pDesc->uciUserIdxs[groupIndex];
   const auto& perTbPrms      = pDesc->pTbPrms[ueIdx];

   // G_harq: rate matched sequence length for HARQ-ACK
   // G_harq_rvd: rate matched sequence length for HARQ reserved resources (nBitsHarq <=2)
   // G_csi1: rate matched sequence length for CSI part 1
   // G: rate matched sequence length for UL-SCH
   // LLRseq: size (Qm*nL*nPrb*12*nDataSymbols)
   // nPuschSym: number of symbols allocated for PUSCH (including DMRS)
   // startSym: Starting symbol index of PUSCH (MATLAB 1 indexing)
   // dataSymLoc_array: symbol index for PUSCH data (MATLAB 1 indexing)
   // dmrsSymLoc_array: symbol index for PUSCH DMRS (MATLAB 1 indexing)
   // nPrb: Number of PRBs allocated for PUSCH
   // Nl: Number of layers per UE
   // Qm: Modulation order
   // nBitsHarq: number of HARQ bits
   // N_id: dataScramblingId or N_ID^cell as described in Sec. 6.3.1.1, TS38.211
   // n_rnti: RNTI as described in Sec. 6.3.1.1, TS38.211
   const auto G               = perTbPrms.G;
   const auto G_harq          = perTbPrms.G_harq;
   const auto G_csi1          = perTbPrms.G_csi1;
   // const auto G_harq_rvd      = perTbPrms.G_harq_rvd;
   const auto nBitsHarq       = perTbPrms.nBitsHarq;
   const auto Nl              = perTbPrms.Nl;
   const auto Qm              = perTbPrms.Qm;
   const auto userGroupIndex  = perTbPrms.userGroupIndex;
   const auto NlQm            = Nl*Qm;
   const auto cinit           = perTbPrms.cinit;

   const auto nPrb           = pDesc->nPrbs[userGroupIndex];
   auto&      perTbLlr       = pDesc->tEqOutLLRs[userGroupIndex];

   const auto  nDmrsSym       = pDesc->nDmrsSym;
   const auto& dmrsSymLoc     = pDesc->dmrsSymIdxs;
   // const auto  nDataSym       = pDesc->nDataSym;
   // const auto& dataSymLoc     = pDesc->dataSymIdxs;

   // output buffers
   auto&       schLLRs        = perTbPrms.d_schAndCsi2LLRs;
   auto&       csi1LLRs       = perTbPrms.d_csi1LLRs;
   auto&       harqLLRs       = perTbPrms.d_harqLLrs;

   // This block has nothing to calculate
   if (prbIdx >= nPrb) {
      return;
   }

   const auto dmrsSkipLLR  = isDmrsSym(dmrsSymLoc, nDmrsSym, puschSymIdx);

   // This symbol has no data for us to gather
   if (dmrsSkipLLR) {
      return;
   }

   const auto prevDmrsSym  = countDmrsSymOffset(dmrsSymLoc, nDmrsSym, puschSymIdx);

   const auto llrSeqSymOffset = (nPrb * 12 * (puschSymIdx - prevDmrsSym));
   const auto prbOffset = (prbIdx * 12) + subCarrierIdx;

   const __half nullLLR = __half(0);

   uint32_t descSeq = descrambling::gold32n(cinit, (llrSeqSymOffset + prbOffset) * NlQm);

   const auto& harqRvdStride = pDesc->harqRvdStride[groupIndex].strideMap[puschSymIdx];
   const auto& harqUciStride = pDesc->harqUciStride[groupIndex].strideMap[puschSymIdx];
   const auto& csi1RvdStride = pDesc->csi1RvdStride[groupIndex].strideMap[puschSymIdx];
   const auto& harqAckStride = pDesc->harqAckStride[groupIndex].strideMap[puschSymIdx];

   // Indices of a symbol where harq intersects a CSI stride
   const uint16_t harqRvd_Csi1Lcm = pDesc->harqRvd_CsiLcms[groupIndex].lcmMap[puschSymIdx];

   const uint32_t harqRvdLLRsOffset = getLLrAckOffset(pDesc->harqRvdStride[groupIndex].strideMap, puschSymIdx);
   const uint32_t harqUciLLRsOffset = getLLrAckOffset(pDesc->harqUciStride[groupIndex].strideMap, puschSymIdx);
   const uint32_t csi1RvdLLRsOffset = getLLrAckOffset(pDesc->csi1RvdStride[groupIndex].strideMap, puschSymIdx);
   const uint32_t harqAckLLRsOffset = getLLrAckOffset(pDesc->harqAckStride[groupIndex].strideMap, puschSymIdx);

   // Max subcarrier/nlqm that can be a csi1/harq bit
   const uint32_t csi1RvdMax    = ((csi1RvdStride.rvdCount-1)*csi1RvdStride.rvdStride);
   const uint32_t harqRvdMax    = ((harqRvdStride.rvdCount-1)*harqRvdStride.rvdStride);
   const uint32_t harqUciRvdMax = ((harqUciStride.rvdCount-1)*harqUciStride.rvdStride);
   const uint32_t harqAckRvdMax = ((harqAckStride.rvdCount-1)*harqAckStride.rvdStride);

   /*************************************************************************************************
   * The below is mostly setup:
   *   Create functors that are used for creating a hopefully clearer picture of the
   *     algorithm during harq/csi/ulsch demux
   **************************************************************************************************/

   // Descramble functor
   auto descramble = [descSeq](const auto llr, const int8_t nlqmOffset, const int8_t offset = 0) {
      if (descSeq & (1 << (nlqmOffset + offset)))
         return descramble_toggle_sign(llr);
      return llr;
   };

   // Harq-UCI (non-puncturing)
   auto isHarqUciBit = [&](const uint32_t offset) -> bool {
      return harqUciRvdMax &&
             (offset <= harqUciRvdMax) &&
             (offset % harqUciStride.rvdStride == 0);
   };
   auto assignHarqUci = [&](const uint32_t offset, const auto llr, const int8_t nlqmOffset) {
      uint32_t index = offset / harqUciStride.rvdStride;
      index += harqUciLLRsOffset;
      index *= NlQm;
      index += nlqmOffset;

      if(index < G_harq){
         harqLLRs[index] = descramble(llr, nlqmOffset);
      }
   };
   auto prbOffsetHarq_Csi = [&](const uint32_t offset) -> uint32_t {
      // We only offset CSI when it gets offset by a harq bit
      if (harqRvdMax && harqRvd_Csi1Lcm) {
         const uint32_t maxNumIntersected = (harqRvdMax / harqRvd_Csi1Lcm);
         const uint32_t csi1RvdIntersected = (offset / harqRvd_Csi1Lcm);
         return min(maxNumIntersected, csi1RvdIntersected) + 1;
      }
      else if (harqUciRvdMax) {
         const uint32_t harqUciCeil = min(offset, harqUciRvdMax);
         return (harqUciCeil / harqUciStride.rvdStride) + 1;
      }
      return 0;
   };
   auto prbOffsetHarq_Sch = [&](const uint32_t offset) -> uint32_t {
      // Always offset ulsch bits when harq UCI bits are interspersed
      if (harqUciRvdMax) {
         const uint32_t harqUciCeil = min(offset, harqUciRvdMax);
         return (harqUciCeil / harqUciStride.rvdStride) + 1;
      }
      return 0;
   };

   // CSI1 bits and offsets
   auto isCsi1Bit = [&](const uint32_t offset) -> bool {
      return csi1RvdMax &&
             (offset <= csi1RvdMax) &&
             (offset % csi1RvdStride.rvdStride == 0);
   };
   auto assignCsi1Rvd = [&](const uint32_t offset, const auto llr, const int8_t nlqmOffset) {
      uint32_t index = offset / csi1RvdStride.rvdStride;

      index += csi1RvdLLRsOffset;
      index *= NlQm;
      index += nlqmOffset;

      if (index < G_csi1) {
         csi1LLRs[index] = descramble(llr, nlqmOffset);
      }
   };
   auto prbOffsetCsi1_Sch = [&](const uint32_t offset) -> uint32_t {
      // Always offset ulsch bits when csi1 bits are present
      if (csi1RvdMax) {
         return (min(offset, csi1RvdMax) / csi1RvdStride.rvdStride) + 1;
      }
      return 0;
   };

   // ULSCH assignment
   auto assignSch = [&](const uint32_t offset, const auto llr, const int8_t nlqmOffset) {
      // Add all prior symbol's SCH elements to this offset
      uint32_t index = offset + llrSeqSymOffset;
      // Remove all previous symbols' harq and csi1 bits from offset
      index -= harqUciLLRsOffset;
      index -= csi1RvdLLRsOffset;
      // Apply NlQm before adding offset
      index *= NlQm;
      index += nlqmOffset;

      // Assign LLR
      if (index < G) {
         schLLRs[index] = descramble(llr, nlqmOffset);
      }
   };

   // Harq/HarqACK bits and indexing
   auto isHarqRvdBit = [&](const uint32_t offset) -> bool {
      return harqRvdMax &&
             (offset <= (harqRvdMax)) &&
             (offset % (harqRvdStride.rvdStride) == 0);
   };
   auto assignHarqRvd = [&](const uint32_t offset, const auto llr, const int8_t nlqmOffset) {
      uint32_t index = offset / harqRvdStride.rvdStride;
      index += harqRvdLLRsOffset;
      index *= NlQm;
      index += nlqmOffset;
      // DESCRAMBLING IS DONE OUTSIDE
      if(index < G_harq){
         harqLLRs[index] = llr;
      }
   };


   // Binding harq assignments for n_harq_bits
   auto assignHarqRvd_n_harq_1 = [&](const uint32_t offset, const auto llr, const int8_t nlqmOffset, const int8_t qmIdx) {
      if (Qm == 1) {
         assignHarqRvd(offset, descramble(llr, nlqmOffset), nlqmOffset);
      }
      // TODO: Specific hackery for matlab harq ack descramble (uciUlschDemuxDescram.m ln 421)
      // Qm(2) % 2 cannot equal two, so the bit is skipped?
      else {
         if (qmIdx == 0) {
            assignHarqRvd(offset, descramble(llr, nlqmOffset), nlqmOffset);
         }
         else if (qmIdx == 1 && Qm > 2) {
            assignHarqRvd(offset, descramble(llr, nlqmOffset, -1), nlqmOffset);
         }
      }
   };
   auto assignHarqRvd_n_harq_2 = [&](const uint32_t offset, const auto llr, const int8_t nlqmOffset, const int8_t qmIdx) {
      if (Qm == 1 || Qm == 2) {
         assignHarqRvd(offset, descramble(llr, nlqmOffset), nlqmOffset);
      }
      else {
         if (qmIdx == 0 || qmIdx == 1) {
            assignHarqRvd(offset, descramble(llr, nlqmOffset), nlqmOffset);
         }
      }
   };

   // Check harqAck stride against harqRvd to check if punctured
   auto isHarqRvdPunctured = [&](const uint32_t offset) -> bool {
      if (harqAckRvdMax) {
         uint32_t harqIndex = offset / harqRvdStride.rvdStride;
         return (harqIndex <= harqAckRvdMax) &&
                (harqIndex % harqAckStride.rvdStride) == 0;
      }
      return false;
   };

   /*************************************************************************************************
   * Actual placements begin here:
   *   In order to check indices by stride, we prioritize harq bits
   *   After harq, we remove prior bits from the true index so that CSI1 can check its stride correctly
   *   After CSI1, we perform the same step removing bits so that SCH bits are placed correctly
   **************************************************************************************************/
   const bool prbHarqUciBit = isHarqUciBit(prbOffset);
   const bool prbHarqRvdBit = isHarqRvdBit(prbOffset);
   const bool prbIsPunctured = isHarqRvdPunctured(prbOffset);
   // Remove indices intersected by harqRvd
   // example: csi_stride(5)+harq_stride(3) -> at prbIndex 15, the index would belong to harq, CSI gets index 16 by removing prior LCMs
   const int32_t csiIndex = prbOffset - prbOffsetHarq_Csi(prbOffset);

   const bool prbCsi1RvdBit = isCsi1Bit(csiIndex);

   // Remove both harq-uci and csi indices
   const int32_t schIndex = prbOffset - prbOffsetHarq_Sch(prbOffset) - prbOffsetCsi1_Sch(csiIndex);

   for (int8_t nlIdx = 0; nlIdx < Nl; nlIdx++) {
      for (int8_t qmIdx = 0; qmIdx < Qm; qmIdx++) {
         // load LLR irrespective of the deduced offset.
         const auto& llr = perTbLlr({static_cast<int>(qmIdx), static_cast<int>(nlIdx), static_cast<int>(prbOffset), puschSymIdx - prevDmrsSym});
         const int8_t nlqmOffset = qmIdx + (Qm*nlIdx);

         if (prbHarqUciBit) {
            assignHarqUci(prbOffset, llr, nlqmOffset);
         }
         else if (prbHarqRvdBit) {
            if (nBitsHarq == 1) {
               assignHarqRvd_n_harq_1(prbOffset, llr, nlqmOffset, qmIdx);
               if (!prbIsPunctured) {
                  assignSch(prbOffset, llr, nlqmOffset);
               }
            }
            else if (nBitsHarq == 2) {
               assignHarqRvd_n_harq_2(prbOffset, llr, nlqmOffset, qmIdx);
               if (!prbIsPunctured) {
                  assignSch(prbOffset, llr, nlqmOffset);
               }
            }
            else {
               assignHarqRvd(prbOffset, llr, nlqmOffset);
               assignSch(prbOffset, llr, nlqmOffset);
            }
         }
         else if (prbCsi1RvdBit) {
            assignCsi1Rvd(csiIndex, llr, nlqmOffset);
         }
         else {
            assignSch(schIndex, llr, nlqmOffset);
         }
      }
   }
}

static inline uint8_t calculateL1(const uint8_t                       l1Csi,
                                  const uciOnPuschSegLLRs1DynDescr_t* pCpuDynDesc)
{
   const auto& startSym      = pCpuDynDesc->startSym;
   // const auto& nPuschDataSym = pCpuDynDesc->nDataSym;
   const auto& dataSymIdxs   = pCpuDynDesc->dataSymIdxs;
   const auto& nPuschDmrsSym = pCpuDynDesc->nDmrsSym;
   const auto& dmrsSymIdxs   = pCpuDynDesc->dmrsSymIdxs;

   if (nPuschDmrsSym) {
      auto dmrsIter = std::find_if(std::begin(dataSymIdxs), std::end(dataSymIdxs), [firstDmrsIdx = dmrsSymIdxs[0], startSym](auto dataSymIdx) {
         return (firstDmrsIdx - startSym) < (dataSymIdx - startSym);
      });

      if (dmrsIter != std::end(dataSymIdxs))
         return *dmrsIter;
   }
   return l1Csi;
}

static void buildCodewordMap(const uint16_t                      nUciUes,
                             const uint16_t*                     pUciUserIdxs,
                             const PerTbParams&                  tbPrmsCpu,
                             const uint16_t                      numPrbs,
                             const uciOnPuschSegLLRs1DynDescr_t* pCpuDynDesc,
                             uciRvdStrideArray&                  harqRvdStride,
                             uciRvdStrideArray&                  harqUciStride,
                             uciRvdStrideArray&                  csi1RvdStride,
                             uciRvdStrideArray&                  harqAckStride,
                             uciRvdLcmArray&                     harqRvd_CsiLcms)
{
   // Zero out stride maps per symbol
   memset(&harqRvdStride, 0, sizeof(harqRvdStride));
   memset(&harqUciStride, 0, sizeof(harqUciStride));
   memset(&csi1RvdStride, 0, sizeof(csi1RvdStride));
   memset(&harqAckStride, 0, sizeof(harqAckStride));

   std::array<uint32_t, kSymbolMax> mScUci{};
   mScUci.fill(numPrbs * 12);

   const auto& startSym      = pCpuDynDesc->startSym;
   const auto& nPuschSym     = pCpuDynDesc->nPuschSym;
   const auto& nPuschDataSym = pCpuDynDesc->nDataSym;
   const auto& dataSymIdxs   = pCpuDynDesc->dataSymIdxs;
   const auto& nPuschDmrsSym = pCpuDynDesc->nDmrsSym;
   const auto& dmrsSymIdxs   = pCpuDynDesc->dmrsSymIdxs;

   for (uint32_t i = 0; i < nPuschDmrsSym; i++) {
      mScUci[dmrsSymIdxs[i] - startSym] = 0;
   }

   auto mScUlsch = mScUci;

   // OFDM symbol index of the first OFDM that does not carry DMRS
   const uint8_t l1Csi = (nPuschDataSym) ? dataSymIdxs[0] - startSym : 0;
   const uint8_t l1 = calculateL1(l1Csi, pCpuDynDesc);

   const uint32_t GAck1      = tbPrmsCpu.G_harq; // Number of coded harq-ack bits
   const uint32_t GCsiPart11 = tbPrmsCpu.G_csi1; // Number of coded CSI part 1 bits
   constexpr uint32_t nHopPusch  = 1;

   const uint16_t NlQm = tbPrmsCpu.Nl * tbPrmsCpu.Qm;

   constexpr uint32_t GAck2      = 0;  // No freq. hopping
   constexpr uint32_t GCsiPart12 = 0;  // No freq. hopping
   constexpr uint32_t l2         = 0;  // No freq. hopping
   constexpr uint32_t l2Csi      = 0;  // No freq. hopping

   std::array<uint32_t, kSymbolMax> phiBarRvdStride{};
   std::array<uint32_t, kSymbolMax> mBarPhiBarScRvd{};
   std::array<uint32_t, kSymbolMax> mBarPhiCsiStride{};

   // Step 1 (HARQ Reserved bits)
   if (tbPrmsCpu.G_harq_rvd) {
      uint32_t GAckRvd1 = tbPrmsCpu.G_harq_rvd;
      uint32_t GAckRvd2 = 0;
      uint32_t GAckRvdVal[2] = {GAckRvd1, GAckRvd2};
      uint32_t lPrime[2] = {l1, l2};
      uint32_t mCountAck[2] = {0, 0};

      for (uint32_t i = 0; i < nHopPusch; i++) {
         uint32_t l = lPrime[i];
         while (mCountAck[i] < GAckRvdVal[i]) {
            if (l > nPuschSym) // Symbol index cannot be more than number of PUSCh symbols
                  break;

            if (mScUci[l] > 0) {
               uint32_t GAckRvdDiff = GAckRvdVal[i] - mCountAck[i];  // Number of remaining reserved elements

               uint16_t d = static_cast<uint16_t>(std::floor(((float)mScUci[l]*NlQm) / GAckRvdDiff));
               uint16_t mReCount = static_cast<uint16_t>(std::ceil((float)GAckRvdDiff/NlQm));

               if (GAckRvdDiff >= (mScUci[l] * NlQm)) {
                  d = 1;
                  mReCount = mScUlsch[l];
               }

               const uint16_t countAck = mReCount * NlQm;
               mCountAck[i] += countAck;
               harqRvdStride.strideMap[l] = uciRvdStride{d, mReCount};
            }
            l++;
         }
      }
   }

   // Step 2 (HARQ bits > 2)
   if (!tbPrmsCpu.G_harq_rvd && tbPrmsCpu.G_harq) {
      uint32_t mCountAck[2] = {0, 0};
      uint32_t mCountAckAll = 0;
      uint32_t lPrime[2] = {l1, l2};
      uint32_t GHarqAck[2] = {GAck1, GAck2};

      for (uint32_t i = 0; i < nHopPusch; i++) {
         uint32_t l = lPrime[i];
         while (mCountAck[i] < GHarqAck[i]) {
            if (l > nPuschSym) // Symbol index cannot be more than number of PUSCh symbols
               break;
            if (mScUci[l] > 0) {
               const uint32_t GAckDiff = GHarqAck[i] - mCountAck[i];
               const uint32_t mScUciNlQm = mScUci[l] * NlQm;

               uint16_t d = static_cast<uint16_t>(std::floor((float)mScUciNlQm / GAckDiff));
               uint16_t mReCount = static_cast<uint16_t>(std::ceil((float)GAckDiff/NlQm));
               if (GAckDiff >= mScUciNlQm) {
                  d = 1;
                  mReCount = mScUci[l];
               }

               const uint16_t countUci = mReCount*NlQm;
               harqUciStride.strideMap[l] = {d, mReCount};
               mCountAck[i] += countUci;
               mCountAckAll += countUci;

               mScUci[l]   -= mReCount;
               mScUlsch[l] -= mReCount;
            }
            l++;
         }
      }
   }

   // Step 3 (CSI Part 1 bits)
   if (tbPrmsCpu.G_csi1) {
      uint32_t mCountCsiPart1[2] = {0, 0};
      uint32_t mCountCsiPart1All = 0;
      uint32_t lPrimeCsi[2] = {l1Csi, l2Csi};
      uint32_t GCsiPart1[2] = {GCsiPart11, GCsiPart12};

      for (uint32_t i = 0; i < nHopPusch; i++) {
         auto l = lPrimeCsi[i];

         while ((mScUci[l] - mBarPhiBarScRvd[l]) <= 0) {
            l = l+1;
            if (l > nPuschSym) {
               break; // Symbol index cannot be larger than number of PUSCh symbols
            }
         }

         while (mCountCsiPart1[i] < GCsiPart1[i]) {
            if (l > nPuschSym) {
               break; // Symbol index cannot be larger than number of PUSCh symbols
            }
            auto mBarDiff = mScUci[l] - mBarPhiBarScRvd[l];
            auto GCsi1Diff = GCsiPart1[i] - mCountCsiPart1[i];

            if (mBarDiff > 0) {
               uint16_t d = 1;
               uint16_t mReCount = mBarDiff;
               if (GCsi1Diff < mBarDiff*NlQm) {
                  d = static_cast<uint16_t>(std::floor(float(mBarDiff * NlQm) / GCsi1Diff));
                  mReCount = static_cast<uint16_t>(std::ceil((float)GCsi1Diff / NlQm));
               }

               auto phiBarDiff = [phiBarRvd = phiBarRvdStride[l]](uint32_t jd) {
                  if (phiBarRvd == 0)
                     return jd;
                  return jd + (jd / phiBarRvd) + 1;
               };

               const uint16_t csiCount_Harq = mReCount - harqRvdStride.strideMap[l].rvdCount;
               const uint16_t csiCount = csiCount_Harq * NlQm;

               mCountCsiPart1All += csiCount;
               mCountCsiPart1[i] += csiCount;

               csi1RvdStride.strideMap[l] = {d, csiCount_Harq};
               // Find the LCM of the CSI stride and Harq Rvd strides
               harqRvd_CsiLcms.lcmMap[l] = cuphy_lcm(d, harqRvdStride.strideMap[l].rvdStride);

               mBarPhiCsiStride[l] = d;
               mScUci[l]   -= mReCount;
               mScUlsch[l] -= mReCount;
            }
            l++;
         }
      }
   }

   // Step 4 is part of the kernel

   // Step 5(HARQ-ACK bits <=2)

   if (tbPrmsCpu.G_harq_rvd && tbPrmsCpu.G_harq) {
      uint32_t mCountAck[2] = {0, 0};
      uint32_t lPrime[2] = {l1, l2};
      uint32_t gHarqAck[2] = {GAck1, GAck2};

      for (uint32_t i = 0; i < nHopPusch; i++) {
         uint32_t l = lPrime[i];

         uint32_t phiBarScRvd = harqRvdStride.strideMap[l].rvdCount;

         uint32_t stride = phiBarRvdStride[l];

         while (mCountAck[i] < gHarqAck[i]) {
            if (phiBarScRvd > 0) {
               uint32_t gAckDiff = gHarqAck[i] - mCountAck[i];
               uint16_t d = 1;
               uint16_t mReCount = phiBarScRvd;

               if (gAckDiff < phiBarScRvd * NlQm) {
                  d = static_cast<uint16_t>(std::floor(((float)phiBarScRvd * NlQm) / gAckDiff));
                  mReCount = static_cast<uint16_t>(std::ceil((float)gAckDiff / NlQm));
               }

               const uint16_t countAck = mReCount*NlQm;
               mCountAck[i] += countAck;
               harqAckStride.strideMap[l] = {d, mReCount};
            }
            l++;
         }
      }
   }
}


void uciOnPuschSegLLRs1::kernelSelect(uint16_t                            nUciUes,
                                      uint16_t*                           pUciUserIdxs,
                                      PerTbParams*                        pTbPrmsCpu,
                                      uint16_t*                           pNumPrbs,
                                      uint8_t                             nPuschDataSym,
                                      cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg)
{
   // determine max number of subcarriers
   uint16_t maxNumSubcarriers = 0;

   for(int i = 0; i < nUciUes; ++i)
   {
      uint16_t ueIdx          = pUciUserIdxs[i];
      uint16_t ueGrpIdx       = pTbPrmsCpu[ueIdx].userGroupIndex;
      uint16_t numSubcarriers = pNumPrbs[ueGrpIdx] * 12;

      if(numSubcarriers > maxNumSubcarriers) {
         maxNumSubcarriers = numSubcarriers;
      }
   }

   // launch geometry
   // One thread block covers one entire PRB: subcarrier * kMaxSymbols
   // Each block is launched with the max number of symbols
   dim3 blockDim(12, 14);
   // Each block per Uci handles one PRB.
   dim3 gridDim((maxNumSubcarriers) / 12, nUciUes);

    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(uciOnPuschSegLLRs1Kernel);
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



void uciOnPuschSegLLRs1::setup(cuphyUciOnPuschSegLLRs1Hndl_t        uciOnPuschSegLLRs1Hndl,
                               uint16_t                             nUciUes,
                               uint16_t*                            pUciUserIdxs,
                               PerTbParams*                         pTbPrmsCpu,
                               PerTbParams*                         pTbPrmsGpu,
                               uint16_t                             nUeGrps,
                               cuphyTensorPrm_t*                    pTensorPrmsEqOutLLRs,
                               uint16_t*                            pNumPrbs,
                               uint8_t                              startSym,
                               uint8_t                              nPuschSym,
                               uint8_t                              nPuschDataSym,
                               uint8_t*                             pDataSymIdxs,
                               uint8_t                              nPuschDmrsSym,
                               uint8_t*                             pDmrsSymIdxs,
                               uciOnPuschSegLLRs1DynDescr_t*        pCpuDynDesc,
                               void*                                pGpuDynDesc,
                               uint8_t                              enableCpuToGpuDescrAsyncCpy,
                               cuphyUciOnPuschSegLLRs1LaunchCfg_t*  pLaunchCfg,
                               cudaStream_t                         strm)
{
   // uci user parameters
   pCpuDynDesc->pTbPrms = pTbPrmsGpu;
   for(int uciIdx = 0; uciIdx <  nUciUes; ++uciIdx){
      pCpuDynDesc->uciUserIdxs[uciIdx] = pUciUserIdxs[uciIdx];
   }

   // input buffers
   for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
   {
      auto                     tensorDesc       = static_cast<tensor_desc&> (*pTensorPrmsEqOutLLRs[ueGrpIdx].desc);
      const tensor_layout_any& tEqOutLLRsLayout = tensorDesc.layout();
      void*                    tEqOutLLRsAddr   = pTensorPrmsEqOutLLRs[ueGrpIdx].pAddr;

      pCpuDynDesc->tEqOutLLRs[ueGrpIdx] = std::move(tensor_ref_any<CUPHY_R_16F>(tEqOutLLRsAddr, tEqOutLLRsLayout.dimensions.begin(), tEqOutLLRsLayout.strides.begin()));
      pCpuDynDesc->nPrbs[ueGrpIdx]      = pNumPrbs[ueGrpIdx];
   }

   // time allocation
   pCpuDynDesc->startSym  = startSym;
   pCpuDynDesc->nPuschSym = nPuschSym;
   pCpuDynDesc->nDataSym  = nPuschDataSym;
   pCpuDynDesc->nDmrsSym  = nPuschDmrsSym;
   memcpy(pCpuDynDesc->dataSymIdxs, pDataSymIdxs, nPuschSym);
   memcpy(pCpuDynDesc->dmrsSymIdxs, pDmrsSymIdxs, nPuschSym);

   // save pointer to GPU descriptor
   uciOnPuschSegLLRs1KernelArgs_t& kernelArgs = m_kernelArgs;
   kernelArgs.pDynDescr = reinterpret_cast<uciOnPuschSegLLRs1DynDescr_t*>(pGpuDynDesc);

   // Optional descriptor copy to GPU memory
   if(enableCpuToGpuDescrAsyncCpy)
   {
      cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(uciOnPuschSegLLRs1DynDescr_t), cudaMemcpyHostToDevice, strm);
   }

   // select kernel (includes launch geometry). Populate launchCfg.
   kernelSelect(nUciUes, pUciUserIdxs, pTbPrmsCpu, pNumPrbs, nPuschDataSym, pLaunchCfg);
   pLaunchCfg->kernelArgs[0] = &m_kernelArgs.pDynDescr;
   pLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);

   for (int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx) {
      auto& harqRvdStride = pCpuDynDesc->harqRvdStride[ueGrpIdx];
      auto& harqUciStride = pCpuDynDesc->harqUciStride[ueGrpIdx];
      auto& csi1RvdStride = pCpuDynDesc->csi1RvdStride[ueGrpIdx];
      auto& harqAckStride = pCpuDynDesc->harqAckStride[ueGrpIdx];
      auto& harqRvd_CsiLcms = pCpuDynDesc->harqRvd_CsiLcms[ueGrpIdx];
      // Empty stride maps before building the dataset
      harqRvdStride = {};
      harqUciStride = {};
      csi1RvdStride = {};
      harqAckStride = {};

      harqRvd_CsiLcms = {};
      // Calculate strides and any intersections
      buildCodewordMap(
         nUciUes, pUciUserIdxs, pTbPrmsCpu[ueGrpIdx], pNumPrbs[ueGrpIdx], pCpuDynDesc,
         harqRvdStride, harqUciStride, csi1RvdStride, harqAckStride, harqRvd_CsiLcms);
   }
}

void uciOnPuschSegLLRs1::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
   dynDescrSizeBytes  = sizeof(uciOnPuschSegLLRs1DynDescr_t);
   dynDescrAlignBytes = alignof(uciOnPuschSegLLRs1DynDescr_t);
}


