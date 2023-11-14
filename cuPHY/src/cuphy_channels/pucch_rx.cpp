/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cstddef>
#include <string>
#include "pucch_rx.hpp"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "util.hpp"
#include "convert_tensor.cuh"

#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "cuphy_internal.h"

//#define MEMTRACE      //FixMe uncomment to enable memtrace  in cuPHY runs, but note that call to memtrace_set_config(0);
//                      //will disable mem tracing on that thread onward

PucchRx::PucchRx(cuphyPucchStatPrms_t const* pStatPrms, cudaStream_t strm) :
    m_LinearAlloc(getBufferSize(pStatPrms), &m_memoryFootprint),
    m_kernelStatDescr("PucchStatDescr"),
    m_kernelDynDescr("PucchDynDescr"),
    m_cuphyPucchStatPrms(*(pStatPrms)),
    m_polSegPrmsBufCpu(CUPHY_MAX_N_POL_CWS),
    m_polCwPrmsBufCpu(CUPHY_MAX_N_POL_CWS),
    m_polDcdrListSz(pStatPrms->polarDcdrListSz),
    m_rmCwPrmsBufCpu(CUPHY_PUCCH_F3_MAX_UCI * pStatPrms->nMaxCellsPerSlot),
    m_tPrmDataRxBufCpu(pStatPrms->nMaxCellsPerSlot),
    m_tRefDataRxBufCpu(pStatPrms->nMaxCellsPerSlot),
    m_cuStream(strm)
 {
    pStatPrms->pOutInfo->pMemoryFootprint = &m_memoryFootprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel

    // Allocate descriptors for pipeline usage
    allocateDescr();

    // create componets
    createComponents();

    // Resize vectors
    m_polCwTreeTypesAddrVec.resize(CUPHY_MAX_N_POL_UCI_SEGS);
    m_polSegLLRsAddrVec.resize(CUPHY_MAX_N_POL_UCI_SEGS);
    m_polCwLLRsAddrVec.resize(CUPHY_MAX_N_POL_CWS);
    m_polCbEstAddrVec.resize(CUPHY_MAX_N_POL_CWS);
    if (m_polDcdrListSz > 1) {
        m_listPolScratchAddrVec.resize(CUPHY_MAX_N_POL_CWS);
    }
    m_polCwTreeLLRsAddrVec.resize(CUPHY_MAX_N_POL_CWS);
    m_pUciSegEst.resize(CUPHY_MAX_N_POL_UCI_SEGS * pStatPrms->nMaxCellsPerSlot);
    // CUPHY_PUCCH_F2_MAX_UCI and CUPHY_PUCCH_F3_MAX_UCI are defined considering multi-cells per group
    m_F2RmSizesVec.resize(CUPHY_PUCCH_F2_MAX_UCI);
    m_F3RmSizesVec.resize(CUPHY_PUCCH_F3_MAX_UCI);
    m_F3seg1LLRaddrsVec.resize(CUPHY_PUCCH_F3_MAX_UCI);
    m_F2seg1LLRaddrsVec.resize(CUPHY_PUCCH_F2_MAX_UCI);
    m_F2nBitsUciSeg1.resize(CUPHY_PUCCH_F2_MAX_UCI);
    m_F3nBitsUciSeg1.resize(CUPHY_PUCCH_F3_MAX_UCI);
    m_LinearAlloc.memset(0,strm);
    // set debug prms
    m_outputPrms.debugOutputFlag = false;
    if(nullptr != pStatPrms->pDbg)
    {
        if(pStatPrms->pDbg->pOutFileName != nullptr)
        {
            m_outputPrms.debugOutputFlag = true;
            m_outputPrms.outHdf5File     = hdf5hpp::hdf5_file::open(pStatPrms->pDbg->pOutFileName);
        }else{
            m_outputPrms.debugOutputFlag = false;
        }
    }

    createGraph();
#if CUDA_VERSION >= 12000
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0));
#else            
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0, 0, 0));
#endif

    if(PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1)
    {
        m_memoryFootprint.printMemoryFootprint(this, "PUCCH");
    }
 }

  void PucchRx::expandUciCodingPrms(cuphyPucchUciPrm_t& uciPrms, uint32_t nInfoBits, uint32_t nRmBits)
 {
     //if(nInfoBits < 12) 
     if(nInfoBits <= CUPHY_N_MAX_UCI_BITS_RM)
     {
         m_rmCwPrmsBufCpu[m_nRmCws].K         = nInfoBits;
         m_rmCwPrmsBufCpu[m_nRmCws].E         = nRmBits;
         m_rmCwPrmsBufCpu[m_nRmCws].exitFlag  = 0;
        m_outputPrms.nUciPayloadBytes        += sizeof(uint32_t);
        m_nRmCws                             += 1;
     } else {
        cuphyPolarUciSegPrm_t& polSegPrms = m_polSegPrmsBufCpu[m_nPolSegs];

        // crc size (38.212 6.3.1.2.1)
        polSegPrms.nCrcBits = (nInfoBits <= 19) ? 6 : 11;
        polSegPrms.exitFlag = 0;

        // code block segmentation (38.212 6.3.1.3.1)
        // code block size         (38.212 5.2.1)
        if(((nInfoBits >= 360) && (nRmBits >= 1088)) || (nInfoBits >= 1013))
        {
            polSegPrms.nCbs           = 2;
            polSegPrms.K_cw           = div_round_up(nInfoBits, static_cast<uint32_t>(2)) + polSegPrms.nCrcBits;
            polSegPrms.E_cw           = nRmBits / 2;
            polSegPrms.zeroInsertFlag = nInfoBits % 2;
        }else
        {
            polSegPrms.nCbs           = 1;
            polSegPrms.K_cw           = nInfoBits + polSegPrms.nCrcBits;
            polSegPrms.E_cw           = nRmBits;
            polSegPrms.zeroInsertFlag = 0;
        }

        // encoded cb(s) size (38.212 5.3.1)
        uint32_t n_temp        = static_cast<uint32_t>(ceil(log2(static_cast<double>(polSegPrms.E_cw))) - 1);
        uint32_t two_to_n_temp = 1 << n_temp;

        uint32_t n_1;
        if((8*polSegPrms.E_cw <= 9*two_to_n_temp) && (16*polSegPrms.K_cw <= 9*polSegPrms.E_cw))
        {
            n_1 = n_temp;
        }else
        {
            n_1 = n_temp + 1;
        }

        uint32_t n_2   = static_cast<uint32_t>(ceil(log2(static_cast<double>(polSegPrms.K_cw) * 8)));
        uint32_t n_min = 5;
        uint32_t n_max = 10;

        polSegPrms.n_cw = std::max(std::min(std::min(n_1, n_2), n_max), n_min);
        polSegPrms.N_cw = 1 << polSegPrms.n_cw;

        // child cb(s)
        for(int i = 0; i < polSegPrms.nCbs; ++i)
        {
            cuphyPolarCwPrm_t& polCwPrms = m_polCwPrmsBufCpu[m_nPolCbs];

            polCwPrms.N_cw     =  polSegPrms.N_cw; 
            polCwPrms.nCrcBits =  polSegPrms.nCrcBits;
            polCwPrms.A_cw     =  polSegPrms.K_cw - polSegPrms.nCrcBits;

            polCwPrms.nCbsInUciSeg      = polSegPrms.nCbs;
            polCwPrms.cbIdxWithinUciSeg = i;
            polCwPrms.zeroInsertFlag    = polSegPrms.zeroInsertFlag;
            
            polSegPrms.childCbIdxs[i]  =  m_nPolCbs;
            m_nPolCbs                    +=  1;
        }
        m_outputPrms.nUciPayloadBytes += 4*polSegPrms.nCbs*div_round_up(m_polCwPrmsBufCpu[m_nPolCbs - 1].A_cw, static_cast<uint16_t>(32)); // 4 bytes per word
        m_nPolSegs += 1;        
     }
  
    // Append HARQ, SR, CSI part 1 uint32 words
     if (uciPrms.bitLenHarq > 0) {
         m_outputPrms.nUciPayloadBytes += 4*div_round_up(uciPrms.bitLenHarq, static_cast<uint16_t>(32)); // 4 bytes per word
     }
     if (uciPrms.bitLenSr > 0) {
         m_outputPrms.nUciPayloadBytes += 4; // According to SCF FAPIv10, bitLenSr <= 4
     }
     if (uciPrms.bitLenCsiPart1 > 0) {
         m_outputPrms.nUciPayloadBytes += 4*div_round_up(uciPrms.bitLenCsiPart1, static_cast<uint16_t>(32)); // 4 bytes per word
     }
 }

 void PucchRx::expandF234UciParameters(cuphyPucchUciPrm_t& uciPrms, F234RmSizes_t& rmSizes, uint16_t nBitsUciSeg1)
 {
    if(nBitsUciSeg1 > 0)
    {
        //if(nBitsUciSeg1 < 12)
        if(nBitsUciSeg1 <= CUPHY_N_MAX_UCI_BITS_RM)
        {
            m_rmCwPrmsBufCpu[m_nRmCws].DTXthreshold      = uciPrms.DTXthreshold;
            m_rmCwPrmsBufCpu[m_nRmCws].en_DTXest         = 0; //disable DTXest in rm_decoder.cu
        }
        expandUciCodingPrms(uciPrms, nBitsUciSeg1, rmSizes.E_seg1);
        m_outputPrms.nUciSegs += 1;
    }    
 }


 void PucchRx::allocateBackendBuffers(cuphyPucchUciPrm_t&         uciPrms, 
                                      cuphyPucchF234OutOffsets_t& outOffsets, 
                                      F234RmSizes_t&              rmSizes,
                                      void*                       pSeg1LLRs,
                                      uint16_t                    nBitsUciSeg1,
                                      uint16_t&                   F234uciIdx,
                                      uint16_t&                   rmCwIdx,
                                      uint16_t&                   polSegIdx,
                                      uint32_t&                   uciWordOffset)
 {
    void*    pUciPayloadsGpuVoid = static_cast<void*>(m_outputPrms.pUciPayloadsGpu);
    
    outOffsets.HarqDetectionStatusOffset  = F234uciIdx;
    outOffsets.CsiP1DetectionStatusOffset = F234uciIdx;
    outOffsets.CsiP2DetectionStatusOffset = F234uciIdx;

    if (nBitsUciSeg1 > 0)
    {
        //if (nBitsUciSeg1 < 12)
        if(nBitsUciSeg1 <= CUPHY_N_MAX_UCI_BITS_RM)
        {
            outOffsets.uciSeg1PayloadByteOffset = uciWordOffset*sizeof(uint32_t);
            m_rmCwPrmsBufCpu[rmCwIdx].d_LLRs       = static_cast<__half*>(pSeg1LLRs);
            m_rmCwPrmsBufCpu[rmCwIdx].d_cbEst      = static_cast<uint32_t*>(pUciPayloadsGpuVoid) + uciWordOffset;

            uciWordOffset += 1;
            rmCwIdx       += 1;
        } else {
            cuphyPolarUciSegPrm_t& polSegPrms    = m_polSegPrmsBufCpu[polSegIdx];
            uint16_t N_cw                        =  polSegPrms.N_cw;
            uint16_t nBytes_cwTree               =  2 * N_cw;

            m_polCwTreeTypesAddrVec[polSegIdx] =  static_cast<uint8_t*>(m_LinearAlloc.alloc(nBytes_cwTree));
            m_polSegLLRsAddrVec[polSegIdx]     =  static_cast<__half*>(pSeg1LLRs);
            polSegPrms.pUciSegLLRs             =  static_cast<__half*>(pSeg1LLRs);

            uint16_t nDecodedCbWords = div_round_up(static_cast<uint16_t>(polSegPrms.K_cw - polSegPrms.nCrcBits), static_cast<uint16_t>(32));
            uint16_t nUciSegBits     = polSegPrms.nCbs * (polSegPrms.K_cw - polSegPrms.nCrcBits) - polSegPrms.zeroInsertFlag;
            uint16_t nUciSegWords    = div_round_up(nUciSegBits, static_cast<uint16_t>(32));

            outOffsets.uciSeg1PayloadByteOffset = uciWordOffset*sizeof(uint32_t);
            m_pUciSegEst[polSegIdx]             = static_cast<uint32_t*>(pUciPayloadsGpuVoid) + uciWordOffset;
            uciWordOffset                      += nUciSegWords;

            for (int i = 0; i < polSegPrms.nCbs; ++i)
            {
                uint8_t cwIdx = polSegPrms.childCbIdxs[i];

                m_polCwTreeLLRsAddrVec[cwIdx]          = static_cast<__half*>(m_LinearAlloc.alloc(sizeof(__half) * (2 * N_cw) * m_polDcdrListSz));
                m_polCwLLRsAddrVec[cwIdx]              = m_polCwTreeLLRsAddrVec[cwIdx] + N_cw;
                if (m_polDcdrListSz > 1) {
                    m_listPolScratchAddrVec[cwIdx]     = static_cast<bool*>(m_LinearAlloc.alloc(sizeof(bool) * (2 * N_cw * m_polDcdrListSz)));
                }
                m_polCwPrmsBufCpu[cwIdx].pCwTreeTypes  = m_polCwTreeTypesAddrVec[polSegIdx];
                m_polCwPrmsBufCpu[cwIdx].pCrcStatus   = m_outputPrms.pHarqDetectionStatusGpu + F234uciIdx;
                m_polCwPrmsBufCpu[cwIdx].pCrcStatus1  = m_outputPrms.pCsiP1DetectionStatusGpu + F234uciIdx;
                m_polCwPrmsBufCpu[cwIdx].en_CrcStatus = CUPHY_DET_EN + CUPHY_PUCCH_DET_EN;
                m_polCwPrmsBufCpu[cwIdx].exitFlag     = 0;
                if(polSegPrms.nCbs == 1)
                {
                    m_polCbEstAddrVec[cwIdx] = m_pUciSegEst[polSegIdx];
                }else
                {
                    m_polCbEstAddrVec[cwIdx] = static_cast<uint32_t*>(m_LinearAlloc.alloc(nDecodedCbWords));
                }
                m_polCwPrmsBufCpu[cwIdx].pUciSegEst = m_pUciSegEst[polSegIdx];
                m_polCwPrmsBufCpu[cwIdx].pCbEst     = m_polCbEstAddrVec[cwIdx];
                m_polCwPrmsBufCpu[cwIdx].pCwLLRs    = m_polCwLLRsAddrVec[cwIdx];
            }

        
            outOffsets.uciSeg1CrcFlagOffset     = polSegIdx;
            outOffsets.srCrcFlagOffset          = outOffsets.uciSeg1CrcFlagOffset;
            outOffsets.harqCrcFlagOffset        = outOffsets.uciSeg1CrcFlagOffset;
            outOffsets.csi1CrcFlagOffset        = outOffsets.uciSeg1CrcFlagOffset;

            polSegIdx                         += 1;
        }

        if (uciPrms.bitLenHarq > 0) {
            outOffsets.harqPayloadByteOffset = uciWordOffset*sizeof(uint32_t);
            uciWordOffset += div_round_up(uciPrms.bitLenHarq, static_cast<uint16_t>(32));
        } else {
            outOffsets.harqPayloadByteOffset = 0;
        }
        if (uciPrms.bitLenSr > 0) {
            outOffsets.srPayloadByteOffset = uciWordOffset*sizeof(uint32_t);
            uciWordOffset++; // According to SCF FAPIv10, bitLenSr <= 4
        } else {
            outOffsets.srPayloadByteOffset = 0;
        }
        if (uciPrms.bitLenCsiPart1 > 0) {
            outOffsets.csi1PayloadByteOffset = uciWordOffset*sizeof(uint32_t);
            uciWordOffset += div_round_up(uciPrms.bitLenCsiPart1, static_cast<uint16_t>(32));
        } else {
            outOffsets.csi1PayloadByteOffset = 0;
        }
   }
 }

 void PucchRx::allocateDeviceMemory()
 {
    // F0 and F1 output buffers
    m_outputPrms.pF0UciOutGpu = static_cast<cuphyPucchF0F1UciOut_t*>(m_LinearAlloc.alloc(m_nF0Ucis * sizeof(cuphyPucchF0F1UciOut_t)));
    m_outputPrms.pF1UciOutGpu = static_cast<cuphyPucchF0F1UciOut_t*>(m_LinearAlloc.alloc(m_nF1Ucis * sizeof(cuphyPucchF0F1UciOut_t)));

    // F2, F3, F4 output buffers
    m_outputPrms.pCrcFlagsGpu             = static_cast<uint8_t*>(m_LinearAlloc.alloc(m_outputPrms.nUciSegs));
    m_pPolCrcFlags                        = m_outputPrms.pCrcFlagsGpu;

    if(0<m_outputPrms.nUciPayloadBytes)
    {
        m_tUciPayload.desc().set(CUPHY_R_8U, m_outputPrms.nUciPayloadBytes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tUciPayload);
        m_outputPrms.pUciPayloadsGpu          = static_cast<uint8_t*>(m_tUciPayload.addr());
    }

    if(0<(m_nF3Ucis + m_nF2Ucis))
    {
        m_tDtxFlags.desc().set(CUPHY_R_8U, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tDtxFlags);
        m_outputPrms.pDtxFlagsGpu             = static_cast<uint8_t*>(m_tDtxFlags.addr());
        m_pF2dtxFlags                         = m_outputPrms.pDtxFlagsGpu;
        m_pF3dtxFlags                         = m_outputPrms.pDtxFlagsGpu + m_nF2Ucis;

        m_tSinr.desc().set(CUPHY_R_32F, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tSinr);
        m_outputPrms.pSinrGpu                 = static_cast<float*>(m_tSinr.addr());
        m_pF2pSinr                            = m_outputPrms.pSinrGpu;
        m_pF3pSinr                            = m_outputPrms.pSinrGpu + m_nF2Ucis;

        m_tRssi.desc().set(CUPHY_R_32F, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRssi);
        m_outputPrms.pRssiGpu                 = static_cast<float*>(m_tRssi.addr());
        m_pF2pRssi                            = m_outputPrms.pRssiGpu;
        m_pF3pRssi                            = m_outputPrms.pRssiGpu + m_nF2Ucis;

        m_tRsrp.desc().set(CUPHY_R_32F, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRsrp);
        m_outputPrms.pRsrpGpu                 = static_cast<float*>(m_tRsrp.addr());
        m_pF2pRsrp                            = m_outputPrms.pRsrpGpu;
        m_pF3pRsrp                            = m_outputPrms.pRsrpGpu + m_nF2Ucis;
            
        m_tInterf.desc().set(CUPHY_R_32F, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tInterf);
        m_outputPrms.pInterfGpu               = static_cast<float*>(m_tInterf.addr());
        m_pF2pInterf                          = m_outputPrms.pInterfGpu;
        m_pF3pInterf                          = m_outputPrms.pInterfGpu + m_nF2Ucis;

        m_tNoiseVar.desc().set(CUPHY_R_32F, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tNoiseVar);
        m_outputPrms.pNoiseVarGpu             = static_cast<float*>(m_tNoiseVar.addr());
        m_pF2pNoiseVar                        = m_outputPrms.pNoiseVarGpu;
        m_pF3pNoiseVar                        = m_outputPrms.pNoiseVarGpu + m_nF2Ucis;

        m_tTaEst.desc().set(CUPHY_R_32F, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tTaEst);
        m_outputPrms.pTaEstGpu                = static_cast<float*>(m_tTaEst.addr());
        m_pF2pTaEst                           = m_outputPrms.pTaEstGpu;
        m_pF3pTaEst                           = m_outputPrms.pTaEstGpu + m_nF2Ucis;

        m_tHarqDetectionStatus.desc().set(CUPHY_R_8U, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tHarqDetectionStatus);
        m_outputPrms.pHarqDetectionStatusGpu       = static_cast<uint8_t*>(m_tHarqDetectionStatus.addr());

        m_tCsiP1DetectionStatus.desc().set(CUPHY_R_8U, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tCsiP1DetectionStatus);
        m_outputPrms.pCsiP1DetectionStatusGpu      = static_cast<uint8_t*>(m_tCsiP1DetectionStatus.addr());

        m_tCsiP2DetectionStatus.desc().set(CUPHY_R_8U, (m_nF3Ucis + m_nF2Ucis), cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tCsiP2DetectionStatus);
        m_outputPrms.pCsiP2DetectionStatusGpu      = static_cast<uint8_t*>(m_tCsiP2DetectionStatus.addr());
    }

    // LLR buffers (equalizer output, and decoder inputs)
    uint16_t rmCwIdx          = 0;
    uint16_t polSegIdx        = 0;
    uint32_t uciWordOffset    = 0;
    //uint32_t uciDtxNvOffset   = 0;
    uint16_t F234uciIdx       = 0;

    for(int F2uciIdx = 0; F2uciIdx < m_nF2Ucis; ++F2uciIdx)
    {
        void* pSeg1LLRs               = m_LinearAlloc.alloc(m_F2RmSizesVec[F2uciIdx].E_seg1 * sizeof(__half));
        m_F2seg1LLRaddrsVec[F2uciIdx] = static_cast<__half*>(pSeg1LLRs);
        //m_outputPrms.pPucchF2OutOffsetsCpu[F2uciIdx].dtxFlagOffset = F234uciIdx;
        m_outputPrms.pPucchF2OutOffsetsCpu[F2uciIdx].snrOffset     = F234uciIdx;
        m_outputPrms.pPucchF2OutOffsetsCpu[F2uciIdx].RSSIoffset    = F234uciIdx;
        m_outputPrms.pPucchF2OutOffsetsCpu[F2uciIdx].RSRPoffset    = F234uciIdx;
        m_outputPrms.pPucchF2OutOffsetsCpu[F2uciIdx].InterfOffset  = F234uciIdx;
        m_outputPrms.pPucchF2OutOffsetsCpu[F2uciIdx].taEstOffset   = F234uciIdx;

        if(m_F2nBitsUciSeg1[F2uciIdx] > 0)
        {
            //if(m_F2nBitsUciSeg1[F2uciIdx] < 12)
            if(m_F2nBitsUciSeg1[F2uciIdx] <= CUPHY_N_MAX_UCI_BITS_RM)
            {
                /////////////////////connect DTX output to d_DTXEst//////////////////////
//                void*    pDtxF2RMGpuVoid  = static_cast<void*>(m_outputPrms.pDtxFlagsGpu);
//                m_rmCwPrmsBufCpu[rmCwIdx].d_DTXEst      = static_cast<uint8_t*>(pDtxF2RMGpuVoid) + F234uciIdx;

                m_rmCwPrmsBufCpu[rmCwIdx].d_DTXStatus      = m_outputPrms.pHarqDetectionStatusGpu + F234uciIdx;
                m_rmCwPrmsBufCpu[rmCwIdx].d_DTXStatus1     = m_outputPrms.pCsiP1DetectionStatusGpu + F234uciIdx;
                m_rmCwPrmsBufCpu[rmCwIdx].d_DTXStatus2     = m_outputPrms.pCsiP2DetectionStatusGpu + F234uciIdx;

                void*    pNoiseVarF2GpuVoid  = static_cast<void*>(m_outputPrms.pNoiseVarGpu);
                m_rmCwPrmsBufCpu[rmCwIdx].d_noiseVar      = static_cast<float*>(pNoiseVarF2GpuVoid) + F234uciIdx;
                m_rmCwPrmsBufCpu[rmCwIdx].Qm              = 0; //not valid for PUCCH F2;
                /////////////////////enable DTXest in rm_decoder.cu//////////////////////
                m_rmCwPrmsBufCpu[rmCwIdx].en_DTXest = CUPHY_DTX_EN + CUPHY_DET_EN + CUPHY_PUCCH_DET_EN;

                //uciDtxNvOffset++;
            }
        }
        allocateBackendBuffers(m_pF2UciPrms[F2uciIdx], m_outputPrms.pPucchF2OutOffsetsCpu[F2uciIdx], m_F2RmSizesVec[F2uciIdx], pSeg1LLRs, m_F2nBitsUciSeg1[F2uciIdx], F234uciIdx, rmCwIdx, polSegIdx, uciWordOffset);
        F234uciIdx += 1;
    }

    for(int F3uciIdx = 0; F3uciIdx < m_nF3Ucis; ++F3uciIdx)
    {
        void* pSeg1LLRs               = m_LinearAlloc.alloc(m_F3RmSizesVec[F3uciIdx].E_seg1 * sizeof(__half));
        m_F3seg1LLRaddrsVec[F3uciIdx] = static_cast<__half*>(pSeg1LLRs);
        //m_outputPrms.pPucchF3OutOffsetsCpu[F3uciIdx].dtxFlagOffset = F234uciIdx;
        m_outputPrms.pPucchF3OutOffsetsCpu[F3uciIdx].snrOffset     = F234uciIdx;
        m_outputPrms.pPucchF3OutOffsetsCpu[F3uciIdx].RSSIoffset    = F234uciIdx;
        m_outputPrms.pPucchF3OutOffsetsCpu[F3uciIdx].RSRPoffset    = F234uciIdx;
        m_outputPrms.pPucchF3OutOffsetsCpu[F3uciIdx].InterfOffset  = F234uciIdx;
        m_outputPrms.pPucchF3OutOffsetsCpu[F3uciIdx].taEstOffset   = F234uciIdx;

        if(m_F3nBitsUciSeg1[F3uciIdx] > 0)
        {
            //if(m_F3nBitsUciSeg1[F3uciIdx] < 12)
            if(m_F3nBitsUciSeg1[F3uciIdx] <= CUPHY_N_MAX_UCI_BITS_RM)
            {
                /////////////////////connect DTX output to d_DTXEst//////////////////////
//                void*    pDtxF3RMGpuVoid  = static_cast<void*>(m_outputPrms.pDtxFlagsGpu);
//                m_rmCwPrmsBufCpu[rmCwIdx].d_DTXEst      = static_cast<uint8_t*>(pDtxF3RMGpuVoid) + F234uciIdx;

                m_rmCwPrmsBufCpu[rmCwIdx].d_DTXStatus      = m_outputPrms.pHarqDetectionStatusGpu + F234uciIdx;
                m_rmCwPrmsBufCpu[rmCwIdx].d_DTXStatus1     = m_outputPrms.pCsiP1DetectionStatusGpu + F234uciIdx;
                m_rmCwPrmsBufCpu[rmCwIdx].d_DTXStatus2     = m_outputPrms.pCsiP2DetectionStatusGpu + F234uciIdx;

                void*    pNoiseVarF3GpuVoid  = static_cast<void*>(m_outputPrms.pNoiseVarGpu);
                m_rmCwPrmsBufCpu[rmCwIdx].d_noiseVar      = static_cast<float*>(pNoiseVarF3GpuVoid) + F234uciIdx;
                m_rmCwPrmsBufCpu[rmCwIdx].Qm              = 0; //not valid for PUCCH F3;
                /////////////////////enable DTXest in rm_decoder.cu//////////////////////
                m_rmCwPrmsBufCpu[rmCwIdx].en_DTXest = CUPHY_DTX_EN + CUPHY_DET_EN + CUPHY_PUCCH_DET_EN;
            }
        }

        allocateBackendBuffers(m_pF3UciPrms[F3uciIdx], m_outputPrms.pPucchF3OutOffsetsCpu[F3uciIdx], m_F3RmSizesVec[F3uciIdx], pSeg1LLRs, m_F3nBitsUciSeg1[F3uciIdx], F234uciIdx, rmCwIdx, polSegIdx, uciWordOffset);
        F234uciIdx += 1;
    }

    if(m_nRmCws > 0)
     {
         m_pRmCwPrmsGpu = static_cast<cuphyRmCwPrm_t*>(m_LinearAlloc.alloc(m_nRmCws * sizeof(cuphyRmCwPrm_t)));
     }

    if(m_nPolSegs > 0)
    {
        m_pPolSegPrmsGpu = static_cast<cuphyPolarUciSegPrm_t*>(m_LinearAlloc.alloc(m_nPolSegs * sizeof(cuphyPolarUciSegPrm_t)));
        m_pPolCwPrmsGpu  = static_cast<cuphyPolarCwPrm_t*>(m_LinearAlloc.alloc(m_nPolCbs * sizeof(cuphyPolarCwPrm_t)));
    }
 }


 void PucchRx::createGraph()
 {
#if CUDART_VERSION < 11000
    throw cuphy::cuda_driver_exception("Graph mode requires CUDA driver kernel node params which requires CUDA 11.0 or higher");
#endif

     CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));

     void* arg;
     void* kernelParams[1] = {&arg};
     // Initialize empty nodes with 0 and 1 input pointer args
     CUPHY_CHECK(cuphySetEmptyKernelNodeParams(&m_emptyNode0paramDriver));
     CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode1paramDriver, 1, &(kernelParams[0])));

     // Use empty node as a root
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_emptyRootNode, m_graph, nullptr, 0, &m_emptyNode0paramDriver));

     // add node(s), initially start with some kernel parameters, at setup, do the updating
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_pucchF0RxKernelNode, m_graph, &m_emptyRootNode, 1, &m_emptyNode1paramDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_pucchF1RxKernelNode, m_graph, &m_pucchF0RxKernelNode, 1, &m_emptyNode1paramDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_pucchF2RxKernelNode, m_graph, &m_pucchF1RxKernelNode, 1, &m_emptyNode1paramDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_pucchF3RxKernelNode, m_graph, &m_pucchF2RxKernelNode, 1, &m_emptyNode1paramDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_rmDecoderKernelNode, m_graph, &m_pucchF3RxKernelNode, 1, &m_emptyNode1paramDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_compCwTreeTypesKernelNode, m_graph, &m_rmDecoderKernelNode, 1, &m_emptyNode1paramDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_polSegDeRmDeItlKernelNode, m_graph, &m_compCwTreeTypesKernelNode, 1, &m_emptyNode1paramDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_polarDecoderKernelNode, m_graph, &m_polSegDeRmDeItlKernelNode, 1, &m_emptyNode1paramDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_pucchF234RxKernelNode, m_graph, &m_polarDecoderKernelNode, 1, &m_emptyNode1paramDriver));
 }

 void PucchRx::updateGraph()
 {
#ifdef MEMTRACE
     memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE|MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif

#if CUDART_VERSION < 11000
     throw cuphy::cuda_driver_exception("Graph mode requires CUDA driver kernel node params which requires CUDA 11.0 or higher");
#endif

     if(m_nF0Ucis > 0)
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF0RxKernelNode, 1));
#endif
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF0RxKernelNode, &(m_pucchF0RxLaunchCfg.kernelNodeParamsDriver)));
     }
     else
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF0RxKernelNode, 0));
#else
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF0RxKernelNode, &m_emptyNode1paramDriver));
#endif
     }

     if(m_nF1Ucis > 0)
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF1RxKernelNode, 1));
#endif
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF1RxKernelNode, &(m_pucchF1RxLaunchCfg.kernelNodeParamsDriver)));
     }
     else
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF1RxKernelNode, 0));
#else
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF1RxKernelNode, &m_emptyNode1paramDriver));
#endif
     }

     if(m_nF2Ucis > 0)
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF2RxKernelNode, 1));
#endif
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF2RxKernelNode, &(m_pucchF2RxLaunchCfg.kernelNodeParamsDriver)));
     }
     else
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF2RxKernelNode, 0));
#else
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF2RxKernelNode, &m_emptyNode1paramDriver));
#endif
     }

     if(m_nF3Ucis > 0)
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF3RxKernelNode, 1));
#endif
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF3RxKernelNode, &(m_pucchF3RxLaunchCfg.kernelNodeParamsDriver)));
     }
     else
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF3RxKernelNode, 0));
#else
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF3RxKernelNode, &m_emptyNode1paramDriver));
#endif
     }

     if ((m_nF2Ucis > 0) || (m_nF3Ucis > 0)) {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF234RxKernelNode, 1));
#endif
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF234RxKernelNode, &(m_pucchF234UciSegLaunchCfg.kernelNodeParamsDriver)));
     }
     else
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_pucchF234RxKernelNode, 0));
#else
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_pucchF234RxKernelNode, &m_emptyNode1paramDriver));
#endif
     }

     if(m_nRmCws > 0)
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rmDecoderKernelNode, 1));
#endif
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rmDecoderKernelNode, &(m_rmDecoderLaunchCfg.kernelNodeParamsDriver)));
     }
     else
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rmDecoderKernelNode, 0));
#else
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rmDecoderKernelNode, &m_emptyNode1paramDriver));
#endif
     }

     if(m_nPolSegs > 0)
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_compCwTreeTypesKernelNode, 1));
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_polSegDeRmDeItlKernelNode, 1));
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_polarDecoderKernelNode, 1));
#endif
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_compCwTreeTypesKernelNode, &(m_compCwTreeTypesLaunchCfg.kernelNodeParamsDriver)));
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_polSegDeRmDeItlKernelNode, &(m_polSegDeRmDeItlLaunchCfg.kernelNodeParamsDriver)));
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_polarDecoderKernelNode, &(m_polarDecoderLaunchCfg.kernelNodeParamsDriver)));
     }
     else
     {
#if CUDART_VERSION >= 11060
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_compCwTreeTypesKernelNode, 0));
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_polSegDeRmDeItlKernelNode, 0));
         CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_polarDecoderKernelNode, 0));
#else
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_compCwTreeTypesKernelNode, &m_emptyNode1paramDriver));
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_polSegDeRmDeItlKernelNode, &m_emptyNode1paramDriver));
         CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_polarDecoderKernelNode, &m_emptyNode1paramDriver));
#endif
     }

#ifdef MEMTRACE
     memtrace_set_config(0);
#endif
 }


 cuphyStatus_t PucchRx::setupCmn(cuphyPucchDynPrms_t *pDynPrm)
 {
    cuphyStatus_t ret = CUPHY_STATUS_SUCCESS;
    // reset linear allocation
    m_LinearAlloc.reset();

    // extract stream
    m_cuStream = pDynPrm->cuStream;

    // copy dynamic parameters
    m_cuphyPucchCellGrpDynPrm = *(pDynPrm->pCellGrpDynPrm);     //ToDo?? this is a shallow copy

    // copy common cell parameters
    auto     dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    m_pCellCmnPrms                 = reinterpret_cast<cuphyPucchCellPrm_t*>(dynCpuDescrStartAddrs[PUCCH_CELL_INFO]);
    for(int i = 0; i < m_cuphyPucchStatPrms.nMaxCellsPerSlot; i++)
    {
        m_pCellCmnPrms[i].nRxAnt         = m_pCellStatPrms[i].nRxAnt;
        m_pCellCmnPrms[i].mu             = m_pCellStatPrms[i].mu;
        m_pCellCmnPrms[i].pucchHoppingId = m_pCellDynPrms[i].pucchHoppingId;
        m_pCellCmnPrms[i].slotNum        = m_pCellDynPrms[i].slotNum;
    }

    // Expand F234 uci parameters
    m_outputPrms.nUciSegs         = 0;
    m_outputPrms.nUciPayloadBytes = 0;
    m_nRmCws                      = 0;
    m_nPolSegs                    = 0;
    m_nPolCbs                     = 0;

    for(int uciIdx = 0; uciIdx < m_nF2Ucis; ++uciIdx)
    {
        m_F2nBitsUciSeg1[uciIdx]    = m_pF2UciPrms[uciIdx].bitLenHarq + m_pF2UciPrms[uciIdx].bitLenCsiPart1 + m_pF2UciPrms[uciIdx].bitLenSr;
        m_F2RmSizesVec[uciIdx]      = compRateMatchSizesF2(m_pF2UciPrms[uciIdx]);
        expandF234UciParameters(m_pF2UciPrms[uciIdx], m_F2RmSizesVec[uciIdx], m_F2nBitsUciSeg1[uciIdx]);
    }

    for(int uciIdx = 0; uciIdx < m_nF3Ucis; ++uciIdx)
    {
        m_F3nBitsUciSeg1[uciIdx]            = m_pF3UciPrms[uciIdx].bitLenHarq + m_pF3UciPrms[uciIdx].bitLenCsiPart1 + m_pF3UciPrms[uciIdx].bitLenSr;
        m_F3RmSizesVec[uciIdx]              = compRateMatchSizesF3(m_pF3UciPrms[uciIdx]);
        expandF234UciParameters(m_pF3UciPrms[uciIdx], m_F3RmSizesVec[uciIdx], m_F3nBitsUciSeg1[uciIdx]);
    }

    // Input tensors
    for(uint16_t i = 0; i < m_cuphyPucchStatPrms.nMaxCellsPerSlot; i++)
    {
        m_tPrmDataRxBufCpu[i] = pDynPrm->pDataIn->pTDataRx[i];
    }

    // CPU output pointers
    m_outputPrms.pUciPayloadsCpu          = pDynPrm->pDataOut->pUciPayloads;
    m_outputPrms.pCrcFlagsCpu             = pDynPrm->pDataOut->pCrcFlags;
    //m_outputPrms.pDtxFlagsCpu             = pDynPrm->pDataOut->pDtxFlags;
    m_outputPrms.pRssiCpu                 = pDynPrm->pDataOut->pRssi;
    m_outputPrms.pRsrpCpu                 = pDynPrm->pDataOut->pRsrp;
    m_outputPrms.pSinrCpu                 = pDynPrm->pDataOut->pSinr;
    m_outputPrms.pInterfCpu               = pDynPrm->pDataOut->pInterf;
    m_outputPrms.pTaEstCpu                = pDynPrm->pDataOut->pTaEst;
    //m_outputPrms.pNumCsi2BitsCpu          = pDynPrm->pDataOut->pNumCsi2Bits;
    m_outputPrms.cpuCopyOn                = pDynPrm->cpuCopyOn;
    m_outputPrms.pF0UciOutCpu             = pDynPrm->pDataOut->pF0UcisOut;
    m_outputPrms.pF1UciOutCpu             = pDynPrm->pDataOut->pF1UcisOut;
    m_outputPrms.pPucchF3OutOffsetsCpu    = pDynPrm->pDataOut->pPucchF3OutOffsets;
    m_outputPrms.pPucchF2OutOffsetsCpu    = pDynPrm->pDataOut->pPucchF2OutOffsets;
    m_outputPrms.pHarqDetectionStatusCpu  = pDynPrm->pDataOut->HarqDetectionStatus;
    m_outputPrms.pCsiP1DetectionStatusCpu = pDynPrm->pDataOut->CsiP1DetectionStatus;
    m_outputPrms.pCsiP2DetectionStatusCpu = pDynPrm->pDataOut->CsiP2DetectionStatus;

    // device memory
    allocateDeviceMemory();

    // optional debug output
    if (m_outputPrms.debugOutputFlag)
    {
        for(auto i = 0; i < m_cuphyPucchStatPrms.nMaxCellsPerSlot; i++)
        {
            const auto cellIdx  = m_pCellDynPrms[i].cellPrmStatIdx;
            const int  NF       = CUPHY_N_TONES_PER_PRB * m_pCellStatPrms[cellIdx].nPrbUlBwp;
            const int  NT       = 14;
            const int  N_BS_ANT = m_pCellStatPrms[cellIdx].nRxAnt;

            m_tRefDataRxBufCpu[i].desc().set(CUPHY_C_16F, NF, NT, N_BS_ANT, cuphy::tensor_flags::align_tight);
            m_tRefDataRxBufCpu[i].set_addr(m_tPrmDataRxBufCpu[i].pAddr);
        }
    }
    return ret;
 }

 cuphyStatus_t PucchRx::setupComponents(bool enableCpuToGpuDescrAsyncCpy, cuphyPucchDynPrms_t *pDynPrm)
 {
    cuphyStatus_t ret = CUPHY_STATUS_SUCCESS;
    auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();

    if(m_nF0Ucis > 0)
    {
        ret = cuphySetupPucchF0Rx(m_pucchF0RxHndl,
                                  m_tPrmDataRxBufCpu.addr(),
                                  m_outputPrms.pF0UciOutGpu,
                                  m_cuphyPucchCellGrpDynPrm.nCells,
                                  m_nF0Ucis,
                                  m_pF0UciPrms,
                                  m_pCellCmnPrms,
                                  enableCpuToGpuDescrAsyncCpy,
                                  static_cast<void*>(dynCpuDescrStartAddrs[PUCCH_F0_Rx]),
                                  static_cast<void*>(dynGpuDescrStartAddrs[PUCCH_F0_Rx]),
                                  &m_pucchF0RxLaunchCfg,
                                  m_cuStream);

        if(CUPHY_STATUS_SUCCESS != ret)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "cuphySetupPucchF0Rx()");
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_F0_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }
    }

    if(m_nF1Ucis > 0)
    {
        ret = cuphySetupPucchF1Rx(m_pucchF1RxHndl,
                                  m_tPrmDataRxBufCpu.addr(),
                                  m_outputPrms.pF1UciOutGpu,
                                  m_cuphyPucchCellGrpDynPrm.nCells,
                                  m_nF1Ucis,
                                  m_pF1UciPrms,
                                  m_pCellCmnPrms,
                                  enableCpuToGpuDescrAsyncCpy,
                                  static_cast<void*>(dynCpuDescrStartAddrs[PUCCH_F1_Rx]),
                                  static_cast<void*>(dynGpuDescrStartAddrs[PUCCH_F1_Rx]),
                                  &m_pucchF1RxLaunchCfg,
                                  m_cuStream);

        if(CUPHY_STATUS_SUCCESS != ret)
        {
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_F1_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }
    }

    if(m_nF2Ucis > 0)
    {
        ret = cuphySetupPucchF2Rx(m_pucchF2RxHndl,
                                  m_tPrmDataRxBufCpu.addr(),
                                  m_F2seg1LLRaddrsVec.data(),
                                  m_pF2dtxFlags,
                                  m_pF2pSinr,
                                  m_pF2pRssi,
                                  m_pF2pRsrp,
                                  m_pF2pInterf,
                                  m_pF2pNoiseVar,
                                  m_pF2pTaEst,
                                  m_cuphyPucchCellGrpDynPrm.nCells,
                                  m_nF2Ucis,
                                  m_pF2UciPrms,
                                  m_pCellCmnPrms,
                                  enableCpuToGpuDescrAsyncCpy,
                                  static_cast<void*>(dynCpuDescrStartAddrs[PUCCH_F2_RX]),
                                  static_cast<void*>(dynGpuDescrStartAddrs[PUCCH_F2_RX]),
                                  &m_pucchF2RxLaunchCfg,
                                  m_cuStream);

        if(CUPHY_STATUS_SUCCESS != ret)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "cuphySetupPucchF2Rx()");
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_F2_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }
    }

    if(m_nF3Ucis > 0)
    {
        ret = cuphySetupPucchF3Rx(m_pucchF3RxHndl,
                                  m_tPrmDataRxBufCpu.addr(),
                                  m_F3seg1LLRaddrsVec.data(),
                                  m_pF3dtxFlags,
                                  m_pF3pSinr,
                                  m_pF3pRssi,
                                  m_pF3pRsrp,
                                  m_pF3pInterf,
                                  m_pF3pNoiseVar,
                                  m_pF3pTaEst,
                                  m_cuphyPucchCellGrpDynPrm.nCells,
                                  m_nF3Ucis,
                                  m_pF3UciPrms,
                                  m_pCellCmnPrms,
                                  enableCpuToGpuDescrAsyncCpy,
                                  static_cast<void*>(dynCpuDescrStartAddrs[PUCCH_F3_RX]),                     
                                  static_cast<void*>(dynGpuDescrStartAddrs[PUCCH_F3_RX]),                     
                                  &m_pucchF3RxLaunchCfg,                      
                                  m_cuStream); 

        if(CUPHY_STATUS_SUCCESS != ret)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "cuphySetupPucchF3Rx()");
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_F3_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }
    }

    if ((m_nF2Ucis > 0) || (m_nF3Ucis > 0)) {
        ret = cuphySetupPucchF234UciSeg(m_pucchF234UciSegHndl,
                                        m_nF2Ucis,
                                        m_nF3Ucis,
                                        m_pF2UciPrms,
                                        m_pF3UciPrms,
                                        m_outputPrms.pPucchF2OutOffsetsCpu,
                                        m_outputPrms.pPucchF3OutOffsetsCpu,
                                        m_outputPrms.pUciPayloadsGpu,
                                        static_cast<void*>(dynCpuDescrStartAddrs[PUCCH_F234_UCI_SEG]),                     
                                        static_cast<void*>(dynGpuDescrStartAddrs[PUCCH_F234_UCI_SEG]), 
                                        enableCpuToGpuDescrAsyncCpy,
                                        &m_pucchF234UciSegLaunchCfg,
                                        m_cuStream);
        if(CUPHY_STATUS_SUCCESS != ret)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "cuphySetupPucchF234UciSeg()");
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_F234_UCI_SEG_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }
    }

    if(m_nRmCws > 0)
    {
        ret = cuphySetupRmDecoder(m_rmDecodeHndl,
                                  m_nRmCws,
                                  m_pRmCwPrmsGpu,
                                  static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                  static_cast<void*>(dynCpuDescrStartAddrs[RM_DECODE]),
                                  static_cast<void*>(dynGpuDescrStartAddrs[RM_DECODE]),
                                  &m_rmDecoderLaunchCfg,
                                  m_cuStream);
        if(CUPHY_STATUS_SUCCESS != ret)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "cuphySetupRmDecoder()");
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_RM_DECODE_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }
    }

    if(m_nPolSegs > 0)
    {
        ret = cuphySetupCompCwTreeTypes(m_compCwTreeTypesHndl,
                                        m_nPolSegs,
                                        m_polSegPrmsBufCpu.addr(),
                                        m_pPolSegPrmsGpu,            
                                        m_polCwTreeTypesAddrVec.data(),        
                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE]),
                                        static_cast<void*>(dynGpuDescrStartAddrs[POL_COMP_CW_TREE]),
                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE_ADDRS]),
                                        enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                        &m_compCwTreeTypesLaunchCfg,                      
                                        m_cuStream);

        if(CUPHY_STATUS_SUCCESS != ret)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "cuphySetupCompCwTreeTypes()");
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_COMP_CW_TREE_TYPE_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }

        ret = cuphySetupPolSegDeRmDeItl(m_polSegDeRmDeItlHndl,
                                        m_nPolSegs,  
                                        m_nPolCbs,
                                        m_polSegPrmsBufCpu.addr(),
                                        m_pPolSegPrmsGpu,
                                        m_polCwPrmsBufCpu.addr(),
                                        m_pPolCwPrmsGpu,            
                                        m_polSegLLRsAddrVec.data(), 
                                        m_polCwLLRsAddrVec.data(),       
                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL]),
                                        static_cast<void*>(dynGpuDescrStartAddrs[POL_SEG_DERM_DEITL]),
                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_CW_ADDRS]),
                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_UCI_ADDRS]),
                                        enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                        &m_polSegDeRmDeItlLaunchCfg,                  
                                        m_cuStream);


        if(CUPHY_STATUS_SUCCESS != ret)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "cuphySetupPolSegDeRmDeItl()");
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_POLAR_SEG_RATE_MATCH_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }

        ret = cuphySetupPolarDecoder(m_polarDecoderHndl,
                                     m_nPolCbs,
                                     m_polCwTreeLLRsAddrVec.data(),
                                     m_pPolCwPrmsGpu,
                                     m_polCwPrmsBufCpu.addr(),
                                     m_polCbEstAddrVec.data(),
                                     m_listPolScratchAddrVec.data(),
                                     m_polDcdrListSz,
                                     m_pPolCrcFlags,
                                     static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                     static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE]),
                                     static_cast<void*>(dynGpuDescrStartAddrs[POL_DECODE]), 
                                     static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_LLR_ADDRS]),
                                     static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_CB_ADDRS]),
                                     static_cast<void*>(dynCpuDescrStartAddrs[LIST_POL_DECODE_SCRATCH_ADDRS]),
                                     &m_polarDecoderLaunchCfg,
                                     m_cuStream);

        if(CUPHY_STATUS_SUCCESS != ret)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "cuphySetupPolarDecoder()");
            pDynPrm->pStatusOut->status = cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_POLAR_DECODE_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return ret;
        }
    }
    return CUPHY_STATUS_SUCCESS;
 }
 
 cuphyStatus_t PucchRx::setup(cuphyPucchDynPrms_t *pDynPrm)
 {
    cuphyStatus_t ret = CUPHY_STATUS_SUCCESS;
    // common setup (shared by both stream and graph modes)
    ret = setupCmn(pDynPrm);
    if(ret != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Common Setup failed");
        return ret;
    }

    // setup
    bool enableCpuToGpuDescrAsyncCpy = false;
    ret = setupComponents(enableCpuToGpuDescrAsyncCpy, pDynPrm);
    if(ret != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Component Setup failed");
        return ret;
    }

    // executable graph setup
    m_cudaGraphModeEnabled = (pDynPrm->procModeBmsk & PUCCH_PROC_MODE_FULL_SLOT_GRAPHS) ? true : false;
    if (m_cudaGraphModeEnabled) 
    {
        updateGraph();
    }

    // copy descriptors to GPU
    if(!enableCpuToGpuDescrAsyncCpy)
     {
         m_kernelDynDescr.asyncCpuToGpuCpy(m_cuStream);
     }


    if(m_nRmCws > 0)
    {
        cudaMemcpyAsync(static_cast<void*>(m_pRmCwPrmsGpu), static_cast<void*>(m_rmCwPrmsBufCpu.addr()), m_nRmCws *sizeof(cuphyRmCwPrm_t), cudaMemcpyHostToDevice, m_cuStream);
    }

    if(m_nPolSegs > 0)
    {
        cudaMemcpyAsync(static_cast<void*>(m_pPolSegPrmsGpu), static_cast<void*>(m_polSegPrmsBufCpu.addr()), m_nPolSegs * sizeof(cuphyPolarUciSegPrm_t), cudaMemcpyHostToDevice, m_cuStream);
        cudaMemcpyAsync(static_cast<void*>(m_pPolCwPrmsGpu) , static_cast<void*>(m_polCwPrmsBufCpu.addr()), m_nPolCbs * sizeof(cuphyPolarCwPrm_t), cudaMemcpyHostToDevice, m_cuStream);
    }
    return ret;
  }

void PucchRx::copyOutputToCPU()
 {
     if(m_nF0Ucis > 0)
     {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pF0UciOutCpu, m_outputPrms.pF0UciOutGpu, sizeof(cuphyPucchF0F1UciOut_t) * m_nF0Ucis, cudaMemcpyDeviceToHost, m_cuStream));
     }

    if(m_nF1Ucis > 0)
     {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pF1UciOutCpu, m_outputPrms.pF1UciOutGpu, sizeof(cuphyPucchF0F1UciOut_t) * m_nF1Ucis, cudaMemcpyDeviceToHost, m_cuStream));
     }

     if(m_outputPrms.nUciPayloadBytes > 0)
     {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pUciPayloadsCpu, m_outputPrms.pUciPayloadsGpu, sizeof(uint8_t) * m_outputPrms.nUciPayloadBytes, cudaMemcpyDeviceToHost, m_cuStream));
        //CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pDtxFlagsCpu, m_outputPrms.pDtxFlagsGpu, sizeof(uint8_t) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pSinrCpu, m_outputPrms.pSinrGpu, sizeof(float) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pRssiCpu, m_outputPrms.pRssiGpu, sizeof(float) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pRsrpCpu, m_outputPrms.pRsrpGpu, sizeof(float) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pInterfCpu, m_outputPrms.pInterfGpu, sizeof(float) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pTaEstCpu, m_outputPrms.pTaEstGpu, sizeof(float) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pCrcFlagsCpu, m_outputPrms.pCrcFlagsGpu, sizeof(uint8_t) * m_outputPrms.nUciSegs, cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pHarqDetectionStatusCpu, m_outputPrms.pHarqDetectionStatusGpu, sizeof(uint8_t) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pCsiP1DetectionStatusCpu, m_outputPrms.pCsiP1DetectionStatusGpu, sizeof(uint8_t) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pCsiP2DetectionStatusCpu, m_outputPrms.pCsiP2DetectionStatusGpu, sizeof(uint8_t) * (m_nF3Ucis + m_nF2Ucis), cudaMemcpyDeviceToHost, m_cuStream));
     }
 }

 void PucchRx::run()
 {
     if(m_cudaGraphModeEnabled)
     {
         MemtraceDisableScope md; // Disable temporarity GT-7257
         CU_CHECK_EXCEPTION(cuGraphLaunch(m_graphExec, m_cuStream));
     }
     else
     {
         // PUCCH F0 reciever
         if(m_nF0Ucis > 0)
         {
             const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pucchF0RxLaunchCfg.kernelNodeParamsDriver;
             CUresult pucchF0RxRunStatus = launch_kernel(kernelNodeParamsDriver, m_cuStream);
             if(CUDA_SUCCESS != pucchF0RxRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
         }

         // PUCCH F1 reciever
         if(m_nF1Ucis > 0)
         {
             const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pucchF1RxLaunchCfg.kernelNodeParamsDriver;
             CUresult pucchF1RxRunStatus = launch_kernel(kernelNodeParamsDriver, m_cuStream);
             if(CUDA_SUCCESS != pucchF1RxRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
         }

         // PUCCH F2 reciever (front-end)
         if(m_nF2Ucis > 0)
         {
             const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pucchF2RxLaunchCfg.kernelNodeParamsDriver;
             CUresult pucchF2RxRunStatus = launch_kernel(kernelNodeParamsDriver, m_cuStream);
             if(CUDA_SUCCESS != pucchF2RxRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
         }

         // PUCCH F3 reciever (front-end)
         if(m_nF3Ucis > 0)
         {
             const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pucchF3RxLaunchCfg.kernelNodeParamsDriver;
             CUresult pucchF3RxRunStatus = launch_kernel(kernelNodeParamsDriver, m_cuStream);
             if(CUDA_SUCCESS != pucchF3RxRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
         }

         // Reed-Muller decoder
         if(m_nRmCws > 0)
         {
             const CUDA_KERNEL_NODE_PARAMS&  kernelNodeParamsDriver = m_rmDecoderLaunchCfg.kernelNodeParamsDriver;
             CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));

         }

         if(m_nPolSegs > 0)
         {
             const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver1 = m_compCwTreeTypesLaunchCfg.kernelNodeParamsDriver;
             CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver1, m_cuStream));

             const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver2 = m_polSegDeRmDeItlLaunchCfg.kernelNodeParamsDriver;
             CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver2, m_cuStream));

             const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver3 = m_polarDecoderLaunchCfg.kernelNodeParamsDriver;
             CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver3, m_cuStream));
         }

         // PF2/3/4 UCI Segmentation
         if ((m_nF2Ucis > 0) || (m_nF3Ucis > 0)) {
             const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pucchF234UciSegLaunchCfg.kernelNodeParamsDriver;
             CUresult pucchF234UciSegRunStatus = launch_kernel(kernelNodeParamsDriver, m_cuStream);
             if(CUDA_SUCCESS != pucchF234UciSegRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
         }
     }

     if(m_outputPrms.cpuCopyOn)
     {
         copyOutputToCPU();
     }
 }

 template <fmtlog::LogLevel log_level>
 void PucchRx::printUciPrms(const cuphyPucchUciPrm_t* uciPrms)
 {
    NVLOG_FMT(log_level, NVLOG_PUCCH,"uciPrms:         {:p}",  static_cast<void*>(const_cast<cuphyPucchUciPrm_t*>(uciPrms)));
    NVLOG_FMT(log_level, NVLOG_PUCCH,"formatType:      {}", uciPrms->formatType);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"cellPrmDynIdx:   {}", uciPrms->cellPrmDynIdx);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"cellPrmStatIdx:  {}", uciPrms->cellPrmStatIdx);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"uciOutputIdx:    {}", uciPrms->uciOutputIdx);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"rnti:            {}", uciPrms->rnti);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"bwpStart:        {}", uciPrms->bwpStart);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"multiSlotTxIndicator: {}", uciPrms->multiSlotTxIndicator);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"pi2Bpsk:         {}", uciPrms->pi2Bpsk);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"startPrb:        {}", uciPrms->startPrb);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"prbSize:         {}", uciPrms->prbSize);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"startSym:        {}", uciPrms->startSym);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nSym:            {}", uciPrms->nSym);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"freqHopFlag:     {}", uciPrms->freqHopFlag);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"secondHopPrb:    {}", uciPrms->secondHopPrb);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"groupHopFlag:    {}", uciPrms->groupHopFlag);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"sequenceHopFlag: {}", uciPrms->sequenceHopFlag);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"initialCyclicShift: {}", uciPrms->initialCyclicShift);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"timeDomainOccIdx:{}", uciPrms->timeDomainOccIdx);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"srFlag:          {}", uciPrms->srFlag);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"bitLenSr:        {}", uciPrms->bitLenSr);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"bitLenHarq:      {}", uciPrms->bitLenHarq);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"bitLenCsiPart1:  {}", uciPrms->bitLenCsiPart1);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"AddDmrsFlag:     {}", uciPrms->AddDmrsFlag);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"dataScramblingId:{}", uciPrms->dataScramblingId);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"DmrsScramblingId:{}", uciPrms->DmrsScramblingId);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"maxCodeRate:     {}", uciPrms->maxCodeRate);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nBitsCsi2:       {}", uciPrms->nBitsCsi2);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"rankBitOffset:   {}", uciPrms->rankBitOffset);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nRanksBits:      {}", uciPrms->nRanksBits);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"DTXthreshold:    {}", uciPrms->DTXthreshold);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"numPart2s:       {}", uciPrms->uciP1P2Crpd_t.numPart2s);
 }

 template <fmtlog::LogLevel log_level>
 void PucchRx::printDynApiPrms(cuphyPucchDynPrms_t* pDynPrm)
 {
    NVLOG_FMT(log_level, NVLOG_PUCCH,"===============================================");
    NVLOG_FMT(log_level, NVLOG_PUCCH,"===========print PUCCH DynApiPrms==============");
    NVLOG_FMT(log_level, NVLOG_PUCCH,"procModeBmsk: {:x}", pDynPrm->procModeBmsk);

    const cuphyPucchCellGrpDynPrm_t* pCellGrpDynPrm = pDynPrm->pCellGrpDynPrm;
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nCells: {}", pCellGrpDynPrm->nCells);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"===============================================\n");
    for (uint16_t i = 0 ; i < pCellGrpDynPrm->nCells; i++)
    {
        NVLOG_FMT(log_level, NVLOG_PUCCH,"-->Cell[{}]\n", i);
        cuphyPucchCellDynPrm_t* pCellDynPrm = &pCellGrpDynPrm->pCellPrms[i];
        NVLOG_FMT(log_level, NVLOG_PUCCH,"pCellPrms:      {:p}", static_cast<void*>(pCellDynPrm));
        NVLOG_FMT(log_level, NVLOG_PUCCH,"cellPrmStatIdx: {}", pCellDynPrm->cellPrmStatIdx);
        NVLOG_FMT(log_level, NVLOG_PUCCH,"cellPrmDynIdx:  {}", pCellDynPrm->cellPrmDynIdx);
        NVLOG_FMT(log_level, NVLOG_PUCCH,"slotNum:        {}", pCellDynPrm->slotNum);
        NVLOG_FMT(log_level, NVLOG_PUCCH,"pucchHoppingId: {}",  pCellDynPrm->pucchHoppingId);
    }

    const std::vector<cuphyPucchUciPrm_t*> paramVec {pCellGrpDynPrm->pF0UciPrms,
                           pCellGrpDynPrm->pF1UciPrms,pCellGrpDynPrm->pF2UciPrms,
                           pCellGrpDynPrm->pF3UciPrms};
    const std::vector<int> uciCntVec {pCellGrpDynPrm->nF0Ucis,pCellGrpDynPrm->nF1Ucis,
                                        pCellGrpDynPrm->nF2Ucis,pCellGrpDynPrm->nF3Ucis};
    for(int fmtNum = 0; fmtNum < uciCntVec.size(); fmtNum++)
    {
        NVLOG_FMT(log_level, NVLOG_PUCCH,"===============================================");
        NVLOG_FMT(log_level, NVLOG_PUCCH,"nF{}Ucis: {}",fmtNum,  uciCntVec[fmtNum]);
        NVLOG_FMT(log_level, NVLOG_PUCCH,"===============================================");
        for (int uciIdx = 0; uciIdx < uciCntVec[fmtNum]; uciIdx++) {
            NVLOG_FMT(log_level, NVLOG_PUCCH,"-->UCI[{}]", uciIdx);
            const cuphyPucchUciPrm_t* uciPrms = &paramVec[fmtNum][uciIdx];
            printUciPrms<log_level>(uciPrms);
        }
    }
 }

 template <fmtlog::LogLevel log_level>
 void PucchRx::printStaticApiPrms(cuphyPucchStatPrms_t const* pStaticPrm)
 {
    NVLOG_FMT(log_level, NVLOG_PUCCH,"===============================================");
    NVLOG_FMT(log_level, NVLOG_PUCCH,"=======print PUCCH printStaticApiPrms==========");
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nMaxCells:        {}", pStaticPrm->nMaxCells);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nMaxCellsPerSlot: {}", pStaticPrm->nMaxCellsPerSlot);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"uciOutputMode:    {}", pStaticPrm->uciOutputMode);

    const cuphyCellStatPrm_t* pCellStatPrms = pStaticPrm->pCellStatPrms;
    NVLOG_FMT(log_level, NVLOG_PUCCH,"===============================================");
    NVLOG_FMT(log_level, NVLOG_PUCCH,"cuphyCellStatPrm_t:");
    NVLOG_FMT(log_level, NVLOG_PUCCH,"===============================================");

    NVLOG_FMT(log_level, NVLOG_PUCCH,"phyCellId: {}", pCellStatPrms->phyCellId);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nRxAnt:    {}", pCellStatPrms->nRxAnt);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nTxAnt:    {}", pCellStatPrms->nTxAnt);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nPrbUlBwp: {}", pCellStatPrms->nPrbUlBwp);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nPrbDlBwp: {}", pCellStatPrms->nPrbDlBwp);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"mu:        {}", pCellStatPrms->mu);

    const cuphyPucchCellStatPrm_t* pPucchCellStatPrms = pCellStatPrms->pPucchCellStatPrms;
    NVLOG_FMT(log_level, NVLOG_PUCCH,"nCsirsPorts:  {}", pPucchCellStatPrms->nCsirsPorts);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"N1:           {}", pPucchCellStatPrms->N1);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"N2:           {}", pPucchCellStatPrms->N2);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"csiReportingBand: {}", pPucchCellStatPrms->csiReportingBand);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"codebookType: {}", pPucchCellStatPrms->codebookType);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"codebookMode: {}", pPucchCellStatPrms->codebookMode);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"isCqi:        {}", pPucchCellStatPrms->isCqi);
    NVLOG_FMT(log_level, NVLOG_PUCCH,"isLi:         {}", pPucchCellStatPrms->isLi);
 }

 void PucchRx::writeDbgBufSynch(cudaStream_t cuStream)
 {
    if(m_outputPrms.debugOutputFlag)
    {
        // Input Data
        for (size_t i = 0; i < m_tRefDataRxBufCpu.size(); i++) {
            std::string name = "DataRx_cell_" + std::to_string(i);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefDataRxBufCpu[i], name.c_str(), cuStream);
        }
        // Output Data
        if(0<m_outputPrms.nUciPayloadBytes)
        {
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tUciPayload,           m_tUciPayload.desc(),           "pucchF234_UciPayload", cuStream);
        }
        if(0<(m_nF3Ucis + m_nF2Ucis))
        {
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tHarqDetectionStatus,  m_tHarqDetectionStatus.desc(),  "pucchF234_HarqDetStat", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tCsiP1DetectionStatus, m_tCsiP1DetectionStatus.desc(), "pucchF234_CsiPart1DetStat", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tCsiP2DetectionStatus, m_tCsiP2DetectionStatus.desc(), "pucchF234_CsiPart2DetStat", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tDtxFlags,             m_tDtxFlags.desc(),             "pucchF234_DTXFlags", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tSinr,                 m_tSinr.desc(),                 "pucchF234_Sinr", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRssi,                 m_tRssi.desc(),                 "pucchF234_Rssi", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRsrp,                 m_tRsrp.desc(),                 "pucchF234_Rsrp", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tInterf,               m_tInterf.desc(),               "pucchF234_Interf", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tNoiseVar,             m_tNoiseVar.desc(),             "pucchF234_NoiseVar", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tTaEst,                m_tTaEst.desc(),                "pucchF234_TaEst", cuStream);
        }
    }
 }

 size_t PucchRx::getBufferSize(cuphyPucchStatPrms_t const* pStatPrms)
 {

     const int32_t EXTRA_PADDING = MAX_N_USER_GROUPS_SUPPORTED * 128; // upper bound for extra memory required per allocation due to 128 alignment
                                                                      // ToDo FIXME could change this upper limit to be less conservative
     size_t nBytesBuffer = 0;

     //Find the max UL BWP and max layers across all cells
     uint32_t max_nPrbUlBwp = 0;
     for(uint16_t i = 0; i < pStatPrms->nMaxCells; i++)
     {
         if(max_nPrbUlBwp < pStatPrms->pCellStatPrms[i].nPrbUlBwp)
         {
             max_nPrbUlBwp = pStatPrms->pCellStatPrms[i].nPrbUlBwp;
         }
     }

     // List polar decoder
     if (pStatPrms->polarDcdrListSz > 1) {
         nBytesBuffer += sizeof(bool) * (2 * CUPHY_POLAR_DECODER_MAX_BITS * pStatPrms->polarDcdrListSz) + EXTRA_PADDING;
     }

     // descram LLRs
     nBytesBuffer += max_nPrbUlBwp * 12 * 14 * sizeof(__half) + EXTRA_PADDING;

     // F3 output bytes
     nBytesBuffer += 14*12*14*2 * CUPHY_PUCCH_F3_MAX_UCI + EXTRA_PADDING;

     // F0 UCI outputs
     nBytesBuffer += sizeof(cuphyPucchF0F1UciOut_t) * CUPHY_PUCCH_F0_MAX_GRPS * CUPHY_PUCCH_F0_MAX_UCI_PER_GRP + EXTRA_PADDING;

     // account for F2/F3 TaEst, Sinr, Rssi, Rsrp, and Interf
     nBytesBuffer += (CUPHY_PUCCH_F3_MAX_UCI + CUPHY_PUCCH_F2_MAX_UCI)*sizeof(float)*5;

     // account for F2/F3 NoiseVar
     nBytesBuffer += (CUPHY_PUCCH_F3_MAX_UCI + CUPHY_PUCCH_F2_MAX_UCI)*sizeof(float);

     // account for F2/F3 DtxFlag and CrcFlag
     nBytesBuffer += (CUPHY_PUCCH_F3_MAX_UCI + CUPHY_PUCCH_F2_MAX_UCI)*2;

     // account for multiple-cells per group
     nBytesBuffer *= pStatPrms->nMaxCells;

     // F1 UCI outputs for all cells
     nBytesBuffer += sizeof(cuphyPucchF0F1UciOut_t) * CUPHY_PUCCH_F1_MAX_GRPS * CUPHY_PUCCH_F1_MAX_UCI_PER_GRP + EXTRA_PADDING*pStatPrms->nMaxCells;

     return nBytesBuffer;
 }

 void PucchRx::allocateDescr()
 {
     // zero-initialize sizes and allignments
     std::array<size_t, N_PUCCH_COMPONENTS> statDescrSizeBytes{};
     std::array<size_t, N_PUCCH_COMPONENTS> statDescrAlignBytes{};
     std::array<size_t, N_PUCCH_COMPONENTS> dynDescrSizeBytes{};
     std::array<size_t, N_PUCCH_COMPONENTS> dynDescrAlignBytes{};

     size_t *pStatDescrSizeBytes  = statDescrSizeBytes.data();
     size_t *pStatDescrAlignBytes = statDescrAlignBytes.data();
     size_t *pDynDescrSizeBytes   = dynDescrSizeBytes.data();
     size_t *pDynDescrAlignBytes  = dynDescrAlignBytes.data();

     // get sizes and alignments
     pDynDescrSizeBytes[PUCCH_CELL_INFO]  = sizeof(cuphyPucchCellPrm_t) * m_cuphyPucchStatPrms.nMaxCellsPerSlot;
     pDynDescrAlignBytes[PUCCH_CELL_INFO] = alignof(cuphyPucchCellPrm_t);

     cuphyStatus_t status = cuphyPucchF0RxGetDescrInfo(&pDynDescrSizeBytes[PUCCH_F0_Rx],
                                                       &pDynDescrAlignBytes[PUCCH_F0_Rx]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyPucchF0RxGetDescrInfo()");
     }

     status = cuphyPucchF1RxGetDescrInfo(&pDynDescrSizeBytes[PUCCH_F1_Rx],
                                         &pDynDescrAlignBytes[PUCCH_F1_Rx]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyPucchF1RxGetDescrInfo()");
     }

     status = cuphyPucchF2RxGetDescrInfo(&pDynDescrSizeBytes[PUCCH_F2_RX],
                                         &pDynDescrAlignBytes[PUCCH_F2_RX]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyPucchF2RxGetDescrInfo()");
     }

     status = cuphyPucchF3RxGetDescrInfo(&pDynDescrSizeBytes[PUCCH_F3_RX],
                                         &pDynDescrAlignBytes[PUCCH_F3_RX]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyPucchF3RxGetDescrInfo()");
     }

     status = cuphyPucchF234UciSegGetDescrInfo(&pDynDescrSizeBytes[PUCCH_F234_UCI_SEG],
                                               &pDynDescrAlignBytes[PUCCH_F234_UCI_SEG]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyPucchF234UciSegGetDescrInfo()");
     }

     status = cuphyCompCwTreeTypesGetDescrInfo(&pDynDescrSizeBytes[POL_COMP_CW_TREE], &pDynDescrAlignBytes[POL_COMP_CW_TREE]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCompCwTreeTypesGetDescrInfo()");
     }

     pDynDescrSizeBytes[POL_COMP_CW_TREE_ADDRS]  = sizeof(uint8_t*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPucchStatPrms.nMaxCellsPerSlot;
     pDynDescrAlignBytes[POL_COMP_CW_TREE_ADDRS] = alignof(uint8_t*);

     status = cuphyPolSegDeRmDeItlGetDescrInfo(&pDynDescrSizeBytes[POL_SEG_DERM_DEITL], &pDynDescrAlignBytes[POL_SEG_DERM_DEITL]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyPolSegDeRmDeItlGetDescrInfo()");
     }

     pDynDescrSizeBytes[POL_SEG_DERM_DEITL_CW_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPucchStatPrms.nMaxCellsPerSlot;
     pDynDescrAlignBytes[POL_SEG_DERM_DEITL_CW_ADDRS] = alignof(__half*);

     pDynDescrSizeBytes[POL_SEG_DERM_DEITL_UCI_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPucchStatPrms.nMaxCellsPerSlot;
     pDynDescrAlignBytes[POL_SEG_DERM_DEITL_UCI_ADDRS] = alignof(__half*);

     status = cuphyPolarDecoderGetDescrInfo(&pDynDescrSizeBytes[POL_DECODE], &pDynDescrAlignBytes[POL_DECODE]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyPolarDecoderGetDescrInfo()");
     }

     pDynDescrSizeBytes[POL_DECODE_LLR_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_CWS * m_cuphyPucchStatPrms.nMaxCellsPerSlot;
     pDynDescrAlignBytes[POL_DECODE_LLR_ADDRS] = alignof(__half*);

     pDynDescrSizeBytes[POL_DECODE_CB_ADDRS]  = sizeof(uint32_t*) * CUPHY_MAX_N_POL_CWS * m_cuphyPucchStatPrms.nMaxCellsPerSlot;
     pDynDescrAlignBytes[POL_DECODE_CB_ADDRS] = alignof(uint32_t*);

     pDynDescrSizeBytes[LIST_POL_DECODE_SCRATCH_ADDRS]  = sizeof(bool*) * CUPHY_MAX_N_POL_CWS * m_cuphyPucchStatPrms.nMaxCellsPerSlot;
     pDynDescrAlignBytes[LIST_POL_DECODE_SCRATCH_ADDRS] = alignof(bool*);

     status = cuphyRmDecoderGetDescrInfo(&pDynDescrSizeBytes[RM_DECODE], &pDynDescrAlignBytes[RM_DECODE]);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyRmDecoderGetDescrInfo()");
     }

     // Allocate descriptors
     m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes, &m_memoryFootprint);
     m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes, &m_memoryFootprint);

 }

 void PucchRx::createComponents()
 {
     cuphyStatus_t status = cuphyCreatePucchF0Rx(&m_pucchF0RxHndl, m_cuStream);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreatePucchF0Rx()");
     }

     status = cuphyCreatePucchF1Rx(&m_pucchF1RxHndl, m_cuStream);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreatePucchF1Rx()");
     }

     status = cuphyCreatePucchF2Rx(&m_pucchF2RxHndl, m_cuStream);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreatePucchF2Rx()");
     }

     status = cuphyCreatePucchF3Rx(&m_pucchF3RxHndl, m_cuStream);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreatePucchF3Rx()");
     }

     status = cuphyCreatePucchF234UciSeg(&m_pucchF234UciSegHndl);
     if(CUPHY_STATUS_SUCCESS != status)
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreatePucchF234UciSeg()");
     }

     unsigned int rmFlags = 0;
     cuphy::context       ctx;
     status = cuphyCreateRmDecoder(ctx.handle(), &m_rmDecodeHndl, rmFlags, &m_memoryFootprint);
     if(CUPHY_STATUS_SUCCESS != status) 
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreatePucchRmDecoder()");
     }

     status = cuphyCreateCompCwTreeTypes(&m_compCwTreeTypesHndl);
     if(CUPHY_STATUS_SUCCESS != status) 
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreateCompCwTreeTypes()");
     }

     status = cuphyCreatePolSegDeRmDeItl(&m_polSegDeRmDeItlHndl);
     if(CUPHY_STATUS_SUCCESS != status) 
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreatePolSegDeRmDeItl()");
     }

     status = cuphyCreatePolarDecoder(&m_polarDecoderHndl);
     if(CUPHY_STATUS_SUCCESS != status) 
     {
         throw cuphy::cuphy_fn_exception(status, "cuphyCreatePolarDecoder()");
     }
 }

void PucchRx::destroyComponents()
{
    cuphyStatus_t statusDestroy = cuphyDestroyPucchF0Rx(m_pucchF0RxHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy)
    {
        throw cuphy::cuphy_fn_exception(statusDestroy, "cuphyDestroyPucchF0Rx()");
    }

    statusDestroy = cuphyDestroyPucchF1Rx(m_pucchF1RxHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy)
    {
        throw cuphy::cuphy_fn_exception(statusDestroy, "cuphyDestroyPucchF1Rx()");
    }

    statusDestroy = cuphyDestroyPucchF2Rx(m_pucchF2RxHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy)
    {
        throw cuphy::cuphy_fn_exception(statusDestroy, "cuphyDestroyPucchF2Rx()");
    }

    statusDestroy = cuphyDestroyPucchF3Rx(m_pucchF3RxHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy)
    {
        throw cuphy::cuphy_fn_exception(statusDestroy, "cuphyDestroyPucchF3Rx()");
    }

    statusDestroy = cuphyDestroyPucchF234UciSeg(m_pucchF234UciSegHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy)
    {
        throw cuphy::cuphy_fn_exception(statusDestroy, "cuphyDestroyPucchF234UciSeg()");
    }

    statusDestroy = cuphyDestroyRmDecoder(m_rmDecodeHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy) 
    {
        throw cuphy::cuphy_fn_exception(statusDestroy, "cuphyDestroyRmDecoder()");
    }
     
}

PucchRx::~PucchRx()
{
    CUDA_CHECK(cudaGraphDestroy(m_graph));
    CUDA_CHECK(cudaGraphExecDestroy(m_graphExec));
    destroyComponents();
}
// NOTE: currently only supports 1 UCI segment (HARQ + SR + CSI1)
  PucchRx::F234RmSizes_t PucchRx::compRateMatchSizesF2(cuphyPucchUciPrm_t& F2uciPrms)
  {
      F234RmSizes_t rmSizes;
      rmSizes.E_seg1 = F2uciPrms.nSym*F2uciPrms.prbSize*16;

      return rmSizes;
  }


// NOTE: currently only supports 1 UCI segment (HARQ + SR + CSI1)
  PucchRx::F234RmSizes_t PucchRx::compRateMatchSizesF3(cuphyPucchUciPrm_t& F3uciPrms)
  {
    F234RmSizes_t rmSizes;
    uint8_t nSym         = F3uciPrms.nSym;
    uint8_t freqHopFlag  = F3uciPrms.freqHopFlag;
    uint8_t AddDmrsFlag  = F3uciPrms.AddDmrsFlag;
    uint8_t prbSize      = F3uciPrms.prbSize;
    uint8_t pi2Bpsk      = F3uciPrms.pi2Bpsk;

    uint8_t nSym_data = 0;
    switch (int(nSym))
     {
       case 4:
        if (freqHopFlag) {
          nSym_data = 2;
        }
        else {
          nSym_data = 3;
        }
        break;
       case 5:
        nSym_data = nSym - 2;
        break;
       case 6:
        nSym_data = nSym - 2;
        break;
       case 7:
        nSym_data = nSym - 2;
        break;
       case 8:
        nSym_data = nSym - 2;
        break;
       case 9:
        nSym_data = nSym - 2;  
        break;
       case 10:
        if (AddDmrsFlag) {
          nSym_data = nSym - 4;
        } else {
          nSym_data = nSym - 2;
        }
        break;
       case 11:
        if (AddDmrsFlag) {
          nSym_data = nSym - 4;
        } else {
          nSym_data = nSym - 2;
        }
        break;
       case 12:
        if (AddDmrsFlag) {
          nSym_data = nSym - 4;
        } else {
          nSym_data = nSym - 2;
        }
        break;
       case 13:
        if (AddDmrsFlag) {
          nSym_data = nSym - 4;
        } else {
          nSym_data = nSym - 2;
        }
        break;
       case 14:
        if (AddDmrsFlag) {
          nSym_data = nSym - 4;
        } else {
          nSym_data = nSym - 2;
        }
        break;
       default:
        throw std::out_of_range(fmt::format("Invalid number of symbols ({}) for PUCCH format 3",nSym));
     }

    uint16_t E_seg1 = 0;
     if (pi2Bpsk) {
       E_seg1 = 12 * nSym_data * prbSize;
     } else {
       E_seg1 = 24 * nSym_data * prbSize;
     }

     rmSizes.E_seg1 = E_seg1;
     return rmSizes;
  }

 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchRx()

 cuphyStatus_t CUPHYWINAPI cuphyCreatePucchRx(cuphyPucchRxHndl_t* pPucchRxHndl, cuphyPucchStatPrms_t const* pStatPrms, cudaStream_t cuStream) 
 {
     if(!pPucchRxHndl || !pStatPrms || !cuStream)
     {
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }
     
     *pPucchRxHndl = nullptr;
     return cuphy::tryCallableAndCatch([&]
     {
        if (pStatPrms->pDbg->enableStatApiLogging) {
            PucchRx::printStaticApiPrms(pStatPrms);
        }
        PucchRx* p    = new PucchRx(pStatPrms, cuStream);
        *pPucchRxHndl = static_cast<cuphyPucchRxHndl_t>(p);
     });
 }


#if 0
const void* cuphyGetMemoryFootprintTrackerPucchRx(cuphyPucchRxHndl_t pucchRxHndl)
{
    if(pucchRxHndl == nullptr)
    {
        return nullptr;
    }
    PucchRx* pipeline_ptr  = static_cast<PucchRx*>(pucchRxHndl);
    return pipeline_ptr->getMemoryTracker();
}
#endif

const void* PucchRx::getMemoryTracker()
{
    return &m_memoryFootprint;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphySetupPucchRx()

 cuphyStatus_t CUPHYWINAPI cuphySetupPucchRx(cuphyPucchRxHndl_t pucchRxHndl, cuphyPucchDynPrms_t* pDynPrms, cuphyPucchBatchPrmHndl_t const batchPrmHndl)
 {
    MemtraceDisableScope md; // Disable temporarity GT-7257
    if(!pucchRxHndl || !pDynPrms)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&]
    {
        PUSH_RANGE("cuphySetupPucchRx", 1);
        if (pDynPrms->pDbg->enableDynApiLogging) {
            PucchRx::printDynApiPrms(pDynPrms);
        }
        PucchRx* p = static_cast<PucchRx*>(pucchRxHndl);
        cuphyStatus_t ret = p->setup(pDynPrms);
        if(0)
        {
            PucchRx::printDynApiPrms<fmtlog::ERR>(pDynPrms);
        }
        POP_RANGE;
    });
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 // cuphyRunPucchRx()

cuphyStatus_t CUPHYWINAPI cuphyRunPucchRx(cuphyPucchRxHndl_t pucchRxHndl, uint64_t procModeBmsk)
{
    MemtraceDisableScope md; // Disable temporarity GT-7257
    if(!pucchRxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&]
    {
        PUSH_RANGE("cuphyRunPucchRx", 2);
        PucchRx* p = static_cast<PucchRx*>(pucchRxHndl);
        p->run();
        POP_RANGE;
    });
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 // cuphyDestroyPucchRx()

 cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchRx(cuphyPucchRxHndl_t pucchRxHndl)
 {
    if(!pucchRxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    PucchRx* p = static_cast<PucchRx*>(pucchRxHndl);
    delete p;
    return CUPHY_STATUS_SUCCESS;
 }


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 // cuphyWriteDbgBufSynchPucch()

 cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchPucch(cuphyPucchRxHndl_t pucchRxHndl, cudaStream_t cuStream)
 {
     if(!pucchRxHndl)
     {
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }
    return cuphy::tryCallableAndCatch([&]
    {
        PucchRx* p = static_cast<PucchRx*>(pucchRxHndl);
        p->writeDbgBufSynch(cuStream);
        // p->copyOutputToCPU(cuStream);
        //p->printInfo();
    });
 }
