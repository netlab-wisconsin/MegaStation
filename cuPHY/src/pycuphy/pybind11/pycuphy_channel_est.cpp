/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <vector>

#include "cuphy.h"
#include "util.hpp"
#include "cuphy.hpp"
#include "pycuphy_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace cuphy;

using namespace std::complex_literals;
namespace py = pybind11;

#ifndef PYCUPHY_CHANNEL_EST_CPP
#define PYCUPHY_CHANNEL_EST_CPP


typedef struct
{
    void *              inputDevicePtr;           // Input data device memory pointer
    void *              outputChannelDevicePtr;   // Output channel estimate device memory pointer
    void *              outputChannelHostPtr;     // Output channel estimate host memory pointer
    void *              debugDevicePtr;           // Debug information device memory pointer

    cudaStream_t        cuStream;           // CUDA Stream

} cuphyChannelEstimationPrms_t;


namespace pycuphy {

class ChannelEstimator {
    public:
        enum DescriptorTypes
        {
            CH_EST               = 0,
            CH_EST_PARAMS        = CH_EST + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST,
            N_CH_EST_DESCR_TYPES = CH_EST_PARAMS + 1
        };

        ChannelEstimator(
            const py::dict& chEstFilters,
            const uint64_t inputDevicePtr,
            const uint64_t outputChannelHostPtr,
            const uint64_t outputChannelDevicePtr,
            const uint64_t debugDevicePtr,
            uint64_t cuStream);
        ~ChannelEstimator();

        py::array_t<std::complex<float>> estimate(const py::array& inputData,
                                                  const uint8_t nUes,
                                                  const uint16_t numRxAnt,
                                                  const uint16_t slotNum,
                                                  const uint16_t startPrb,
                                                  const uint16_t numPrb,
                                                  const uint8_t startSym,
                                                  const uint8_t numSym,
                                                  const uint16_t dmrsSymLocBmsk,
                                                  const uint16_t dmrsScrmId,
                                                  const uint8_t dmrsMaxLen,
                                                  const uint8_t dmrsAddlnPos,
                                                  const uint8_t numDmrsCdmGrpsNoData,
                                                  const uint8_t scid,
                                                  const std::vector<uint8_t>& numLayers,
                                                  const std::vector<uint16_t>& dmrsPortBmsks);

    private:

        void destroy();
        void setFilters(const py::dict& chEstFilters);
        void allocateDescr();
        void setChEstParams(const uint8_t nUes,
                            const uint16_t numRxAnt,
                            const uint16_t slotNum,
                            const uint16_t startPrb,
                            const uint16_t numPrb,
                            const uint8_t startSym,
                            const uint8_t numSym,
                            const uint16_t dmrsSymLocBmsk,
                            const uint16_t dmrsScrmId,
                            const uint8_t dmrsMaxLen,
                            const uint8_t dmrsAddlnPos,
                            const uint8_t numDmrsCdmGrpsNoData,
                            const uint8_t scid,
                            const std::vector<uint8_t>& numLayers,
                            const std::vector<uint16_t>& dmrsPortBmsks);

        cuphyChannelEstimationPrms_t m_chEstPrms;
        cuphyPuschRxChEstHndl_t m_puschRxChEstHndl;

        cuphy::kernelDescrs<N_CH_EST_DESCR_TYPES> m_kernelStatDescr;
        cuphy::kernelDescrs<N_CH_EST_DESCR_TYPES> m_kernelDynDescr;

        cuphyPuschRxUeGrpPrms_t* m_puschRxUeGrpPrmsCpu;
        cuphyPuschRxUeGrpPrms_t* m_puschRxUeGrpPrmsGpu;

        // Filter tensors.
        cuphy::tensor_device m_WFreq;
        cuphy::tensor_device m_WFreq4;
        cuphy::tensor_device m_WFreqSmall;
        cuphy::tensor_device m_ShiftSeq;
        cuphy::tensor_device m_ShiftSeq4;
        cuphy::tensor_device m_UnshiftSeq;
        cuphy::tensor_device m_UnshiftSeq4;

        cuphyTensorPrm_t m_tWFreq;
        cuphyTensorPrm_t m_tWFreq4;
        cuphyTensorPrm_t m_tWFreqSmall;
        cuphyTensorPrm_t m_tShiftSeq;
        cuphyTensorPrm_t m_tShiftSeq4;
        cuphyTensorPrm_t m_tUnshiftSeq;
        cuphyTensorPrm_t m_tUnshiftSeq4;

};


void ChannelEstimator::setFilters(const py::dict& chEstFilters) {
    m_WFreq = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(chEstFilters["WFreq"],
                                                                                CUPHY_R_32F,
                                                                                CUPHY_R_32F,
                                                                                cuphy::tensor_flags::align_tight,
                                                                                m_chEstPrms.cuStream);
    m_tWFreq.desc = m_WFreq.desc().handle();
    m_tWFreq.pAddr = m_WFreq.addr();

    m_WFreq4 = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(chEstFilters["WFreq4"],
                                                                                 CUPHY_R_32F,
                                                                                 CUPHY_R_32F,
                                                                                 cuphy::tensor_flags::align_tight,
                                                                                 m_chEstPrms.cuStream);
    m_tWFreq4.desc = m_WFreq4.desc().handle();
    m_tWFreq4.pAddr = m_WFreq4.addr();


    m_WFreqSmall = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(chEstFilters["WFreqSmall"],
                                                                                     CUPHY_R_32F,
                                                                                     CUPHY_R_32F,
                                                                                     cuphy::tensor_flags::align_tight,
                                                                                     m_chEstPrms.cuStream);
    m_tWFreqSmall.desc = m_WFreqSmall.desc().handle();
    m_tWFreqSmall.pAddr = m_WFreqSmall.addr();


    m_ShiftSeq = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(chEstFilters["ShiftSeq"],
                                                                                                 CUPHY_C_32F,
                                                                                                 CUPHY_C_16F,
                                                                                                cuphy::tensor_flags::align_tight,
                                                                                                 m_chEstPrms.cuStream);
    m_tShiftSeq.desc = m_ShiftSeq.desc().handle();
    m_tShiftSeq.pAddr = m_ShiftSeq.addr();

    m_ShiftSeq4 = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(chEstFilters["ShiftSeq4"],
                                                                                                  CUPHY_C_32F,
                                                                                                  CUPHY_C_16F,
                                                                                                  cuphy::tensor_flags::align_tight,
                                                                                                  m_chEstPrms.cuStream);
    m_tShiftSeq4.desc = m_ShiftSeq4.desc().handle();
    m_tShiftSeq4.pAddr = m_ShiftSeq4.addr();

    m_UnshiftSeq = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(chEstFilters["UnShiftSeq"],
                                                                                                   CUPHY_C_32F,
                                                                                                   CUPHY_C_16F,
                                                                                                   cuphy::tensor_flags::align_tight,
                                                                                                   m_chEstPrms.cuStream);
    m_tUnshiftSeq.desc = m_UnshiftSeq.desc().handle();
    m_tUnshiftSeq.pAddr = m_UnshiftSeq.addr();

    m_UnshiftSeq4 = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(chEstFilters["UnShiftSeq4"],
                                                                                                    CUPHY_C_32F,
                                                                                                    CUPHY_C_16F,
                                                                                                    cuphy::tensor_flags::align_tight,
                                                                                                    m_chEstPrms.cuStream);
    m_tUnshiftSeq4.desc = m_UnshiftSeq4.desc().handle();
    m_tUnshiftSeq4.pAddr = m_UnshiftSeq4.addr();
}


void ChannelEstimator::allocateDescr() {

    std::array<size_t, N_CH_EST_DESCR_TYPES> statDescrSizeBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> statDescrAlignBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> dynDescrSizeBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> dynDescrAlignBytes{};

    size_t* pStatDescrSizeBytes  = statDescrSizeBytes.data();
    size_t* pStatDescrAlignBytes = statDescrAlignBytes.data();
    size_t* pDynDescrSizeBytes   = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes  = dynDescrAlignBytes.data();

    cuphyStatus_t status = cuphyPuschRxChEstGetDescrInfo(&pStatDescrSizeBytes[CH_EST],
                                                         &pStatDescrAlignBytes[CH_EST],
                                                         &pDynDescrSizeBytes[CH_EST],
                                                         &pDynDescrAlignBytes[CH_EST]);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxChEstGetDescrInfo()");
    }

    for(uint32_t chEstTimeIdx = 1; chEstTimeIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeIdx) {
        pStatDescrSizeBytes[CH_EST + chEstTimeIdx]  = pStatDescrSizeBytes[CH_EST];
        pStatDescrAlignBytes[CH_EST + chEstTimeIdx] = pStatDescrAlignBytes[CH_EST];
        pDynDescrSizeBytes[CH_EST + chEstTimeIdx]   = pDynDescrSizeBytes[CH_EST];
        pDynDescrAlignBytes[CH_EST + chEstTimeIdx]  = pDynDescrAlignBytes[CH_EST];
    }

    // Note: Supporting only one user group at the moment with the Python bindings though
    // memory gets allocated here.
    pDynDescrSizeBytes[CH_EST_PARAMS]  = sizeof(cuphyPuschRxUeGrpPrms_t) * MAX_N_USER_GROUPS_SUPPORTED;
    pDynDescrAlignBytes[CH_EST_PARAMS] = alignof(cuphyPuschRxUeGrpPrms_t);

    // Allocate descriptors (CPU and GPU).
    m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes);
    m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes);

}

ChannelEstimator::ChannelEstimator(
    const py::dict& chEstFilters,
    const uint64_t inputDevicePtr,
    const uint64_t outputChannelHostPtr,
    const uint64_t outputChannelDevicePtr,
    const uint64_t debugDevicePtr,
    uint64_t cuStream
):
    m_kernelStatDescr("ChEstStatDescr"),
    m_kernelDynDescr("ChEstDynDescr")
{
    // Store pointers to pre-allocated mem.
    m_chEstPrms.inputDevicePtr = (void *)inputDevicePtr;
    m_chEstPrms.outputChannelHostPtr = (void *)outputChannelHostPtr;
    m_chEstPrms.outputChannelDevicePtr = (void *)outputChannelDevicePtr;
    m_chEstPrms.debugDevicePtr = (void *)debugDevicePtr;
    m_chEstPrms.cuStream = (cudaStream_t)cuStream;

    // Allocate descriptors.
    allocateDescr();

    // Set channel estimation filters. Move them to device memory.
    setFilters(chEstFilters);

    // Create the channel estimator object.
    auto statCpuDescrStartAddrs = m_kernelStatDescr.getCpuStartAddrs();
    auto statGpuDescrStartAddrs = m_kernelStatDescr.getGpuStartAddrs();
    bool enableCpuToGpuDescrAsyncCpy = true;

    cuphyStatus_t status = cuphyCreatePuschRxChEst(&m_puschRxChEstHndl,
                                                   &m_tWFreq,
                                                   &m_tWFreq4,
                                                   &m_tWFreqSmall,
                                                   &m_tShiftSeq,
                                                   &m_tShiftSeq4,
                                                   &m_tUnshiftSeq,
                                                   &m_tUnshiftSeq4,
                                                   nullptr, //!!FixMe to support early harq setup
                                                   enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                   reinterpret_cast<void**>(&statCpuDescrStartAddrs[CH_EST]),
                                                   reinterpret_cast<void**>(&statGpuDescrStartAddrs[CH_EST]),
                                                   m_chEstPrms.cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxChEst()");
    }
}


ChannelEstimator::~ChannelEstimator() {
    destroy();
}


void ChannelEstimator::setChEstParams(const uint8_t nUes,
                                      const uint16_t numRxAnt,
                                      const uint16_t slotNum,
                                      const uint16_t startPrb,
                                      const uint16_t numPrb,
                                      const uint8_t startSym,
                                      const uint8_t numSym,
                                      const uint16_t dmrsSymLocBmsk,
                                      const uint16_t dmrsScrmId,
                                      const uint8_t dmrsMaxLen,
                                      const uint8_t dmrsAddlnPos,
                                      const uint8_t numDmrsCdmGrpsNoData,
                                      const uint8_t scid,
                                      const std::vector<uint8_t>& numLayers,
                                      const std::vector<uint16_t>& dmrsPortBmsks) {
    // Set channel estimation parameters. These are the UE group common parameters:
    m_puschRxUeGrpPrmsCpu->nUes = nUes;
    m_puschRxUeGrpPrmsCpu->nRxAnt = numRxAnt;
    m_puschRxUeGrpPrmsCpu->slotNum = slotNum;
    m_puschRxUeGrpPrmsCpu->startPrb = startPrb;
    m_puschRxUeGrpPrmsCpu->nPrb = numPrb;
    m_puschRxUeGrpPrmsCpu->puschStartSym = startSym;
    m_puschRxUeGrpPrmsCpu->nPuschSym = numSym;
    m_puschRxUeGrpPrmsCpu->nDmrsSyms = static_cast<uint32_t>((1 + dmrsAddlnPos) * dmrsMaxLen);
    m_puschRxUeGrpPrmsCpu->dmrsSymLocBmsk = dmrsSymLocBmsk;
    m_puschRxUeGrpPrmsCpu->dmrsScrmId = dmrsScrmId;
    m_puschRxUeGrpPrmsCpu->dmrsMaxLen = dmrsMaxLen;
    m_puschRxUeGrpPrmsCpu->dmrsAddlnPos = dmrsAddlnPos;
    m_puschRxUeGrpPrmsCpu->nDmrsCdmGrpsNoData = numDmrsCdmGrpsNoData;
    m_puschRxUeGrpPrmsCpu->nTimeChEsts = dmrsAddlnPos + 1;

    uint16_t groupDmrsPortBmsk = 0;
    static constexpr uint16_t GRP_DMRS_PORT_MSK = (1U << 12) - 1; // DMRS port info in bit 0 to bit 11
    uint8_t arrDmrsPorts[MAX_N_LAYERS_PUSCH][2];
    uint8_t layerIdx      = 0;
    uint8_t ueGrpLayerIdx = 0;

    // Set scid
    m_puschRxUeGrpPrmsCpu->scid = scid;

    uint8_t dmrsCnt = 0;
    for(uint8_t i = startSym; i < (startSym + numSym); ++i) {
        if(1 & (dmrsSymLocBmsk >> i)) {
            m_puschRxUeGrpPrmsCpu->dmrsSymLoc[dmrsCnt++] = i;
        }
    }
    m_puschRxUeGrpPrmsCpu->dmrsCnt = dmrsCnt;

    // Initialize number of layers (it is a sum over UEs).
    m_puschRxUeGrpPrmsCpu->nLayers = 0;

    // Set UE-specific parameters.
    for(uint8_t ueIdx = 0; ueIdx < nUes; ueIdx++) {
        m_puschRxUeGrpPrmsCpu->ueIdxs[ueIdx] = ueIdx;
        m_puschRxUeGrpPrmsCpu->nUeLayers[ueIdx] = numLayers[ueIdx];
        m_puschRxUeGrpPrmsCpu->nLayers += numLayers[ueIdx];

        // Derive active DMRS grids.
        uint16_t dmrsPortBmsk = dmrsPortBmsks[ueIdx] & GRP_DMRS_PORT_MSK;

        groupDmrsPortBmsk = groupDmrsPortBmsk | dmrsPortBmsk;

        for(uint16_t k = 0; k < 8; k++) {
            if((dmrsPortBmsk >> k) & 0x1) {
                arrDmrsPorts[layerIdx][0] = k;
                arrDmrsPorts[layerIdx][1] = k & 0x2;
                m_puschRxUeGrpPrmsCpu->dmrsPortIdxs[layerIdx] = k;
                layerIdx++;
            }
        }

        for(int j = 0; j < m_puschRxUeGrpPrmsCpu->nUeLayers[ueIdx]; ++j) {
            m_puschRxUeGrpPrmsCpu->ueGrpLayerToUeIdx[ueGrpLayerIdx] = ueIdx;
            ueGrpLayerIdx++;
        }
    }

    // Finish derivation of active DMRS grids.
    uint32_t gridBmsk0 = static_cast<uint32_t>(((0x33 & groupDmrsPortBmsk) != 0) ? 1 : 0);
    uint32_t gridBmsk1 = static_cast<uint32_t>(((0xCC & groupDmrsPortBmsk) != 0) ? 1 : 0);

    m_puschRxUeGrpPrmsCpu->nDmrsGridsPerPrb = 2;
    m_puschRxUeGrpPrmsCpu->activeDMRSGridBmsk = gridBmsk0 | (gridBmsk1 << 1);

    // Derive active TOCCs and FOCCs.
    uint32_t ToccBmsk0_0 = static_cast<uint32_t>(((0x03 & groupDmrsPortBmsk) != 0) ? 1 : 0);
    uint32_t ToccBmsk0_1 = static_cast<uint32_t>(((0x30 & groupDmrsPortBmsk) != 0) ? 1 : 0);
    uint32_t ToccBmsk1_0 = static_cast<uint32_t>(((0x0C & groupDmrsPortBmsk) != 0) ? 1 : 0);
    uint32_t ToccBmsk1_1 = static_cast<uint32_t>(((0xC0 & groupDmrsPortBmsk) != 0) ? 1 : 0);

    uint32_t FoccBmsk0_0 = static_cast<uint32_t>(((0x11 & groupDmrsPortBmsk) != 0) ? 1 : 0);
    uint32_t FoccBmsk0_1 = static_cast<uint32_t>(((0x22 & groupDmrsPortBmsk) != 0) ? 1 : 0);
    uint32_t FoccBmsk1_0 = static_cast<uint32_t>(((0x44 & groupDmrsPortBmsk) != 0) ? 1 : 0);
    uint32_t FoccBmsk1_1 = static_cast<uint32_t>(((0x88 & groupDmrsPortBmsk) != 0) ? 1 : 0);

    m_puschRxUeGrpPrmsCpu->activeTOCCBmsk[0] = ToccBmsk0_0 | (ToccBmsk0_1 << 1);
    m_puschRxUeGrpPrmsCpu->activeTOCCBmsk[1] = ToccBmsk1_0 | (ToccBmsk1_1 << 1);
    m_puschRxUeGrpPrmsCpu->activeFOCCBmsk[0] = FoccBmsk0_0 | (FoccBmsk0_1 << 1);
    m_puschRxUeGrpPrmsCpu->activeFOCCBmsk[1] = FoccBmsk1_0 | (FoccBmsk1_1 << 1);

    uint32_t sumToccBmsk0 = ToccBmsk0_0 + ToccBmsk0_1;
    uint32_t sumToccBmsk1 = ToccBmsk1_0 + ToccBmsk1_1;
    uint32_t sumFoccBmsk0 = FoccBmsk0_0 + FoccBmsk0_1;
    uint32_t sumFoccBmsk1 = FoccBmsk1_0 + FoccBmsk1_1;

    for(int i = 0; i < m_puschRxUeGrpPrmsCpu->nLayers; i++)
    {
        if(arrDmrsPorts[i][1] == 0) { // First grid
            if(sumToccBmsk0 == 1) {
                if(sumFoccBmsk0 == 1) {
                    m_puschRxUeGrpPrmsCpu->OCCIdx[i] = 0 + (arrDmrsPorts[i][1] << 1);
                }
                else { // sumFoccBmsk0 == 2
                    m_puschRxUeGrpPrmsCpu->OCCIdx[i] = (arrDmrsPorts[i][0] & 0x1) + (arrDmrsPorts[i][1] << 1);
                }
            }
            else { // sumToccBmsk0 == 2
                if(sumFoccBmsk0 == 1) {
                    m_puschRxUeGrpPrmsCpu->OCCIdx[i] = (arrDmrsPorts[i][0] < 4 ? 0 : 1) + (arrDmrsPorts[i][1] << 1);
                }
                else { // sumFoccBmsk0 == 2
                    m_puschRxUeGrpPrmsCpu->OCCIdx[i] = ((arrDmrsPorts[i][0] >> 1) & 0x2) + (arrDmrsPorts[i][0] & 0x1) + (arrDmrsPorts[i][1] << 1);
                }
            }
        }
        else { // Second grid
            if(sumToccBmsk1 == 1) {
                if(sumFoccBmsk1 == 1) {
                    m_puschRxUeGrpPrmsCpu->OCCIdx[i] = 0 + (arrDmrsPorts[i][1] << 1);
                }
                else {
                    m_puschRxUeGrpPrmsCpu->OCCIdx[i] = (arrDmrsPorts[i][0] & 0x1) + (arrDmrsPorts[i][1] << 1);
                }
            }
            else {
                if(sumFoccBmsk1 == 1) {
                    m_puschRxUeGrpPrmsCpu->OCCIdx[i] = (arrDmrsPorts[i][0] < 4 ? 0 : 1) + (arrDmrsPorts[i][1] << 1);
                }
                else
                {
                    m_puschRxUeGrpPrmsCpu->OCCIdx[i] = ((arrDmrsPorts[i][0] >> 1) & 0x2) + (arrDmrsPorts[i][0] & 0x1) + (arrDmrsPorts[i][1] << 1);
                }
            }
        }
    }
}


py::array_t<std::complex<float>> ChannelEstimator::estimate(const py::array& inputData,
                                                            const uint8_t nUes,
                                                            const uint16_t numRxAnt,
                                                            const uint16_t slotNum,
                                                            const uint16_t startPrb,
                                                            const uint16_t numPrb,
                                                            const uint8_t startSym,
                                                            const uint8_t numSym,
                                                            const uint16_t dmrsSymLocBmsk,
                                                            const uint16_t dmrsScrmId,
                                                            const uint8_t dmrsMaxLen,
                                                            const uint8_t dmrsAddlnPos,
                                                            const uint8_t numDmrsCdmGrpsNoData,
                                                            const uint8_t scid,
                                                            const std::vector<uint8_t>& numLayers,
                                                            const std::vector<uint16_t>& dmrsPortBmsks) {
    const uint16_t nUeGrps = 1;  // Only one supported for now.

    // Read input data into device memory.
    cuphy::tensor_device deviceRxDataTensor = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(inputData,
                                                                                   m_chEstPrms.inputDevicePtr,
                                                                                   CUPHY_C_32F,
                                                                                   CUPHY_C_16F,
                                                                                   cuphy::tensor_flags::align_tight,
                                                                                   m_chEstPrms.cuStream);

    // Set channel estimation parameters.
    auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();
    m_puschRxUeGrpPrmsCpu = (cuphyPuschRxUeGrpPrms_t*)(dynCpuDescrStartAddrs[CH_EST_PARAMS]);
    m_puschRxUeGrpPrmsGpu = (cuphyPuschRxUeGrpPrms_t*)(dynGpuDescrStartAddrs[CH_EST_PARAMS]);

    setChEstParams(nUes,
                   numRxAnt,
                   slotNum,
                   startPrb,
                   numPrb,
                   startSym,
                   numSym,
                   dmrsSymLocBmsk,
                   dmrsScrmId,
                   dmrsMaxLen,
                   dmrsAddlnPos,
                   numDmrsCdmGrpsNoData,
                   scid,
                   numLayers,
                   dmrsPortBmsks);

    // Allocate output tensor arrays in device memory.
    cuphy::tensor_device deviceChannelEst = cuphy::tensor_device(m_chEstPrms.outputChannelDevicePtr,
                                                                 CUPHY_C_16F,
                                                                 m_puschRxUeGrpPrmsCpu->nRxAnt,
                                                                 m_puschRxUeGrpPrmsCpu->nLayers,
                                                                 CUPHY_N_TONES_PER_PRB * numPrb,
                                                                 m_puschRxUeGrpPrmsCpu->nTimeChEsts,
                                                                 cuphy::tensor_flags::align_tight);
    cuphy::tensor_device debugTensor = cuphy::tensor_device(m_chEstPrms.debugDevicePtr,
                                                            CUPHY_C_32F,
                                                            CUPHY_N_TONES_PER_PRB * numPrb / 2,
                                                            m_puschRxUeGrpPrmsCpu->nDmrsSyms,
                                                            1,
                                                            1,
                                                            cuphy::tensor_flags::align_tight);

    m_puschRxUeGrpPrmsCpu->tInfoDataRx.pAddr = m_chEstPrms.inputDevicePtr;
    m_puschRxUeGrpPrmsCpu->tInfoDataRx.elemType = CUPHY_C_16F;
    for(int i = 0; i < 3; i++) {
        m_puschRxUeGrpPrmsCpu->tInfoDataRx.strides[i] = deviceRxDataTensor.strides()[i];
    }
    m_puschRxUeGrpPrmsCpu->tInfoHEst.pAddr = m_chEstPrms.outputChannelDevicePtr;
    m_puschRxUeGrpPrmsCpu->tInfoHEst.elemType = CUPHY_C_16F;
    for(int i = 0; i < 4; i++) {
        m_puschRxUeGrpPrmsCpu->tInfoHEst.strides[i] = deviceChannelEst.strides()[i];
    }

    m_puschRxUeGrpPrmsCpu->tInfoChEstDbg.pAddr = m_chEstPrms.debugDevicePtr;
    m_puschRxUeGrpPrmsCpu->tInfoChEstDbg.elemType = CUPHY_C_32F;
    for(int i = 0; i < 3; i++) {
        m_puschRxUeGrpPrmsCpu->tInfoChEstDbg.strides[i] = debugTensor.strides()[i];
    }

    // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
    cuphyPuschRxChEstLaunchCfgs_t chEstLaunchCfgs[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST];
    for(int32_t chEstTimeInst = 0; chEstTimeInst < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInst) {
        chEstLaunchCfgs[chEstTimeInst].nCfgs = 0;
    }

    // Run setup.
    uint8_t enableDftSOfdm = 0;
    bool enableCpuToGpuDescrAsyncCpy = false;
    uint8_t* pPreEarlyHarqWaitKernelStatus_d;
    uint8_t* pPostEarlyHarqWaitKernelStatus_d;
    cuphyStatus_t chEstSetupStatus = cuphySetupPuschRxChEst(m_puschRxChEstHndl,
                                                            m_puschRxUeGrpPrmsCpu,
                                                            m_puschRxUeGrpPrmsGpu,
                                                            nUeGrps,
                                                            enableDftSOfdm,
                                                            pPreEarlyHarqWaitKernelStatus_d,
                                                            pPostEarlyHarqWaitKernelStatus_d,
                                                            1000,  
                                                            1500, 
                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                            reinterpret_cast<void**>(&dynCpuDescrStartAddrs[CH_EST]),
                                                            reinterpret_cast<void**>(&dynGpuDescrStartAddrs[CH_EST]),
                                                            chEstLaunchCfgs,
                                                            0,
                                                            nullptr,    //FixMe (may be) add support for early HARQ sub-slot processing
                                                            nullptr,    //FixMe (may be) add support for early HARQ sub-slot processing
                                                            m_chEstPrms.cuStream);
    if(chEstSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(chEstSetupStatus, "cuphySetupPuschRxChEst()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        m_kernelDynDescr.asyncCpuToGpuCpy(m_chEstPrms.cuStream);
    }
    cudaStreamSynchronize(m_chEstPrms.cuStream);

    // Launch kernel using the CUDA driver API.
    for(uint32_t chEstInstIdx = 0; chEstInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstInstIdx) {
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < chEstLaunchCfgs[chEstInstIdx].nCfgs; ++hetCfgIdx) {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = chEstLaunchCfgs[chEstInstIdx].cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                              kernelNodeParamsDriver.gridDimX,
                                              kernelNodeParamsDriver.gridDimY,
                                              kernelNodeParamsDriver.gridDimZ,
                                              kernelNodeParamsDriver.blockDimX,
                                              kernelNodeParamsDriver.blockDimY,
                                              kernelNodeParamsDriver.blockDimZ,
                                              kernelNodeParamsDriver.sharedMemBytes,
                                              static_cast<CUstream>(m_chEstPrms.cuStream),
                                              kernelNodeParamsDriver.kernelParams,
                                              kernelNodeParamsDriver.extra));
        }
    }
    cudaStreamSynchronize(m_chEstPrms.cuStream);

    // Output channel to host memory.
    uint32_t dim0 = m_puschRxUeGrpPrmsCpu->nRxAnt;
    uint32_t dim1 = m_puschRxUeGrpPrmsCpu->nLayers;
    uint32_t dim2 = CUPHY_N_TONES_PER_PRB * numPrb;
    uint32_t dim3 = m_puschRxUeGrpPrmsCpu->nTimeChEsts;
    cuphy::tensor_pinned hostChannelEst = cuphy::tensor_pinned(m_chEstPrms.outputChannelHostPtr,
                                                               CUPHY_C_32F,
                                                               dim0,
                                                               dim1,
                                                               dim2,
                                                               dim3,
                                                               cuphy::tensor_flags::align_tight);
    hostChannelEst.convert(deviceChannelEst, m_chEstPrms.cuStream);
    cudaStreamSynchronize(m_chEstPrms.cuStream);

    // Return the Numpy array.
    return hostToNumpy<std::complex<float>>((std::complex<float>*)m_chEstPrms.outputChannelHostPtr,
                                            dim0, dim1, dim2, dim3);
}


void ChannelEstimator::destroy() {
    // Destroy the PUSCH channel estimation handle.
    cuphyStatus_t status = cuphyDestroyPuschRxChEst(m_puschRxChEstHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxChEst()");
    }
}

} // namespace pycuphy


#endif // PYCUPHY_CHANNEL_EST_CPP
