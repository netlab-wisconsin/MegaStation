/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>
#include <queue>
#include <condition_variable>
#include <bitset>
#include <map>

#include "util.hpp"
#include "pusch_rx.hpp"
#include "cuphy_channels.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


using namespace cuphy;
using namespace std::complex_literals;
namespace py = pybind11;


#ifndef AERIAL_PYTHON_PUSCH_CPP
#define AERIAL_PYTHON_PUSCH_CPP

namespace pycuphy {

class PuschPipeline {

    public:

    PuschPipeline():
    m_LinearAlloc(20000000)
    {}

    uint64_t createPuschRx(const py::object& statPrms, uint64_t cuStream);
    void setupPuschRx(uint64_t puschHandle, const py::object& dynPrms);
    void runPuschRx(uint64_t puschHandle, uint64_t cuStream);
    void destroyPuschRx(uint64_t puschHandle);
    void writeDbgBufSynch(uint64_t puschHandle, uint64_t cuStream);

    template <typename T>
    cuphy::tensor_ref tensorFromNumpy(const py::array& py_array,
                                      cuphyDataType_t convertToType,
                                      cuphy::tensor_flags tensorDescFlags,
                                      cudaStream_t strm);


    private:

    cuphyPuschStatPrms_t            puschStatPrms;
    cuphyTracker_t                  puschTracker;
    cuphyPuschDynPrms_t             puschDynParams;
    cuphyPuschCellGrpDynPrm_t       cellGrpDynParams;
    cuphyPuschDataIn_t              puschDataIn;
    std::vector<cuphyTensorPrm_t>   tDataRx;
    cuphyPuschStatusOut_t           puschStatusOutput;

    std::vector<cuphy::tensor_ref> tRefTDataRx;

    tensor_device WFreqTensor;
    tensor_device WFreq4Tensor;
    tensor_device WFreqSmallTensor;

    tensor_device shiftSeqTensor;
    tensor_device unshiftSeqTensor;
    tensor_device shiftSeq4Tensor;
    tensor_device unshiftSeq4Tensor;

    cuphyPuschDataOut_t             puschDataOut;
    cuphyPuschDataInOut_t           puschDataInOut;

    cuphy::buffer<uint8_t*, cuphy::pinned_alloc> bHarqBufferPtrs    = std::move(cuphy::buffer<uint8_t*, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_LinearAlloc;

    std::string debug_file_name;
    std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
};


template <typename T>
inline T* pointerFromNumpyArray(const py::array& py_array) {
    py::array_t<T, py::array::f_style | py::array::forcecast> array = py_array;
    py::buffer_info buf                 = array.request();
    T* ptr                              = static_cast<T*>(buf.ptr);
    return ptr;
}


template <typename T>
inline cuphy::tensor_ref PuschPipeline::tensorFromNumpy(const py::array& py_array,
                                                        cuphyDataType_t convertToType,
                                                        cuphy::tensor_flags tensorDescFlags,
                                                        cudaStream_t strm) {
    cudaError_t e = cudaSuccess;

    py::array_t<T, py::array::f_style | py::array::forcecast> array = py_array;
    py::buffer_info buf = array.request();
    cuphy::tensor_ref tRefArray;

    // Create a tensor info with the requested conversion type, but the original layout
    int rank = buf.ndim;
    switch (rank)
    {
    case 1:
        tRefArray.desc().set(convertToType, buf.shape[0], tensorDescFlags);
        break;
    case 2:
        tRefArray.desc().set(convertToType, buf.shape[0], buf.shape[1], tensorDescFlags);
        break;
    case 3:
        tRefArray.desc().set(convertToType, buf.shape[0], buf.shape[1], buf.shape[2], tensorDescFlags);
        break;
    case 4:
        tRefArray.desc().set(convertToType, buf.shape[0], buf.shape[1], buf.shape[2], buf.shape[3],    tensorDescFlags);
        break;

    default:
        throw std::runtime_error("tensorFromNumpy() dimension error !");
        break;
    }

    // TO DO: may optimize for less copy
    if (convertToType == CUPHY_C_16F) {
        m_LinearAlloc.alloc(tRefArray);

        tensor_pinned input_tensor;
        tensor_device new_tensor;

        switch (rank)
        {
        case 1:
            input_tensor = tensor_device(buf.ptr, CUPHY_C_32F,
                                                    buf.shape[0],
                                                    cuphy::tensor_flags::align_tight);
            new_tensor = tensor_device(tRefArray.addr(), CUPHY_C_16F,
                                                    buf.shape[0],
                                                    cuphy::tensor_flags::align_tight);
            break;
        case 2:
            input_tensor = tensor_device(buf.ptr, CUPHY_C_32F,
                                                    buf.shape[0],
                                                    buf.shape[1],
                                                    cuphy::tensor_flags::align_tight);
            new_tensor = tensor_device(tRefArray.addr(), CUPHY_C_16F,
                                                    buf.shape[0],
                                                    buf.shape[1],
                                                    cuphy::tensor_flags::align_tight);
            break;
        case 3:
            input_tensor = tensor_device(buf.ptr, CUPHY_C_32F,
                                                    buf.shape[0],
                                                    buf.shape[1],
                                                    buf.shape[2],
                                                    cuphy::tensor_flags::align_tight);
            new_tensor = tensor_device(tRefArray.addr(), CUPHY_C_16F,
                                                    buf.shape[0],
                                                    buf.shape[1],
                                                    buf.shape[2],
                                                    cuphy::tensor_flags::align_tight);
            break;
        case 4:
            input_tensor = tensor_device(buf.ptr, CUPHY_C_32F,
                                                    buf.shape[0],
                                                    buf.shape[1],
                                                    buf.shape[2],
                                                    buf.shape[3],
                                                    cuphy::tensor_flags::align_tight);
            new_tensor = tensor_device(tRefArray.addr(), CUPHY_C_16F,
                                                    buf.shape[0],
                                                    buf.shape[1],
                                                    buf.shape[2],
                                                    buf.shape[3],
                                                    cuphy::tensor_flags::align_tight);
            break;
        }

        new_tensor.convert(input_tensor, strm);
        CUDA_CHECK(cudaStreamSynchronize(strm));

    }
    else {

        m_LinearAlloc.alloc(tRefArray);

        e = cudaMemcpyAsync(tRefArray.addr(),
                                            buf.ptr,
                                            tRefArray.desc().get_size_in_bytes(),
                                            cudaMemcpyDefault,
                                            strm);
    }

    CUDA_CHECK(cudaStreamSynchronize(strm));

    if(e != cudaSuccess)
    {
        throw cuda_exception(e);
    }

    return tRefArray;
}


template <typename T>
inline tensor_device statPrmsTensorFromNumpy(const py::array& py_array,
                                             cuphyDataType_t  convertToType,
                                             cuphy::tensor_flags tensorDescFlags = tensor_flags::align_default,
                                             cudaStream_t cuStream = 0) {
    py::array_t<T, py::array::f_style | py::array::forcecast> array = py_array;
    py::buffer_info buf                 = array.request();

    cuphy::tensor_pinned h_array_tensor;
    cuphy::tensor_device d_array_tensor;

    cuphyDataType_t  tempType = convertToType;
    if (convertToType == CUPHY_C_16F)
        tempType = CUPHY_C_32F;

    int rank = buf.ndim;
    switch (rank)
    {

    case 2:
        h_array_tensor  = tensor_device(buf.ptr, tempType, buf.shape[0], buf.shape[1], cuphy::tensor_flags::align_tight);
        d_array_tensor  = tensor_device(convertToType, buf.shape[0], buf.shape[1], tensorDescFlags);
        break;

    case 3:
        h_array_tensor  = tensor_device(buf.ptr, tempType, buf.shape[0], buf.shape[1], buf.shape[2], cuphy::tensor_flags::align_tight);
        d_array_tensor  = tensor_device(convertToType, buf.shape[0], buf.shape[1], buf.shape[2], tensorDescFlags);
        break;

    default:
        throw std::runtime_error("Dimesion error for channel estimation filters or sequences!");
        break;
    }

    d_array_tensor.convert(h_array_tensor);  // checked
    CUDA_CHECK(cudaStreamSynchronize(0));

    return d_array_tensor;
}


uint64_t PuschPipeline::createPuschRx(const py::object& statPrms, uint64_t cuStream) {

    puschStatPrms.enableCfoCorrection     = statPrms.attr("enableCfoCorrection").cast<uint8_t>();
    puschStatPrms.enablePuschTdi          = statPrms.attr("enablePuschTdi").cast<uint8_t>();
    puschStatPrms.enableDftSOfdm          = statPrms.attr("enableDftSOfdm").cast<uint8_t>();
    puschStatPrms.enableTbSizeCheck       = statPrms.attr("enableTbSizeCheck").cast<uint8_t>();
    puschStatPrms.stream_priority         = statPrms.attr("stream_priority").cast<int>();
    puschStatPrms.ldpcnIterations         = statPrms.attr("ldpcnIterations").cast<uint8_t>();
    puschStatPrms.ldpcEarlyTermination    = statPrms.attr("ldpcEarlyTermination").cast<uint8_t>();
    puschStatPrms.ldpcUseHalf             = statPrms.attr("ldpcUseHalf").cast<uint8_t>();
    puschStatPrms.ldpcAlgoIndex           = statPrms.attr("ldpcAlgoIndex").cast<uint8_t>();
    puschStatPrms.ldpcFlags               = statPrms.attr("ldpcFlags").cast<uint32_t>();
    puschStatPrms.ldpcKernelLaunch        = static_cast<cuphyPuschLdpcKernelLaunch_t>(statPrms.attr("ldpcKernelLaunch").cast<uint32_t>());
    puschStatPrms.enableRssiMeasurement   = statPrms.attr("enableRssiMeasurement").cast<uint8_t>();
    puschStatPrms.enableSinrMeasurement   = statPrms.attr("enableSinrMeasurement").cast<uint8_t>();
    puschStatPrms.nMaxCells               = statPrms.attr("nMaxCells").cast<uint16_t>();
    puschStatPrms.nMaxCellsPerSlot        = statPrms.attr("nMaxCellsPerSlot").cast<uint16_t>();
    puschStatPrms.nMaxTbs                 = statPrms.attr("nMaxTbs").cast<uint32_t>();
    puschStatPrms.nMaxCbsPerTb            = statPrms.attr("nMaxCbsPerTb").cast<uint32_t>();
    puschStatPrms.nMaxTotCbs              = statPrms.attr("nMaxTotCbs").cast<uint32_t>();
    puschStatPrms.nMaxRx                  = statPrms.attr("nMaxRx").cast<uint32_t>();
    puschStatPrms.nMaxPrb                 = statPrms.attr("nMaxPrb").cast<uint32_t>();
    puschStatPrms.polarDcdrListSz         = 8;
    puschStatPrms.eqCoeffAlgo             = static_cast<cuphyPuschEqCoefAlgoType_t>(py::tuple(statPrms.attr("eqCoeffAlgo").attr("value"))[0].cast<int>());

    puschTracker.pMemoryFootprint         = nullptr;
    puschStatPrms.pOutInfo                = &puschTracker;

    tDataRx.resize(puschStatPrms.nMaxCells);
    tRefTDataRx.resize(puschStatPrms.nMaxCells);

    puschStatPrms.pWFreq                  = new cuphyTensorPrm_t;
    puschStatPrms.pWFreq4                 = new cuphyTensorPrm_t;
    puschStatPrms.pWFreqSmall             = new cuphyTensorPrm_t;
    puschStatPrms.pShiftSeq               = new cuphyTensorPrm_t;
    puschStatPrms.pUnShiftSeq             = new cuphyTensorPrm_t;
    puschStatPrms.pShiftSeq4              = new cuphyTensorPrm_t;
    puschStatPrms.pUnShiftSeq4            = new cuphyTensorPrm_t;

    py::array WFreq                       = statPrms.attr("WFreq");
    py::array WFreq4                      = statPrms.attr("WFreq4");
    py::array WFreqSmall                  = statPrms.attr("WFreqSmall");
    py::array shiftSeq                    = statPrms.attr("ShiftSeq");
    py::array unshiftSeq                  = statPrms.attr("UnShiftSeq");
    py::array shiftSeq4                   = statPrms.attr("ShiftSeq4");
    py::array unshiftSeq4                 = statPrms.attr("UnShiftSeq4");

    WFreqTensor      = statPrmsTensorFromNumpy<float>(WFreq, CUPHY_R_32F, cuphy::tensor_flags::align_tight);
    WFreq4Tensor     = statPrmsTensorFromNumpy<float>(WFreq4, CUPHY_R_32F, cuphy::tensor_flags::align_tight);
    WFreqSmallTensor = statPrmsTensorFromNumpy<float>(WFreqSmall, CUPHY_R_32F, cuphy::tensor_flags::align_tight);

    puschStatPrms.pWFreq->desc = WFreqTensor.desc().handle();
    puschStatPrms.pWFreq->pAddr = WFreqTensor.addr();

    puschStatPrms.pWFreq4->desc = WFreq4Tensor.desc().handle();
    puschStatPrms.pWFreq4->pAddr = WFreq4Tensor.addr();

    puschStatPrms.pWFreqSmall->desc = WFreqSmallTensor.desc().handle();
    puschStatPrms.pWFreqSmall->pAddr = WFreqSmallTensor.addr();

    shiftSeqTensor      = statPrmsTensorFromNumpy<std::complex<float>>(shiftSeq, CUPHY_C_16F, cuphy::tensor_flags::align_tight);
    unshiftSeqTensor    = statPrmsTensorFromNumpy<std::complex<float>>(unshiftSeq, CUPHY_C_16F, cuphy::tensor_flags::align_tight);
    shiftSeq4Tensor     = statPrmsTensorFromNumpy<std::complex<float>>(shiftSeq4, CUPHY_C_16F, cuphy::tensor_flags::align_tight);
    unshiftSeq4Tensor   = statPrmsTensorFromNumpy<std::complex<float>>(unshiftSeq4, CUPHY_C_16F, cuphy::tensor_flags::align_tight);

    puschStatPrms.pShiftSeq->desc = shiftSeqTensor.desc().handle();
    puschStatPrms.pShiftSeq->pAddr = shiftSeqTensor.addr();

    puschStatPrms.pUnShiftSeq->desc = unshiftSeqTensor.desc().handle();
    puschStatPrms.pUnShiftSeq->pAddr = unshiftSeqTensor.addr();

    puschStatPrms.pShiftSeq4->desc = shiftSeq4Tensor.desc().handle();
    puschStatPrms.pShiftSeq4->pAddr = shiftSeq4Tensor.addr();

    puschStatPrms.pUnShiftSeq4->desc = unshiftSeq4Tensor.desc().handle();
    puschStatPrms.pUnShiftSeq4->pAddr = unshiftSeq4Tensor.addr();

    CUDA_CHECK_EXCEPTION(cudaEventCreate(&puschStatPrms.earlyHarqReadyEvent));
    puschStatPrms.enableEarlyHarq = 0;

    // CellStatPrms
    const py::list cellStatPrm = statPrms.attr("cellStatPrms");
    uint16_t nCells = cellStatPrm.size();

    cuphyCellStatPrm_t* pCellStatPrms   = new cuphyCellStatPrm_t[nCells];
    puschStatPrms.pCellStatPrms         = pCellStatPrms;

    for (int cell_id = 0; cell_id < nCells; cell_id ++ ) {

        const py::object cell_static_params = cellStatPrm[cell_id];

        pCellStatPrms[cell_id].phyCellId    = cell_static_params.attr("phyCellId").cast<uint16_t>();
        pCellStatPrms[cell_id].nRxAnt       = cell_static_params.attr("nRxAnt").cast<uint16_t>();
        pCellStatPrms[cell_id].nTxAnt       = cell_static_params.attr("nTxAnt").cast<uint16_t>();
        pCellStatPrms[cell_id].nPrbUlBwp    = cell_static_params.attr("nPrbUlBwp").cast<uint16_t>();
        pCellStatPrms[cell_id].nPrbDlBwp    = cell_static_params.attr("nPrbDlBwp").cast<uint16_t>();
        pCellStatPrms[cell_id].mu           = cell_static_params.attr("mu").cast<uint8_t>();
        if (pCellStatPrms[cell_id].mu > 1) {
            throw std::runtime_error("Unsupported numerology value!");
        }

        pCellStatPrms[cell_id].pPuschCellStatPrms = new cuphyPuschCellStatPrm_t;
        try {  // TODO: UCI is not supported
            const py::object puschCellStatPrms = cell_static_params.attr("puschCellStatPrms");

            pCellStatPrms[cell_id].pPuschCellStatPrms->nCsirsPorts      = puschCellStatPrms.attr("nCsirsPorts").cast<uint8_t>();
            pCellStatPrms[cell_id].pPuschCellStatPrms->N1               = puschCellStatPrms.attr("N1").cast<uint8_t>();
            pCellStatPrms[cell_id].pPuschCellStatPrms->N2               = puschCellStatPrms.attr("N2").cast<uint8_t>();
            pCellStatPrms[cell_id].pPuschCellStatPrms->csiReportingBand = puschCellStatPrms.attr("csiReportingBand").cast<uint8_t>();
            pCellStatPrms[cell_id].pPuschCellStatPrms->codebookType     = puschCellStatPrms.attr("codebookType").cast<uint8_t>();
            pCellStatPrms[cell_id].pPuschCellStatPrms->codebookMode     = puschCellStatPrms.attr("codebookMode").cast<uint8_t>();
            pCellStatPrms[cell_id].pPuschCellStatPrms->isCqi            = puschCellStatPrms.attr("isCqi").cast<uint8_t>();
            pCellStatPrms[cell_id].pPuschCellStatPrms->isLi             = puschCellStatPrms.attr("isLi").cast<uint8_t>();
        }
        catch(...) {
            pCellStatPrms[cell_id].pPuschCellStatPrms->nCsirsPorts      = 4;
            pCellStatPrms[cell_id].pPuschCellStatPrms->N1               = 2;
            pCellStatPrms[cell_id].pPuschCellStatPrms->N2               = 1;
            pCellStatPrms[cell_id].pPuschCellStatPrms->csiReportingBand = 0;
            pCellStatPrms[cell_id].pPuschCellStatPrms->codebookType     = 0;
            pCellStatPrms[cell_id].pPuschCellStatPrms->codebookMode     = 1;
            pCellStatPrms[cell_id].pPuschCellStatPrms->isCqi            = 0;
            pCellStatPrms[cell_id].pPuschCellStatPrms->isLi             = 0;
            pCellStatPrms[cell_id].pPucchCellStatPrms = nullptr;
        }
    }


    // Next, dbg prms.
    if (std::string(py::str(statPrms.attr("dbg"))) == "None") {
        puschStatPrms.pDbg = nullptr;
    }
    else {
        puschStatPrms.pDbg = new cuphyPuschStatDbgPrms_t;
        py::object dbg = statPrms.attr("dbg");
        puschStatPrms.pDbg->descrmOn = dbg.attr("descrmOn").cast<uint8_t>();
        puschStatPrms.pDbg->enableApiLogging = dbg.attr("enableApiLogging").cast<uint8_t>();

        if (std::string(py::str(dbg.attr("outFileName"))) == "None") {
            puschStatPrms.pDbg->pOutFileName = nullptr;
        }
        else {
            debug_file_name = std::string(py::str(dbg.attr("outFileName")));
            puschStatPrms.pDbg->pOutFileName = debug_file_name.c_str();
            debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(debug_file_name.c_str())));
        }
    }

    // Create pipeline.
    std::unique_ptr<cuphyPuschRxHndl_t> pPusch_handle = std::make_unique<cuphyPuschRxHndl_t>();
    cuphyStatus_t status = cuphyCreatePuschRx(pPusch_handle.get(), &puschStatPrms, (cudaStream_t)cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        std::cerr << "Error! cuphyCreatePuschRx(): " << cuphyGetErrorString(status) << std::endl;
        exit(1);
    }

    // Allocate dynamic memory.
    puschDynParams.pCellGrpDynPrm = &cellGrpDynParams;
    cellGrpDynParams.pCellPrms = new cuphyPuschCellDynPrm_t[nCells];
    cellGrpDynParams.pUeGrpPrms = new cuphyPuschUeGrpPrm_t[MAX_N_USER_GROUPS_SUPPORTED];

    for (int i = 0; i < MAX_N_USER_GROUPS_SUPPORTED; i++) {
        cellGrpDynParams.pUeGrpPrms[i].pUePrmIdxs = new uint16_t[CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP];
        cellGrpDynParams.pUeGrpPrms[i].pDmrsDynPrm = new cuphyPuschDmrsPrm_t;
    }

    cellGrpDynParams.pUePrms = new cuphyPuschUePrm_t[MAX_N_USER_GROUPS_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP];

    for (int i = 0; i < MAX_N_USER_GROUPS_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP; i++) {
        cellGrpDynParams.pUePrms[i].pUciPrms = nullptr;
        cellGrpDynParams.pUePrms[i].debug_d_derateCbsIndices = nullptr;
    }

    puschDynParams.pDbg = new cuphyPuschDynDbgPrms_t;

    puschDataIn.pTDataRx                  = &tDataRx[0];
    puschDynParams.pDataIn                = &puschDataIn;
    puschDynParams.pDataOut               = &puschDataOut;
    puschDynParams.pDataInOut             = &puschDataInOut;
    puschDataInOut.pHarqBuffersInOut      = bHarqBufferPtrs.addr(); // currently not returning harq to python
    //status parameters
    puschDynParams.pStatusOut             = &puschStatusOutput;

    return (uint64_t)(*pPusch_handle);
}


void PuschPipeline::setupPuschRx(uint64_t puschHandle, const py::object& dynPrms) {

    uint64_t cuStream                       = dynPrms.attr("cuStream").cast<uint64_t>();
    puschDynParams.cuStream                 = (cudaStream_t)cuStream;
    puschDynParams.procModeBmsk             = dynPrms.attr("procModeBmsk").cast<uint64_t>();
    puschDynParams.cpuCopyOn                = dynPrms.attr("cpuCopyOn").cast<uint8_t>();
    py::object dataIn                       = dynPrms.attr("dataIn");
    py::object dataOut                      = dynPrms.attr("dataOut");
    py::object dataInOut                    = dynPrms.attr("dataInOut");

    puschDynParams.setupPhase = static_cast<cuphyPuschSetupPhase_t>(py::tuple(dynPrms.attr("setupPhase").attr("value"))[0].cast<int>());


    if (puschDynParams.setupPhase == PUSCH_SETUP_PHASE_1) {

        m_LinearAlloc.reset();

        const py::object cellGrpDynPrm              = dynPrms.attr("cellGrpDynPrm");
        const py::list cellPrms                     = cellGrpDynPrm.attr("cellPrms");
        const py::list ueGrpPrms                    = cellGrpDynPrm.attr("ueGrpPrms");
        const py::list uePrms                       = cellGrpDynPrm.attr("uePrms");
        uint16_t nCells                             = cellPrms.size();

        cellGrpDynParams.nCells                     = nCells;
        for (int cell_id = 0; cell_id < nCells; cell_id ++ ) {
            const py::object cellDynParams = cellPrms[cell_id];

            cellGrpDynParams.pCellPrms[cell_id].cellPrmStatIdx  = cellDynParams.attr("cellPrmStatIdx").cast<uint16_t>();
            cellGrpDynParams.pCellPrms[cell_id].cellPrmDynIdx   = cellDynParams.attr("cellPrmDynIdx").cast<uint16_t>();
            cellGrpDynParams.pCellPrms[cell_id].slotNum         = cellDynParams.attr("slotNum").cast<uint16_t>();
        }

        // ueGrpPrms
        uint16_t nUeGrps                       = ueGrpPrms.size();
        cellGrpDynParams.nUeGrps               = nUeGrps;

        for (int ue_group_id = 0; ue_group_id < nUeGrps; ue_group_id++) {
            const py::object ue_grp_params = ueGrpPrms[ue_group_id];

            int cellPrmIdx = ue_grp_params.attr("cellPrmIdx").cast<int>();
            cellGrpDynParams.pUeGrpPrms[ue_group_id].pCellPrm = &cellGrpDynParams.pCellPrms[cellPrmIdx];
            cellGrpDynParams.pUeGrpPrms[ue_group_id].startPrb = ue_grp_params.attr("rbStart").cast<uint16_t>();
            cellGrpDynParams.pUeGrpPrms[ue_group_id].nPrb = ue_grp_params.attr("rbSize").cast<uint16_t>();
            cellGrpDynParams.pUeGrpPrms[ue_group_id].puschStartSym = ue_grp_params.attr("StartSymbolIndex").cast<uint8_t>();
            cellGrpDynParams.pUeGrpPrms[ue_group_id].nPuschSym = ue_grp_params.attr("NrOfSymbols").cast<uint8_t>();
            cellGrpDynParams.pUeGrpPrms[ue_group_id].dmrsSymLocBmsk = ue_grp_params.attr("dmrsSymLocBmsk").cast<uint16_t>();
            cellGrpDynParams.pUeGrpPrms[ue_group_id].rssiSymLocBmsk = ue_grp_params.attr("rssiSymLocBmsk").cast<uint16_t>();

            // pUePrmIdxs
            py::list uePrmIdxs                              = ue_grp_params.attr("uePrmIdxs");
            uint16_t nUes                                   = uePrmIdxs.size();
            cellGrpDynParams.pUeGrpPrms[ue_group_id].nUes   = nUes;

            for (int idx = 0; idx < nUes; idx++) {
                cellGrpDynParams.pUeGrpPrms[ue_group_id].pUePrmIdxs[idx] = uePrmIdxs[idx].cast<uint16_t>();
            }

            // DMRS
            cuphyPuschDmrsPrm_t* pDmrsDynPrm    = cellGrpDynParams.pUeGrpPrms[ue_group_id].pDmrsDynPrm;
            py::object dmrs_dyn_prm             = ue_grp_params.attr("dmrsDynPrm");
            pDmrsDynPrm->nDmrsCdmGrpsNoData     = dmrs_dyn_prm.attr("numDmrsCdmGrpsNoData").cast<uint8_t>();
            pDmrsDynPrm->dmrsScrmId             = dmrs_dyn_prm.attr("ulDmrsScramblingId").cast<uint8_t>();

            try {
                pDmrsDynPrm->dmrsAddlnPos = dmrs_dyn_prm.attr("dmrsAddlnPos").cast<uint8_t>();
                pDmrsDynPrm->dmrsMaxLen   = dmrs_dyn_prm.attr("dmrsMaxLen").cast<uint8_t>();
            }
            catch(...) {
                // TODO eventually these two fields will be removed from the TVs.
                // Added this to help with the transition.
                pDmrsDynPrm->dmrsAddlnPos       = 0;
                pDmrsDynPrm->dmrsMaxLen         = 0;
            }
        }


        // uePrms
        uint16_t nUes                        = uePrms.size();
        cellGrpDynParams.nUes                = nUes;

        for (int ue_id = 0; ue_id < nUes; ue_id++){
            const py::object ue_params = uePrms[ue_id];
            cellGrpDynParams.pUePrms[ue_id].pduBitmap          = ue_params.attr("pduBitmap").cast<uint16_t>();
            uint16_t ueGrpIdx                                  = ue_params.attr("ueGrpPrmIdx").cast<uint16_t>();
            cellGrpDynParams.pUePrms[ue_id].ueGrpIdx           = ueGrpIdx;
            cellGrpDynParams.pUePrms[ue_id].pUeGrpPrm          = &cellGrpDynParams.pUeGrpPrms[ueGrpIdx];
            cellGrpDynParams.pUePrms[ue_id].scid               = ue_params.attr("SCID").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].dmrsPortBmsk       = ue_params.attr("dmrsPortBmsk").cast<uint16_t>();
            cellGrpDynParams.pUePrms[ue_id].mcsTableIndex      = 0; //ue_params.attr("mcsTableIndex").cast<uint8_t>(); // not used
            cellGrpDynParams.pUePrms[ue_id].mcsIndex           = 0; //ue_params.attr("mcsIndex").cast<uint8_t>(); // not used
            cellGrpDynParams.pUePrms[ue_id].targetCodeRate     = ue_params.attr("targetCodeRate").cast<uint16_t>();
            cellGrpDynParams.pUePrms[ue_id].qamModOrder        = ue_params.attr("qamModOrder").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].TBSize             = ue_params.attr("TBSize").cast<uint32_t>(); // for verification purpose
            cellGrpDynParams.pUePrms[ue_id].rv                 = ue_params.attr("rvIndex").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].rnti               = ue_params.attr("RNTI").cast<uint16_t>();
            cellGrpDynParams.pUePrms[ue_id].dataScramId        = ue_params.attr("dataScramblingId").cast<uint16_t>();
            cellGrpDynParams.pUePrms[ue_id].nUeLayers          = ue_params.attr("nrOfLayers").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].ndi                = ue_params.attr("newDataIndicator").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].harqProcessId      = ue_params.attr("harqProcessId").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].i_lbrm             = ue_params.attr("i_lbrm").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].maxLayers          = ue_params.attr("maxLayers").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].maxQm              = ue_params.attr("maxQm").cast<uint8_t>();
            cellGrpDynParams.pUePrms[ue_id].n_PRB_LBRM         = ue_params.attr("n_PRB_LBRM").cast<uint16_t>();
            cellGrpDynParams.pUePrms[ue_id].enableTfPrcd       = ue_params.attr("enableTfPrcd").cast<uint8_t>();

            try {
                if (py::hasattr(ue_params,"uciOnPusch"))
                {
                    py::object uciOnPusch           = ue_params.attr("uciOnPusch");

                    cuphyUciOnPuschPrm_t* pUCI      = cellGrpDynParams.pUePrms[ue_id].pUciPrms;

                    pUCI->nBitsHarq                 = uciOnPusch.attr("nBitsHarq").cast<uint16_t>();
                    pUCI->nBitsCsi1                 = uciOnPusch.attr("nBitsCsi1").cast<uint16_t>();
                    pUCI->alphaScaling              = uciOnPusch.attr("alphaScaling").cast<uint8_t>();
                    pUCI->betaOffsetHarqAck         = uciOnPusch.attr("betaOffsetHarqAck").cast<uint8_t>();
                    pUCI->betaOffsetCsi1            = uciOnPusch.attr("betaOffsetCsi1").cast<uint8_t>();
                    pUCI->betaOffsetCsi2            = uciOnPusch.attr("betaOffsetCsi2").cast<uint8_t>();
                    pUCI->rankBitOffset             = uciOnPusch.attr("nBitsCsi1").cast<uint8_t>();
                    pUCI->nRanksBits                = uciOnPusch.attr("nRanksBits").cast<uint8_t>();
                    pUCI->nCsiReports               = uciOnPusch.attr("nCsiReports").cast<uint8_t>();
                }
                else
                {
                    cellGrpDynParams.pUePrms[ue_id].pUciPrms = nullptr;
                }
            }
            catch(...) {
                printf("Bug handling UCI setup... \n");
                exit(1);
            }

            try {
                if (py::hasattr(ue_params,"debug_d_derateCbsIndices"))
                {
                    printf("Fatal error: PUSCH debug_d_derateCbsIndices is not supported yet... \n");
                    exit(1);
                }
                else
                {
                    cellGrpDynParams.pUePrms[ue_id].debug_d_derateCbsIndices = nullptr;
                }
            }
            catch(...) {
                printf("Bug handling debug_d_derateCbsIndices setup... \n");
                exit(1);
            }
        }

        // dbg prms
        try {
            py::object                    dbg_prms  = dynPrms.attr("dbg");
            puschDynParams.pDbg->enableApiLogging   = dbg_prms.attr("enableApiLogging").cast<uint8_t>();
        }
        catch(...) {
            printf("Couldn't read dyn_dbg_prms");
            puschDynParams.pDbg->enableApiLogging   = 0;
        }

        // data in
        py::list TDataRx   = dataIn.attr("tDataRx");

        for (int cell_id = 0; cell_id < nCells; cell_id ++ ){

            py::array TDataRx_array = TDataRx[cell_id];

            tRefTDataRx[cell_id] = tensorFromNumpy<std::complex<float>>(TDataRx_array, CUPHY_C_16F, tensor_flags::align_tight, (cudaStream_t)cuStream);

            CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
            puschDynParams.pDataIn->pTDataRx[cell_id].desc  = tRefTDataRx[cell_id].desc().handle();
            puschDynParams.pDataIn->pTDataRx[cell_id].pAddr = tRefTDataRx[cell_id].addr();



        }

        // data out
        puschDataOut.pCbCrcs                    = pointerFromNumpyArray<uint32_t>(dataOut.attr("cbCrcs"));
        puschDataOut.pTbCrcs                    = pointerFromNumpyArray<uint32_t>(dataOut.attr("tbCrcs"));
        puschDataOut.pTbPayloads                = pointerFromNumpyArray<uint8_t>(dataOut.attr("tbPayloads"));
        puschDataOut.h_harqBufferSizeInBytes    = pointerFromNumpyArray<uint32_t>(dataOut.attr("harqBufferSizeInBytes"));
        puschDataOut.pUciPayloads               = nullptr; //pointerFromNumpyArray<uint8_t>(dataOut.attr("uciPayloads"));
        puschDataOut.pUciCrcFlags               = nullptr; //pointerFromNumpyArray<uint8_t>(dataOut.attr("uciCrcFlags"));
        //puschDataOut.pUciDTXs                   = nullptr;
        puschDataOut.pNumCsi2Bits               = nullptr; //pointerFromNumpyArray<uint16_t>(dataOut.attr("numCsi2Bits"));
        puschDataOut.pStartOffsetsCbCrc         = pointerFromNumpyArray<uint32_t>(dataOut.attr("startOffsetsCbCrc"));
        puschDataOut.pStartOffsetsTbCrc         = pointerFromNumpyArray<uint32_t>(dataOut.attr("startOffsetsTbCrc"));
        puschDataOut.pStartOffsetsTbPayload     = pointerFromNumpyArray<uint32_t>(dataOut.attr("startOffsetsTbPayload"));
        puschDataOut.pUciOnPuschOutOffsets      = nullptr;
        puschDataOut.pTaEsts                    = pointerFromNumpyArray<float>(dataOut.attr("taEsts"));
        puschDataOut.pRssi                      = pointerFromNumpyArray<float>(dataOut.attr("rssi"));
        puschDataOut.pRsrp                      = pointerFromNumpyArray<float>(dataOut.attr("rsrp"));
        puschDataOut.pNoiseVarPreEq             = pointerFromNumpyArray<float>(dataOut.attr("noiseVarPreEq"));
        puschDataOut.pNoiseVarPostEq            = pointerFromNumpyArray<float>(dataOut.attr("noiseVarPostEq"));
        puschDataOut.pSinrPreEq                 = pointerFromNumpyArray<float>(dataOut.attr("sinrPreEq"));
        puschDataOut.pSinrPostEq                = pointerFromNumpyArray<float>(dataOut.attr("sinrPostEq"));
        puschDataOut.pCfoHz                     = pointerFromNumpyArray<float>(dataOut.attr("cfoHz"));

        puschDataOut.HarqDetectionStatus        = pointerFromNumpyArray<uint8_t>(dataOut.attr("HarqDetectionStatus"));
        puschDataOut.CsiP1DetectionStatus       = pointerFromNumpyArray<uint8_t>(dataOut.attr("CsiP1DetectionStatus"));
        puschDataOut.CsiP2DetectionStatus       = pointerFromNumpyArray<uint8_t>(dataOut.attr("CsiP2DetectionStatus"));

        // If we need UCI output in the future, should have
        // a function after run() to return those value to python.

        // Setup pipeline for slot 0 - phase 1
        puschDynParams.setupPhase = PUSCH_SETUP_PHASE_1;
        cuphySetupPuschRx((cuphyPuschRxHndl_t)puschHandle, &puschDynParams, nullptr);

        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

        // Return output data size.
        uint32_t* pTotNumTbs                    = pointerFromNumpyArray<uint32_t>(dataOut.attr("totNumTbs"));
        uint32_t* pTotNumCbs                    = pointerFromNumpyArray<uint32_t>(dataOut.attr("totNumCbs"));
        uint32_t* pTotNumPayloadBytes           = pointerFromNumpyArray<uint32_t>(dataOut.attr("totNumPayloadBytes"));
        uint16_t* pTotNumUciSegs                = pointerFromNumpyArray<uint16_t>(dataOut.attr("totNumUciSegs"));

        pTotNumTbs[0]           = puschDataOut.totNumTbs;
        pTotNumCbs[0]           = puschDataOut.totNumCbs;
        pTotNumPayloadBytes[0]  = puschDataOut.totNumPayloadBytes;
        pTotNumUciSegs[0]       = puschDataOut.totNumUciSegs;

        return;
        // Finish PUSCH_SETUP_PHASE_1.
    }

    // PUSCH_SETUP_PHASE_2.

    // Allocate HARQ buffers based on the calculated requirements from setupPhase 1
    for(int k = 0; k < puschDataOut.totNumTbs; k++) {
        try {
            py::list harqBuffersInOut = dataInOut.attr("harqBuffersInOut");
            uint8_t* harqBuffers = (uint8_t*)harqBuffersInOut[k].cast<uint64_t>();
            puschDataInOut.pHarqBuffersInOut[k] = harqBuffers;
        }
        catch(...) {
            printf("Warning: Cannot read harqBuffersInOut, set harqBuffers to 0... \n");
            uint8_t* harqBuffers;
            harqBuffers = static_cast<uint8_t*>(m_LinearAlloc.alloc(puschDataOut.h_harqBufferSizeInBytes[k] * sizeof(uint8_t)));
            printf("h_harqBufferSizeInBytes[k]: %d \n", puschDataOut.h_harqBufferSizeInBytes[k]);
            CUDA_CHECK(cudaMemsetAsync(harqBuffers, 0, puschDataOut.h_harqBufferSizeInBytes[k] * sizeof(uint8_t), (cudaStream_t)cuStream));

            puschDataInOut.pHarqBuffersInOut[k] = harqBuffers;
        }
    }


    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    puschDynParams.setupPhase = PUSCH_SETUP_PHASE_2;
    cuphySetupPuschRx((cuphyPuschRxHndl_t)puschHandle, &puschDynParams, nullptr);

    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

}


void PuschPipeline::runPuschRx(uint64_t puschHandle, uint64_t cuStream) {
    cuphyStatus_t status = cuphyRunPuschRx((cuphyPuschRxHndl_t)puschHandle, PUSCH_RUN_PHASE_3);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        std::cerr << "Error! cuphyRunPuschRx(): " << cuphyGetErrorString(status) << std::endl;
        exit(1);
    }
}


void PuschPipeline::writeDbgBufSynch(uint64_t puschHandle, uint64_t cuStream) {

    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    cuphyStatus_t status = cuphyWriteDbgBufSynch((cuphyPuschRxHndl_t)puschHandle, (cudaStream_t)cuStream);

    if(status != CUPHY_STATUS_SUCCESS)
    {
        std::cerr << "Error! cuphyWriteDbgBufSynch(): " << cuphyGetErrorString(status) << std::endl;
        exit(1);
    }
}


void PuschPipeline::destroyPuschRx(uint64_t puschHandle) {

    // Free dynamically allocated memory.

    // Static parameters.
    delete puschStatPrms.pWFreq;
    delete puschStatPrms.pWFreq4;
    delete puschStatPrms.pWFreqSmall;
    delete puschStatPrms.pShiftSeq;
    delete puschStatPrms.pUnShiftSeq;
    delete puschStatPrms.pShiftSeq4;
    delete puschStatPrms.pUnShiftSeq4;

    if (puschStatPrms.pDbg)
        delete puschStatPrms.pDbg;

    for (int i = 0; i < puschStatPrms.nMaxCells; i++) {
        delete puschStatPrms.pCellStatPrms[i].pPuschCellStatPrms;
    }
    delete[] puschStatPrms.pCellStatPrms;

    // Dynamic parameters.
    delete[] cellGrpDynParams.pCellPrms;

    for (int i = 0; i < MAX_N_USER_GROUPS_SUPPORTED; i++){
        delete[] cellGrpDynParams.pUeGrpPrms[i].pUePrmIdxs;
        delete cellGrpDynParams.pUeGrpPrms[i].pDmrsDynPrm;
    }
    delete[] cellGrpDynParams.pUeGrpPrms;

    for (int i = 0; i < MAX_N_USER_GROUPS_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP; i++){
        if (cellGrpDynParams.pUePrms[i].pUciPrms)
            delete cellGrpDynParams.pUePrms[i].pUciPrms;
        if (cellGrpDynParams.pUePrms[i].debug_d_derateCbsIndices)
            delete cellGrpDynParams.pUePrms[i].debug_d_derateCbsIndices;
    }
    delete[] cellGrpDynParams.pUePrms;

    if (puschDynParams.pDbg)
        delete puschDynParams.pDbg;

    // Destroy the PUSCH Rx handle.
    cuphyStatus_t status = cuphyDestroyPuschRx((cuphyPuschRxHndl_t)puschHandle);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        std::cerr << "Error! cuphyDestroyPuschRx(): " << cuphyGetErrorString(status) << std::endl;
        exit(1);
    }
}


} // pycuphy

#endif  // AERIAL_PYTHON_PUSCH_CPP
