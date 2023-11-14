/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <vector>

#include "cuphy.h"
#include "tensor_desc.hpp"
#include "pycuphy_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace cuphy;

using namespace std::complex_literals;
namespace py = pybind11;

#ifndef PYCUPHY_SRS_CHEST_CPP
#define PYCUPHY_SRS_CHEST_CPP


namespace pycuphy {


class SrsChannelEstimator {
public:
    SrsChannelEstimator(
        const py::dict& chEstParams,
        const uint64_t inputDevicePtr,
        uint64_t cuStream);
    ~SrsChannelEstimator();

    const std::vector<py::array_t<std::complex<float>>>& estimate(const py::array& inputData,
                                                                  const uint16_t nSrsUes,
                                                                  const uint16_t nCells,
                                                                  const uint16_t nPrbGrps,
                                                                  const uint16_t startPrbGrp,
                                                                  std::vector<cuphySrsCellPrms_t>& srsCellPrms,
                                                                  std::vector<cuphyUeSrsPrm_t>& ueSrsPrms);
    const std::vector<cuphySrsReport_t>& getSrsReport();
    py::array_t<float> getRbSnrBuffer();
    const std::vector<uint32_t>& getRbSnrBufferOffsets();

private:
    size_t getMaxMem() const;
    void destroy();
    void allocateDescr();
    void setChEstParams(const py::dict& chEstParams);

    cuphy::linear_alloc<128, cuphy::device_alloc> m_linearAlloc;

    // Descriptor variables.
    size_t m_statDescrSizeBytes;
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_statDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_statDescrBufGpu;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    // SRS estimator handle.
    cuphySrsChEst0Hndl_t m_srsChEstHndl;

    // Input data.
    std::vector<cuphyTensorPrm_t> m_tDataRx;

    // Outputs.
    std::vector<py::array_t<std::complex<float>>> m_chEsts;

    cuphySrsReport_t* m_dSrsReports;
    std::vector<cuphySrsReport_t> m_srsReports;

    std::vector<cuphy::tensor_device> m_tSrsChEstVec;
    std::vector<cuphySrsChEstBuffInfo_t> m_srsChEstBuffInfo;

    std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>> m_chEstCpuBuffVec;
    std::vector<void*> m_dChEstToL2Vec;
    std::vector<cuphySrsChEstToL2_t> m_chEstToL2Vec;

    float* m_dSrsRbSnrBuffer;
    std::vector<float> m_srsRbSnrBuffer;
    std::vector<uint32_t> m_srsRbSnrBuffOffsets;

    // Filter tensors and parameters.
    cuphy::tensor_device m_tPrmFocc_table;

    cuphy::tensor_device m_tPrmW_comb2_nPorts1_wide;
    cuphy::tensor_device m_tPrmW_comb2_nPorts2_wide;
    cuphy::tensor_device m_tPrmW_comb2_nPorts4_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts1_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts2_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts4_wide;

    cuphy::tensor_device m_tPrmW_comb2_nPorts1_narrow;
    cuphy::tensor_device m_tPrmW_comb2_nPorts2_narrow;
    cuphy::tensor_device m_tPrmW_comb2_nPorts4_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts1_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts2_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts4_narrow;

    cuphySrsFilterPrms_t m_srsFilterPrms;

    uint16_t m_nSrsUes;
    uint16_t m_nCells;

    void* m_inputDevicePtr;
    cudaStream_t m_cuStream;
};


size_t SrsChannelEstimator::getMaxMem() const {
    size_t maxNumSrsUes        =  1000;
    size_t maxCells            =  32;
    size_t maxRbSnrMem         =  maxNumSrsUes * 273 * sizeof(float);
    size_t maxSrsReportMem     =  maxNumSrsUes * sizeof(cuphySrsReport_t);
    size_t maxChEstToL2Mem     =  maxCells * 273 * 128 * 16 * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL;
    size_t maxMem              =  maxRbSnrMem + maxSrsReportMem + maxChEstToL2Mem;
    return maxMem;
}


void SrsChannelEstimator::setChEstParams(const py::dict& chEstParams) {
    m_tPrmFocc_table = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["focc_table"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmFocc_table.desc = m_tPrmFocc_table.desc().handle();
    m_srsFilterPrms.tPrmFocc_table.pAddr = m_tPrmFocc_table.addr();

    m_tPrmW_comb2_nPorts1_wide = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb2_nPorts1_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts1_wide.desc = m_tPrmW_comb2_nPorts1_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts1_wide.pAddr = m_tPrmW_comb2_nPorts1_wide.addr();

    m_tPrmW_comb2_nPorts2_wide = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb2_nPorts2_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts2_wide.desc = m_tPrmW_comb2_nPorts2_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts2_wide.pAddr = m_tPrmW_comb2_nPorts2_wide.addr();

    m_tPrmW_comb2_nPorts4_wide = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb2_nPorts4_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts4_wide.desc = m_tPrmW_comb2_nPorts4_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts4_wide.pAddr = m_tPrmW_comb2_nPorts4_wide.addr();


    m_tPrmW_comb4_nPorts1_wide = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb4_nPorts1_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts1_wide.desc = m_tPrmW_comb4_nPorts1_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts1_wide.pAddr = m_tPrmW_comb4_nPorts1_wide.addr();

    m_tPrmW_comb4_nPorts2_wide = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb4_nPorts2_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts2_wide.desc = m_tPrmW_comb4_nPorts2_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts2_wide.pAddr = m_tPrmW_comb4_nPorts2_wide.addr();

    m_tPrmW_comb4_nPorts4_wide = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb4_nPorts4_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts4_wide.desc = m_tPrmW_comb4_nPorts4_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts4_wide.pAddr = m_tPrmW_comb4_nPorts4_wide.addr();

    m_tPrmW_comb2_nPorts1_narrow = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb2_nPorts1_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts1_narrow.desc = m_tPrmW_comb2_nPorts1_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts1_narrow.pAddr = m_tPrmW_comb2_nPorts1_narrow.addr();

    m_tPrmW_comb2_nPorts2_narrow = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb2_nPorts2_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts2_narrow.desc = m_tPrmW_comb2_nPorts2_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts2_narrow.pAddr = m_tPrmW_comb2_nPorts2_narrow.addr();

    m_tPrmW_comb2_nPorts4_narrow = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb2_nPorts4_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts4_narrow.desc = m_tPrmW_comb2_nPorts4_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts4_narrow.pAddr = m_tPrmW_comb2_nPorts4_narrow.addr();

    m_tPrmW_comb4_nPorts1_narrow = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb4_nPorts1_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts1_narrow.desc = m_tPrmW_comb4_nPorts1_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts1_narrow.pAddr = m_tPrmW_comb4_nPorts1_narrow.addr();

    m_tPrmW_comb4_nPorts2_narrow = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb4_nPorts2_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts2_narrow.desc = m_tPrmW_comb4_nPorts2_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts2_narrow.pAddr = m_tPrmW_comb4_nPorts2_narrow.addr();

    m_tPrmW_comb4_nPorts4_narrow = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        chEstParams["W_comb4_nPorts4_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts4_narrow.desc = m_tPrmW_comb4_nPorts4_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts4_narrow.pAddr = m_tPrmW_comb4_nPorts4_narrow.addr();

    m_srsFilterPrms.noisEstDebias_comb2_nPorts1 = chEstParams["noisEstDebias_comb2_nPorts1"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb2_nPorts2 = chEstParams["noisEstDebias_comb2_nPorts2"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb2_nPorts4 = chEstParams["noisEstDebias_comb2_nPorts4"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts1 = chEstParams["noisEstDebias_comb4_nPorts1"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts2 = chEstParams["noisEstDebias_comb4_nPorts2"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts4 = chEstParams["noisEstDebias_comb4_nPorts4"].cast<float>();
}


void SrsChannelEstimator::allocateDescr() {
    size_t statDescrAlignBytes, dynDescrAlignBytes;
    cuphyStatus_t statusGetWorkspaceSize = cuphySrsChEst0GetDescrInfo(&m_statDescrSizeBytes,
                                                                      &statDescrAlignBytes,
                                                                      &m_dynDescrSizeBytes,
                                                                      &dynDescrAlignBytes);

    m_statDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_statDescrSizeBytes);
    m_statDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_statDescrSizeBytes);
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);
}


SrsChannelEstimator::SrsChannelEstimator(const py::dict& chEstParams,
                                         const uint64_t inputDevicePtr,
                                         uint64_t cuStream) :
    m_linearAlloc(getMaxMem()) {

    m_inputDevicePtr = (void *)inputDevicePtr;
    m_cuStream = (cudaStream_t)cuStream;

    m_linearAlloc.memset(0, m_cuStream);
    cudaStreamSynchronize(m_cuStream);

    // Allocate descriptors.
    allocateDescr();

    // Set channel esitmation filter parameters.
    setChEstParams(chEstParams);

    // Create the SRS channel estimator object.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t status = cuphyCreateSrsChEst0(&m_srsChEstHndl,
                                                &m_srsFilterPrms,
                                                enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                m_statDescrBufCpu.addr(),
                                                m_statDescrBufGpu.addr(),
                                                m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateSrsChEst0()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        cudaMemcpyAsync(m_statDescrBufGpu.addr(),
                        m_statDescrBufCpu.addr(),
                        m_statDescrSizeBytes,
                        cudaMemcpyHostToDevice,
                        m_cuStream);
    }
    cudaStreamSynchronize(m_cuStream);
}


SrsChannelEstimator::~SrsChannelEstimator() {
    destroy();
}


const std::vector<py::array_t<std::complex<float>>>& SrsChannelEstimator::estimate(
        const py::array& inputData,
        const uint16_t nSrsUes,
        const uint16_t nCells,
        const uint16_t nPrbGrps,
        const uint16_t startPrbGrp,
        std::vector<cuphySrsCellPrms_t>& srsCellPrms,
        std::vector<cuphyUeSrsPrm_t>& ueSrsPrms) {

    m_nSrsUes = nSrsUes;
    m_nCells = nCells;

    m_tDataRx.resize(nCells);
    m_srsReports.resize(nSrsUes);
    m_srsChEstBuffInfo.resize(nSrsUes);
    m_chEstCpuBuffVec.resize(nSrsUes);
    m_dChEstToL2Vec.resize(nSrsUes);
    m_chEstToL2Vec.resize(nSrsUes);
    m_srsRbSnrBuffer.resize(nSrsUes * 273);
    m_srsRbSnrBuffOffsets.resize(nSrsUes);

    // Read input data into device memory.
    cuphy::tensor_device deviceRxDataTensor = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        inputData,
        m_inputDevicePtr,
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);

    m_tDataRx[0].desc = deviceRxDataTensor.desc().handle();
    m_tDataRx[0].pAddr = deviceRxDataTensor.addr();

    // Initializations.
    uint32_t rbSnrBufferSize = nSrsUes * 273;
    m_dSrsRbSnrBuffer = static_cast<float*>(m_linearAlloc.alloc(sizeof(float) * rbSnrBufferSize));
    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx) {

        m_srsRbSnrBuffOffsets[ueIdx] = ueIdx * 273;

        // These need to be initialized to zero.
        m_srsReports[ueIdx].widebandSignalEnergy = 0.f;
        m_srsReports[ueIdx].widebandNoiseEnergy = 0.f;
        m_srsReports[ueIdx].widebandScCorr = __floats2half2_rn(0.f, 0.f);

        uint8_t nUeAnt = ueSrsPrms[ueIdx].nAntPorts;
        uint16_t cellIdx = ueSrsPrms[ueIdx].cellIdx;
        uint16_t nRxAntSrs = srsCellPrms[cellIdx].nRxAntSrs;
        m_tSrsChEstVec.push_back(cuphy::tensor_device(CUPHY_C_32F,
                                                      nPrbGrps,
                                                      nRxAntSrs,
                                                      nUeAnt,
                                                      cuphy::tensor_flags::align_tight));

        size_t nBytesInChEstBuffer = nPrbGrps * nRxAntSrs * nUeAnt * 8;
        // Init ChEst buffer to zero.
        cudaMemsetAsync(m_tSrsChEstVec[ueIdx].addr(), 0, nBytesInChEstBuffer, m_cuStream);

        // Allocate CPU buffer for ChEst.
        size_t maxBufferSize = 273 * 128 * 4 * sizeof(float2);
        m_chEstCpuBuffVec[ueIdx] = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(maxBufferSize));
        m_dChEstToL2Vec[ueIdx] = m_linearAlloc.alloc(maxBufferSize);
        m_chEstToL2Vec[ueIdx].pChEstCpuBuff = m_chEstCpuBuffVec[ueIdx].addr();

        m_srsChEstBuffInfo[ueIdx].tChEstBuffer.desc = m_tSrsChEstVec[ueIdx].desc().handle();
        m_srsChEstBuffInfo[ueIdx].tChEstBuffer.pAddr = m_tSrsChEstVec[ueIdx].addr();
        m_srsChEstBuffInfo[ueIdx].startPrbGrp = startPrbGrp;
    }

    m_dSrsReports = static_cast<cuphySrsReport_t*>(m_linearAlloc.alloc(sizeof(cuphySrsReport_t) * nSrsUes));
    cudaMemcpyAsync(m_dSrsReports,
                    m_srsReports.data(),
                    sizeof(cuphySrsReport_t) * nSrsUes,
                    cudaMemcpyHostToDevice,
                    m_cuStream);
    cudaStreamSynchronize(m_cuStream);

    // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
    cuphySrsChEst0LaunchCfg_t srsChEstLaunchCfg;

    // Setup function populates dynamic descriptor and launch config. Option to copy descriptors to GPU during setup call.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t setupStatus = cuphySetupSrsChEst0(m_srsChEstHndl,
                                                    nSrsUes,
                                                    ueSrsPrms.data(),
                                                    nCells,
                                                    m_tDataRx.data(),
                                                    srsCellPrms.data(),
                                                    m_dSrsRbSnrBuffer,
                                                    m_srsRbSnrBuffOffsets.data(),
                                                    m_dSrsReports,
                                                    m_srsChEstBuffInfo.data(),
                                                    m_dChEstToL2Vec.data(),
                                                    m_chEstToL2Vec.data(),
                                                    static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                    m_dynDescrBufCpu.addr(),
                                                    m_dynDescrBufGpu.addr(),
                                                    &srsChEstLaunchCfg,
                                                    m_cuStream);
    if(!enableCpuToGpuDescrAsyncCpy) {
        cudaMemcpyAsync(m_dynDescrBufGpu.addr(),
                        m_dynDescrBufCpu.addr(),
                        m_dynDescrSizeBytes,
                        cudaMemcpyHostToDevice,
                        m_cuStream);
    }
    cudaStreamSynchronize(m_cuStream);

    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = srsChEstLaunchCfg.kernelNodeParamsDriver;
    CUresult srsChEst0RunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
                                                 kernelNodeParamsDriver.gridDimX,
                                                 kernelNodeParamsDriver.gridDimY,
                                                 kernelNodeParamsDriver.gridDimZ,
                                                 kernelNodeParamsDriver.blockDimX,
                                                 kernelNodeParamsDriver.blockDimY,
                                                 kernelNodeParamsDriver.blockDimZ,
                                                 kernelNodeParamsDriver.sharedMemBytes,
                                                 m_cuStream,
                                                 kernelNodeParamsDriver.kernelParams,
                                                 kernelNodeParamsDriver.extra);
    if(srsChEst0RunStatus != CUDA_SUCCESS) {
        throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
    }
    cudaStreamSynchronize(m_cuStream);

    // Create the return value.
    for(int ueIdx=0; ueIdx < nSrsUes; ueIdx++) {
        cuphyTensorPrm_t* pChEst = &m_srsChEstBuffInfo[ueIdx].tChEstBuffer;
        const ::tensor_desc& tDesc = static_cast<const ::tensor_desc&>(*pChEst->desc);
        const tensor_layout_any& tLayout = tDesc.layout();

        uint16_t nPrbGrpEsts = tLayout.dimensions[0];
        uint16_t nGnbAnts = tLayout.dimensions[1];
        uint16_t nUeAnts = tLayout.dimensions[2];

        // Create the numpy array for the output channel estimates.
        cuphy::tensor_device dChEstTensor = cuphy::tensor_device((std::complex<float>*)pChEst->pAddr, CUPHY_C_32F, nPrbGrpEsts, nGnbAnts, nUeAnts, cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hChEstTensor = cuphy::tensor_pinned(CUPHY_C_32F, nPrbGrpEsts, nGnbAnts, nUeAnts, cuphy::tensor_flags::align_tight);
        hChEstTensor.convert(dChEstTensor, m_cuStream);
        cudaStreamSynchronize(m_cuStream);
        py::array_t<std::complex<float>> chEst = hostToNumpy<std::complex<float>>((std::complex<float>*)hChEstTensor.addr(),
                                                                                  nPrbGrpEsts, nGnbAnts, nUeAnts);

        // The return value is a vector of Numpy arrays.
        m_chEsts.push_back(chEst);
    }

    return m_chEsts;
}


const std::vector<cuphySrsReport_t>& SrsChannelEstimator::getSrsReport() {
    // Copy results to host buffers.
    cudaMemcpyAsync(m_srsReports.data(),
                    m_dSrsReports,
                    sizeof(cuphySrsReport_t) * m_nSrsUes,
                    cudaMemcpyDeviceToHost,
                    m_cuStream);
    cudaStreamSynchronize(m_cuStream);
    return m_srsReports;
}


py::array_t<float> SrsChannelEstimator::getRbSnrBuffer() {
    uint32_t rbSnrBufferSize = m_nSrsUes * 273;

    // Copy results to host buffers.
    cudaMemcpyAsync(m_srsRbSnrBuffer.data(),
                    m_dSrsRbSnrBuffer,
                    sizeof(float) * rbSnrBufferSize,
                    cudaMemcpyDeviceToHost,
                    m_cuStream);
    cudaStreamSynchronize(m_cuStream);

    py::array_t<float> rbSnrs = hostToNumpy<float>((float*)m_srsRbSnrBuffer.data(), m_nSrsUes, 273);
    return rbSnrs;
}


const std::vector<uint32_t>& SrsChannelEstimator::getRbSnrBufferOffsets() {
    return m_srsRbSnrBuffOffsets;
}


void SrsChannelEstimator::destroy() {
    // Destroy the SRS channel estimation handle.
    cuphyStatus_t status = cuphyDestroySrsChEst0(m_srsChEstHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroySrsChEst0()");
    }
}

} // namespace pycuphy


#endif // PYCUPHY_SRS_CHEST_CPP
