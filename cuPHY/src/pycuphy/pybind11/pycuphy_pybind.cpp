/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "pycuphy_pusch.cpp"
#include "pycuphy_pdsch.cpp"
#include "pycuphy_ldpc_decoder.cpp"
#include "pycuphy_ldpc_encoder.cpp"
#include "pycuphy_polar_encoder.cpp"
#include "pycuphy_polar_decoder.cpp"
#include "pycuphy_ldpc_rate_match.cpp"
#include "pycuphy_ldpc_derate_match.cpp"
#include "pycuphy_channel_est.cpp"
#include "pycuphy_srs_chest.cpp"


namespace py = pybind11;
using namespace std::complex_literals;

namespace pycuphy {


template <typename T>
void fromNumpyBitArray(float* src, T* dst, uint32_t npDim0, uint32_t npDim1) {

    static_assert(std::is_integral<T>::value, "Integral destination required.");

    const int ELEMENT_SIZE = sizeof(T) * 8;

    for(uint32_t col = 0; col < npDim1; col++) {
        for(int row = 0; row < npDim0; row += ELEMENT_SIZE) {
            T bits = 0;

            for(int o = 0; o < ELEMENT_SIZE; o++) {
                if(row + o < npDim0) {
                    float bit = *(src + (npDim1 * (row + o) + col));
                    T bit_0 = (T)bit & 0x1;
                    bits |= (bit_0 << o);
                }
            }

            // Target address. Set the data.
            T* dstElem = dst + (row / ELEMENT_SIZE) + (npDim0 / ELEMENT_SIZE) * col;
            *dstElem = bits;
        }
    }

}


template <typename T>
void toNumpyBitArray(T* src, float* dst, uint32_t dstDim0, uint32_t dstDim1) {

    static_assert(std::is_integral<T>::value, "Integral source required.");

    const int ELEMENT_SIZE = sizeof(T) * 8;

    uint32_t srcDim0 = (dstDim0 + ELEMENT_SIZE - 1) / ELEMENT_SIZE;
    for(uint32_t col = 0; col < dstDim1; col++) {
        for(uint32_t row = 0; row < dstDim0; row += ELEMENT_SIZE) {
            T* srcElem = src + srcDim0 * col + (row / ELEMENT_SIZE);
            for(int o = 0; o < ELEMENT_SIZE && (row + o < dstDim0); o++) {
                T bit = ((*srcElem & (1 << o)) >> o) & 1;
                float* dstElem = dst + dstDim1 * (row + o) + col;
                *dstElem = (float)bit;
            }
        }
    }
}


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1) {
    return py::array_t<T>(
        {dim0, dim1},  // Shape
        {sizeof(T), sizeof(T) * dim0},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2) {
    return py::array_t<T>(
        {dim0, dim1, dim2},  // Shape
        {sizeof(T), sizeof(T) * dim0, sizeof(T) * dim0 * dim1},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3) {
    return py::array_t<T>(
        {dim0, dim1, dim2, dim3},  // Shape
        {sizeof(T), sizeof(T) * dim0, sizeof(T) * dim0 * dim1, sizeof(T) * dim0 * dim1 * dim2},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

template <typename T, int flags = py::array::c_style | py::array::forcecast>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags hostTensorFlags,
                                     cuphy::tensor_flags deviceTensorFlags,
                                     cudaStream_t cuStream) {
    py::array_t<T, flags> array = py_array;
    py::buffer_info buf = array.request();

    cuphy::tensor_pinned hostTensor;
    cuphy::tensor_device deviceTensor;

    int rank = buf.ndim;
    switch (rank)
    {
    case 2:
        hostTensor = cuphy::tensor_device(buf.ptr,
                                          convertFromType,
                                          buf.shape[0],
                                          buf.shape[1],
                                          hostTensorFlags);
        deviceTensor = cuphy::tensor_device(inputDevPtr,
                                            convertToType,
                                            buf.shape[0],
                                            buf.shape[1],
                                            deviceTensorFlags);
        break;

    case 3:
        hostTensor = cuphy::tensor_device(buf.ptr,
                                          convertFromType,
                                          buf.shape[0],
                                          buf.shape[1],
                                          buf.shape[2],
                                          hostTensorFlags);
        deviceTensor = cuphy::tensor_device(inputDevPtr,
                                            convertToType,
                                            buf.shape[0],
                                            buf.shape[1],
                                            buf.shape[2],
                                            deviceTensorFlags);
        break;

    default:
        throw std::runtime_error("deviceFromNumpy: Dimension error!");
        break;
    }

    // Obtain a tensor in device memory from host Numpy array.
    deviceTensor.convert(hostTensor, cuStream);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    return deviceTensor;
}


template <typename T, int flags = py::array::c_style | py::array::forcecast>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream) {
    return deviceFromNumpy<T, flags>(py_array,
                                     inputDevPtr,
                                     convertFromType,
                                     convertToType,
                                     cuphy::tensor_flags::align_tight,
                                     tensorDescFlags,
                                     cuStream);
}


template <typename T, int flags = py::array::c_style | py::array::forcecast>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream) {
    py::array_t<T, flags> array = py_array;
    py::buffer_info buf = array.request();

    cuphy::tensor_pinned hostTensor;
    cuphy::tensor_device deviceTensor;

    int rank = buf.ndim;
    switch (rank)
    {
    case 2:
        hostTensor = cuphy::tensor_device(buf.ptr,
                                          convertFromType,
                                          buf.shape[0],
                                          buf.shape[1],
                                          cuphy::tensor_flags::align_tight);
        deviceTensor = cuphy::tensor_device(convertToType,
                                            buf.shape[0],
                                            buf.shape[1],
                                            tensorDescFlags);
        break;

    case 3:
        hostTensor = cuphy::tensor_device(buf.ptr,
                                          convertFromType,
                                          buf.shape[0],
                                          buf.shape[1],
                                          buf.shape[2],
                                          cuphy::tensor_flags::align_tight);
        deviceTensor = cuphy::tensor_device(convertToType,
                                            buf.shape[0],
                                            buf.shape[1],
                                            buf.shape[2],
                                            tensorDescFlags);
        break;

    default:
        throw std::runtime_error("deviceFromNumpy: Dimension error!");
        break;
    }

    // Obtain a tensor in device memory from host Numpy array.
    deviceTensor.convert(hostTensor, cuStream);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    return deviceTensor;
}



template <typename T>
py::array_t<T> deviceToNumpy(uint64_t deviceAddr,
                             uint64_t hostAddr,
                             py::list dimensions,
                             uint64_t cuStream) {
    int nDim = dimensions.size();

    // T needs to be either std::complex<float> or float.
    // TODO: Fix this hack where we need to determine types based on T as it restricts
    // the use of this function.
    cuphyDataType_t deviceDataType, hostDataType;
    if(std::is_same<T, std::complex<float>>::value) {
        deviceDataType = CUPHY_C_16F;
        hostDataType = CUPHY_C_32F;
    }
    else if(std::is_same<T, float>::value) {
        deviceDataType = CUPHY_R_32F;
        hostDataType = CUPHY_R_32F;
    }
    else {
        throw std::runtime_error("deviceToNumpy: Unsupported data type!");
    }

    if(nDim == 2) {
        int dim0 = dimensions[0].cast<int>();
        int dim1 = dimensions[1].cast<int>();

        tensor_device deviceTensor = tensor_device((void*)deviceAddr, deviceDataType, dim0, dim1, cuphy::tensor_flags::align_tight);
        tensor_pinned hostTensor = tensor_pinned((void*)hostAddr, hostDataType, dim0, dim1, cuphy::tensor_flags::align_tight);

        hostTensor.convert(deviceTensor, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
        return hostToNumpy<T>((T*)hostAddr, dim0, dim1);
    }

    else if(nDim ==3) {
        int dim0 = dimensions[0].cast<int>();
        int dim1 = dimensions[1].cast<int>();
        int dim2 = dimensions[2].cast<int>();

        tensor_device deviceTensor = tensor_device((void*)deviceAddr, deviceDataType, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);
        tensor_pinned hostTensor = tensor_pinned((void*)hostAddr, hostDataType, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);

        hostTensor.convert(deviceTensor, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
        return hostToNumpy<T>((T*)hostAddr, dim0, dim1, dim2);
    }

    else {
        throw std::runtime_error("\nInvalid tensor dimensions!\n");
    }
}

}  // pycuphy


PYBIND11_MODULE(_pycuphy, m) {
    m.doc() = "Python bindings for cuPHY"; // optional module docstring

    py::class_<pycuphy::PdschPipeline>(m, "PdschPipeline")
        .def(py::init())
        .def("create_pdsch_tx", &pycuphy::PdschPipeline::createPdschTx)
        .def("setup_pdsch_tx", &pycuphy::PdschPipeline::setupPdschTx)
        .def("run_pdsch_tx", &pycuphy::PdschPipeline::runPdschTx)
        .def("get_ldpc_output", &pycuphy::PdschPipeline::getLdpcOutputPerTbPerCell)
        .def("destroy_pdsch_tx", &pycuphy::PdschPipeline::destroyPdschTx);

    py::class_<pycuphy::PuschPipeline>(m, "PuschPipeline")
        .def(py::init())
        .def("create_pusch_rx", &pycuphy::PuschPipeline::createPuschRx)
        .def("setup_pusch_rx", &pycuphy::PuschPipeline::setupPuschRx)
        .def("run_pusch_rx", &pycuphy::PuschPipeline::runPuschRx)
        .def("destroy_pusch_rx", &pycuphy::PuschPipeline::destroyPuschRx)
        .def("write_dbg_buf_synch", &pycuphy::PuschPipeline::writeDbgBufSynch);

    py::class_<pycuphy::LdpcEncoder>(m, "LdpcEncoder")
        .def(py::init<const uint64_t, const uint64_t, const uint64_t, const uint64_t>())
        .def("encode", &pycuphy::LdpcEncoder::encode)
        .def("set_profiling_iterations", &pycuphy::LdpcEncoder::setProfilingIterations)
        .def("set_puncturing", &pycuphy::LdpcEncoder::setPuncturing);

    py::class_<pycuphy::LdpcDecoder>(m, "LdpcDecoder")
        .def(py::init<const uint64_t, const uint64_t, const uint64_t>())
        .def("decode", &pycuphy::LdpcDecoder::decode)
        .def("set_profiling_iterations", &pycuphy::LdpcDecoder::setProfilingIterations)
        .def("set_num_iterations", &pycuphy::LdpcDecoder::setNumIterations)
        .def("set_throughput_mode", &pycuphy::LdpcDecoder::setThroughputMode)
        .def("set_half_precision", &pycuphy::LdpcDecoder::setHalfPrecision);

    py::class_<pycuphy::LdpcRateMatch>(m, "LdpcRateMatch")
        .def(py::init<const uint64_t, const uint64_t, const uint64_t, const uint64_t, const uint64_t, const bool>())
        .def("rate_match", &pycuphy::LdpcRateMatch::rateMatch)
        .def("set_profiling_iterations", &pycuphy::LdpcRateMatch::setProfilingIterations);

    py::class_<pycuphy::LdpcDerateMatch>(m, "LdpcDerateMatch")
        .def(py::init<const uint64_t, const uint64_t, const bool>())
        .def("derate_match", &pycuphy::LdpcDerateMatch::derateMatch)
        .def("destroy", &pycuphy::LdpcDerateMatch::destroy)
        .def("set_profiling_iterations", &pycuphy::LdpcDerateMatch::setProfilingIterations);

    py::class_<pycuphy::PolarEncoder>(m, "PolarEncoder")
        .def(py::init<const uint64_t, const uint64_t, const uint64_t, const uint64_t, const uint64_t, const uint8_t>())
        .def("encode", &pycuphy::PolarEncoder::encode)
        .def("set_profiling_iterations", &pycuphy::PolarEncoder::setProfilingIterations);

    py::class_<pycuphy::PolarDecoder>(m, "PolarDecoder")
        .def(py::init<const uint64_t, const uint64_t, const uint64_t>())
        .def("decode", &pycuphy::PolarDecoder::decode)
        .def("set_profiling_iterations", &pycuphy::PolarDecoder::setProfilingIterations);

    py::class_<pycuphy::ChannelEstimator>(m, "ChannelEstimator")
        .def(py::init<const py::dict&, const uint64_t, const uint64_t, const uint64_t, const uint64_t, uint64_t>())
        .def("estimate", &pycuphy::ChannelEstimator::estimate);

    py::class_<pycuphy::SrsChannelEstimator>(m, "SrsChannelEstimator")
        .def(py::init<const py::dict&, const uint64_t, uint64_t>())
        .def("estimate", &pycuphy::SrsChannelEstimator::estimate)
        .def("get_srs_report", &pycuphy::SrsChannelEstimator::getSrsReport)
        .def("get_rb_snr_buffer", &pycuphy::SrsChannelEstimator::getRbSnrBuffer)
        .def("get_rb_snr_buffer_offsets", &pycuphy::SrsChannelEstimator::getRbSnrBufferOffsets);

    py::class_<cuphySrsReport_t>(m, "SrsReport")
        .def(py::init<>())
        .def_readwrite("to_est_ms", &cuphySrsReport_t::toEstMicroSec)
        .def_readwrite("wideband_snr", &cuphySrsReport_t::widebandSnr)
        .def_readwrite("wideband_noise_energy", &cuphySrsReport_t::widebandNoiseEnergy)
        .def_readwrite("wideband_signal_energy", &cuphySrsReport_t::widebandSignalEnergy)
        .def_property_readonly("wideband_sc_corr", [](const cuphySrsReport_t& prm) { return std::complex<float>(__high2float(prm.widebandScCorr), __low2float(prm.widebandScCorr)); });

    py::class_<cuphySrsCellPrms_t>(m, "SrsCellPrms")
        .def(py::init<uint16_t, uint16_t, uint8_t, uint8_t, uint16_t, uint8_t>())
        .def_readwrite("slotNum", &cuphySrsCellPrms_t::slotNum)
        .def_readwrite("frameNum", &cuphySrsCellPrms_t::frameNum)
        .def_readwrite("srsStartSym", &cuphySrsCellPrms_t::srsStartSym)
        .def_readwrite("nSrsSym", &cuphySrsCellPrms_t::nSrsSym)
        .def_readwrite("nRxAntSrs", &cuphySrsCellPrms_t::nRxAntSrs)
        .def_readwrite("mu", &cuphySrsCellPrms_t::mu);

    py::class_<cuphyUeSrsPrm_t>(m, "UeSrsPrms")
        .def(py::init<uint16_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint16_t, uint8_t, uint8_t, uint8_t,
                      uint8_t, uint8_t, uint16_t, uint8_t, uint8_t, uint16_t, uint16_t, uint8_t, uint16_t, uint32_t,
                      uint16_t, uint32_t, uint32_t>())
        .def_readwrite("cellIdx", &cuphyUeSrsPrm_t::cellIdx)
        .def_readwrite("nAntPorts", &cuphyUeSrsPrm_t::nAntPorts)
        .def_readwrite("nSyms", &cuphyUeSrsPrm_t::nSyms)
        .def_readwrite("nRepetitions", &cuphyUeSrsPrm_t::nRepetitions)
        .def_readwrite("combSize", &cuphyUeSrsPrm_t::combSize)
        .def_readwrite("startSym", &cuphyUeSrsPrm_t::startSym)
        .def_readwrite("sequenceId", &cuphyUeSrsPrm_t::sequenceId)
        .def_readwrite("configIdx", &cuphyUeSrsPrm_t::configIdx)
        .def_readwrite("bandwidthIdx", &cuphyUeSrsPrm_t::bandwidthIdx)
        .def_readwrite("combOffset", &cuphyUeSrsPrm_t::combOffset)
        .def_readwrite("cyclicShift", &cuphyUeSrsPrm_t::cyclicShift)
        .def_readwrite("frequencyPosition", &cuphyUeSrsPrm_t::frequencyPosition)
        .def_readwrite("frequencyShift", &cuphyUeSrsPrm_t::frequencyShift)
        .def_readwrite("frequencyHopping", &cuphyUeSrsPrm_t::frequencyHopping)
        .def_readwrite("resourceType", &cuphyUeSrsPrm_t::resourceType)
        .def_readwrite("Tsrs", &cuphyUeSrsPrm_t::Tsrs)
        .def_readwrite("Toffset", &cuphyUeSrsPrm_t::Toffset)
        .def_readwrite("groupOrSequenceHopping", &cuphyUeSrsPrm_t::groupOrSequenceHopping)
        .def_readwrite("chEstBuffIdx", &cuphyUeSrsPrm_t::chEstBuffIdx)
        .def_property_readonly("srsAntPortToUeAntMap", [](const cuphyUeSrsPrm_t& prm) { return std::vector<uint8_t>(prm.srsAntPortToUeAntMap, prm.srsAntPortToUeAntMap+4); })
        .def_readwrite("rnti", &cuphyUeSrsPrm_t::rnti)
        .def_readwrite("handle", &cuphyUeSrsPrm_t::handle)
        .def_readwrite("usage", &cuphyUeSrsPrm_t::usage);

    m.def("device_to_numpy", &pycuphy::deviceToNumpy<std::complex<float>>);
    m.def("device_to_numpy", &pycuphy::deviceToNumpy<float>);

 }
