/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"

#define CHECK_CUPHY(stmt)                                                 \
    do {                                                                  \
        cuphyStatus_t s = stmt;                                           \
        if(CUPHY_STATUS_SUCCESS != s)                                     \
        {                                                                 \
            fprintf(stderr, "CUPHY error: %s\n", cuphyGetErrorString(s)); \
            exit(1);                                                      \
        }                                                                 \
    } while(0)

#define CHECK_CUPHYHDF5(stmt)                                                     \
    do {                                                                          \
        cuphyHDF5Status_t s = stmt;                                               \
        if(CUPHYHDF5_STATUS_SUCCESS != s)                                         \
        {                                                                         \
            fprintf(stderr, "CUPHYHDF5 error: %s\n", cuphyHDF5GetErrorString(s)); \
            exit(1);                                                              \
        }                                                                         \
    } while(0)

int main(int argc, char* argv[])
{
    cuphyTensorDescriptor_t tensorDesc;
    int dimensions[3] = {10, 20, 100};
    int strides[3]    = {1,  10, 200};
    CHECK_CUPHY(cuphyCreateTensorDescriptor(&tensorDesc));
    CHECK_CUPHY(cuphySetTensorDescriptor(tensorDesc, CUPHY_C_32F, 3, dimensions, strides, 0));
    CHECK_CUPHY(cuphyDestroyTensorDescriptor(tensorDesc));
    
    cuphyNvlogFmtHelper nvlog_fmt("test_example.log");
    
#if 0
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("texture alignment = %lu, texture pitch alignment = %lu\n", prop.textureAlignment, prop.texturePitchAlignment);
    void*  addr = nullptr;
    size_t pitch = 0;
    for(size_t w = 32; w < 2048; w+=32)
    {
        cudaMallocPitch(&addr, &pitch, w, 3);
        printf("w = %lu, pitch = %lu\n", w, pitch);
        cudaFree(addr);
    }
#endif
#if 0
    hid_t h5File = H5Fopen("test.h5",
                           H5F_ACC_RDONLY,
                           H5P_DEFAULT);
    if(h5File < 0)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error opening HDF5 file");
    }
    else
    {
        printf("HDF5 file opened successfully\n");
        hid_t h5Dataset = H5Dopen(h5File, "A", H5P_DEFAULT);
        if(h5Dataset < 0)
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error opening dataset");
        }
        else
        {
            printf("HDF5 dataset opened successfully\n");
            int             dims[CUPHY_DIM_MAX];
            cuphyDataType_t type;
            int             numDims;
            CHECK_CUPHYHDF5(cuphyHDF5GetDatasetInfo(h5Dataset,
                                                    CUPHY_DIM_MAX,
                                                    &type,
                                                    &numDims,
                                                    dims));
            printf("type = %s\n", cuphyGetDataTypeString(type));
            for(int i = 0; i < numDims; ++i)
            {
                printf("dim[%i]: %i\n", i, dims[i]);
            }
            H5Dclose(h5Dataset);
        }
        H5Fclose(h5File);
    }
#endif
#if 0
    hid_t h5File = H5Fopen("test_single.h5",
                           H5F_ACC_RDONLY,
                           H5P_DEFAULT);
    if(h5File < 0)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error opening HDF5 file");
    }
    else
    {
        printf("HDF5 file opened successfully\n");
        hid_t h5Dataset = H5Dopen(h5File, "A", H5P_DEFAULT);
        if(h5Dataset < 0)
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error opening dataset");
        }
        else
        {
            printf("HDF5 dataset opened successfully\n");
            int             dims[CUPHY_DIM_MAX];
            cuphyDataType_t type;
            int             numDims;
            CHECK_CUPHYHDF5(cuphyHDF5GetDatasetInfo(h5Dataset,
                                                    CUPHY_DIM_MAX,
                                                    &type,
                                                    &numDims,
                                                    dims));
            printf("type = %s\n", cuphyGetDataTypeString(type));
            for(int i = 0; i < numDims; ++i)
            {
                printf("dim[%i]: %i\n", i, dims[i]);
            }
            // Allocate a destination tensor
            cuphyTensorDescriptor_t tensorDesc;
            cuphyStatus_t           s = cuphyCreateTensorDescriptor(&tensorDesc); 
            if(CUPHY_STATUS_SUCCESS == s)
            {
                s = cuphySetTensorDescriptor(tensorDesc,
                                             type,
                                             numDims,
                                             dims,
                                             nullptr,
                                             CUPHY_TENSOR_ALIGN_COALESCE/*0*/);
                if(CUPHY_STATUS_SUCCESS == s)
                {
                    size_t numBytes = 0;
                    s = cuphyGetTensorSizeInBytes(tensorDesc, &numBytes);
                    if(CUPHY_STATUS_SUCCESS == s)
                    {
                        void* deviceAddr = nullptr;
                        cudaError_t sCuda = cudaMalloc(&deviceAddr, numBytes);
                        if(cudaSuccess  == sCuda)
                        {
                            cuphyHDF5Status_t sH5 = cuphyHDF5ReadDataset(tensorDesc,
                                                                         deviceAddr,
                                                                         h5Dataset,
                                                                         0);
                            if(CUPHYHDF5_STATUS_SUCCESS == sH5)
                            {
                                printf("Dataset read successfully.\n");
                            }
                            else
                            {
                                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error reading HDF5 dataset: {}", cuphyHDF5GetErrorString(sH5));
                            }
                            cudaFree(deviceAddr);
                        }
                        else
                        {
                            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error allocating {} bytes on device", numBytes);
                        }
                    }
                    else
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error retrieving tensor size: {}", cuphyGetErrorString(s));
                    }
                }
                else
                {
                    NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error setting tensor descriptor: {}", cuphyGetErrorString(s));
                }
                cuphyDestroyTensorDescriptor(tensorDesc);
            }
            else
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error creating tensor descriptor: {}", cuphyGetErrorString(s));
            }

            
            H5Dclose(h5Dataset);
        }
        H5Fclose(h5File);
    }    
#endif
#if 0
    try
    {
        hdf5hpp::hdf5_file      f;
        f = hdf5hpp::hdf5_file::open("test_single.h5");
        hdf5hpp::hdf5_dataset   dset   = f.open_dataset("A");
        hdf5hpp::hdf5_dataspace dspace = dset.get_dataspace();
        std::vector<hsize_t>    dims   = dspace.get_dimensions();
        hdf5hpp::hdf5_datatype  dtype  = dset.get_datatype();
        printf("rank(A) = %i\n", dspace.get_rank());
        for(int i = 0; i < dspace.get_rank(); ++i)
        {
            printf("dims[%i] = %llu\n", i, dims[i]);
        }
        printf("class(A) = %s\n", dtype.get_class_string());
        printf("datatype size = %lu bytes\n", dtype.get_size_bytes());
        printf("num_elements(A) = %llu\n", dspace.get_num_elements());
        printf("buffer size = %lu bytes\n", dset.get_buffer_size_bytes());
        std::vector<float> data(dspace.get_num_elements());
        dset.read(data.data());
        for(size_t i = 0; i < data.size(); ++i)
        {
            printf("data[%lu] = %f\n", i, data[i]);
        }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
    }
#endif
#if 0
    try
    {
        std::unique_ptr<hdf5hpp::hdf5_file> file_ptr;
        file_ptr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open("test_single.h5")));
        hdf5hpp::hdf5_dataset   dset   = file_ptr->open_dataset("A");
        hdf5hpp::hdf5_dataspace dspace = dset.get_dataspace();
        std::vector<hsize_t>    dims   = dspace.get_dimensions();
        hdf5hpp::hdf5_datatype  dtype  = dset.get_datatype();
        printf("rank(A) = %i\n", dspace.get_rank());
        for(int i = 0; i < dspace.get_rank(); ++i)
        {
            printf("dims[%i] = %llu\n", i, dims[i]);
        }
        printf("class(A) = %s\n", dtype.get_class_string());
        printf("datatype size = %lu bytes\n", dtype.get_size_bytes());
        printf("num_elements(A) = %llu\n", dspace.get_num_elements());
        printf("buffer size = %lu bytes\n", dset.get_buffer_size_bytes());
        std::vector<float> data(dspace.get_num_elements());
        dset.read(data.data());
        for(size_t i = 0; i < data.size(); ++i)
        {
            printf("data[%lu] = %f\n", i, data[i]);
        }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
    }
#endif
#if 0
    cuphy::tensor_pinned tDataSymLocCpu(cuphy::tensor_info(CUPHY_R_8U, {12}),
                                        cuphy::tensor_flags::align_tight);
    uint8_t* pDataSymLoc = static_cast<uint8_t*>(tDataSymLocCpu.addr());
    for(int i = 0; i < 12; ++i)
    {
        pDataSymLoc[i] = static_cast<uint8_t>(i);
    }
    cuphy::tensor_device tDataSymLocGpu(cuphy::tensor_info(CUPHY_R_8U, {12}),
                                        cuphy::tensor_flags::align_tight);
    tDataSymLocGpu = tDataSymLocCpu;

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> check(12);
    cudaError_t e = cudaMemcpy(check.addr(), tDataSymLocGpu.addr(), 12 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if(cudaSuccess != e)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: cudaMemcpy() returned {}", cudaGetErrorString(e));
    }
    for(int i = 0; i < 12; ++i)
    {
        printf("check[%i] = %u\n", i, check.addr()[i]);
    }
#endif
#if 0
    cuphy::tensor_pinned tDataSymLocCpu(cuphy::tensor_info(CUPHY_R_8U, {12}),
                                        cuphy::tensor_flags::align_tight);
    uint8_t* pDataSymLoc = static_cast<uint8_t*>(tDataSymLocCpu.addr());
    for(int i = 0; i < 12; ++i)
    {
        pDataSymLoc[i] = static_cast<uint8_t>(i);
    }
    struct test_struct
    {
        cuphy::tensor_pinned tDataSymLocCpu;
    };
    test_struct s;
    s.tDataSymLocCpu = tDataSymLocCpu;
#endif
#if 0
    typedef cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc> tensor_pinned_R_8U;
    tensor_pinned_R_8U my_tensor(cuphy::tensor_layout({32}));
    my_tensor(11) = 33;
#endif
#if 0
    hdf5hpp::hdf5_file    f         = hdf5hpp::hdf5_file::open("../../cuPHY_data/perf/TV_cuphy_F14-US-01.1_snrdb40.00_MIMO8x16_PRB272_DataSyms10_qam256.h5");
    hdf5hpp::hdf5_dataset dset_gnb  = f.open_dataset("gnb_pars");
    gnb_pars              gpars     = cuphy::gnb_pars_from_dataset_elem(dset_gnb[0]);
    printf("gnb_pars:\n");
    printf("    fc                   = %u\n", gpars.fc);
    printf("    mu                   = %u\n", gpars.mu);
    printf("    nRx                  = %u\n", gpars.nRx);
    printf("    nPrb                 = %u\n", gpars.nPrb);
    printf("    cellId               = %u\n", gpars.cellId);
    printf("    slotNumber           = %u\n", gpars.slotNumber);
    printf("    Nf                   = %u\n", gpars.Nf);
    printf("    Nt                   = %u\n", gpars.Nt);
    printf("    df                   = %u\n", gpars.df);
    printf("    dt                   = %u\n", gpars.dt);
    printf("    numBsAnt             = %u\n", gpars.numBsAnt);
    printf("    numBbuLayers         = %u\n", gpars.numBbuLayers);
    printf("    numTb                = %u\n", gpars.numTb);
    printf("    ldpcnIterations      = %u\n", gpars.ldpcnIterations);
    printf("    ldpcEarlyTermination = %u\n", gpars.ldpcEarlyTermination);
    printf("    ldpcAlgoIndex        = %u\n", gpars.ldpcAlgoIndex);
    printf("    ldpcFlags            = %u\n", gpars.ldpcFlags);
    printf("    ldpcUseHalf          = %u\n", gpars.ldpcUseHalf);
    
    hdf5hpp::hdf5_dataset         dset_tb  = f.open_dataset("tb_pars");
    for(size_t i = 0; i < dset_tb.get_num_elements(); ++i)
    {
        tb_pars pars = cuphy::tb_pars_from_dataset_elem(dset_tb[i]);
        printf("tb_pars[%lu]:\n", i);
        printf("    nRnti            = %u\n",  pars.nRnti);
        printf("    numLayers        = %u\n",  pars.numLayers);
        printf("    layerMap         = %lu\n", pars.layerMap);
        printf("    startPrb         = %u\n",  pars.startPrb);
        printf("    numPrb           = %u\n",  pars.numPrb);
        printf("    startSym         = %u\n",  pars.startSym);
        printf("    numSym           = %u\n",  pars.numSym);
        printf("    dataScramId      = %u\n",  pars.dataScramId);
        printf("    mcsTableIndex    = %u\n",  pars.mcsTableIndex);
        printf("    mcsIndex         = %u\n",  pars.mcsIndex);
        printf("    rv               = %u\n",  pars.rv);
        printf("    dmrsType         = %u\n",  pars.dmrsType);
        printf("    dmrsAddlPosition = %u\n",  pars.dmrsAddlPosition);
        printf("    dmrsMaxLength    = %u\n",  pars.dmrsMaxLength);
        printf("    dmrsScramId      = %u\n",  pars.dmrsScramId);
        printf("    dmrsEnergy       = %u\n",  pars.dmrsEnergy);
        printf("    dmrsCfg          = %u\n",  pars.dmrsCfg);
        printf("    nSCID            = %u\n",  pars.nSCID);
    }
#endif
#if 0
    const int N_TEST_DESCRS=3;
    cuphy::kernelDescrs<N_TEST_DESCRS> test_descrs("TestDescr");

    std::array<size_t, N_TEST_DESCRS> dynDescrSizeBytes{3, 2, 20};
    std::array<size_t, N_TEST_DESCRS> dynDescrAlignBytes{1, 1, 4};

    test_descrs.alloc(dynDescrSizeBytes, dynDescrAlignBytes);
    test_descrs.displayDescrSizes();

    // Confirm proper alignment
    for (int i = 0;i < dynDescrSizeBytes.size(); i++) {

        if((reinterpret_cast<uintptr_t>(test_descrs.getCpuStartAddrs()[i]) % dynDescrAlignBytes[i]) != 0)
            printf("Error! Mismatched alignment for i=%d. Addr %p not divisible by %lu\n",
                    i, test_descrs.getCpuStartAddrs()[i],
                    dynDescrAlignBytes[i]);
            //return 1;
        printf("i=%d:  Addr %p modulo %lu = %lu\n",
                    i, test_descrs.getCpuStartAddrs()[i],
                    dynDescrAlignBytes[i],
                    (reinterpret_cast<uintptr_t>(test_descrs.getCpuStartAddrs()[i]) % dynDescrAlignBytes[i]));
    }
#endif

    printf("%s execution completed.\n", argv[0]);
    NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "{} execution completed", argv[0]);
    return 0;
}
