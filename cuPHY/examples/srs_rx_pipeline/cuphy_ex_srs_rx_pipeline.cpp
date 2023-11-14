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
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "srs_rx.hpp"


#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir



/////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("srs_chEst [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
    printf("    -o  Output HDFS debug file\n");
    printf("    -G\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("srs_rx.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::vector<std::string> inputFilenameVec;
        std::string outputFilename = std::string();

        uint64_t procModeBmsk = 0;

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilenameVec.push_back(argv[iArg++]);
                    break;
                case 'o':
                    if(++iArg < argc)
                    {
                        outputFilename.assign(argv[iArg++]);
                    }
                    break;
                case 'G':
                    ++iArg;
                    procModeBmsk = SRS_PROC_MODE_FULL_SLOT_GRAPHS;
                    NVLOGI_FMT(NVLOG_SRS, "CUDA graph enabled!");
                    break;
                default:
                    NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilenameVec.empty())
        {
            usage();
            exit(1);
        }

        // ---------------------------------------------------------------
        // Initialize main stream

        cuphy::stream cuStrmMain;

        //-----------------------------------------------------------------
        // Open debug file

        std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
        if(!outputFilename.empty())
        {
            debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
        }

        //-----------------------------------------------------------------
        // Initialize GPU memory

        size_t max_nSrsUes        =  1000;      
        size_t max_rbSnr_mem      =  max_nSrsUes * sizeof(float);
        size_t max_srsReport_mem  =  max_nSrsUes * sizeof(cuphySrsReport_t);
        size_t max_mem            =  max_rbSnr_mem + max_srsReport_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //------------------------------------------------------------------
        // Load API parameters

        srsStaticApiDataset  srsStaticApiDataset(inputFilenameVec, cuStrmMain.handle(), outputFilename);
        srsDynApiDataset     srsDynApiDataset(inputFilenameVec,    cuStrmMain.handle(), procModeBmsk);
        srsEvalDataset       srsEvalDataset(inputFilenameVec,      cuStrmMain.handle());  


        //------------------------------------------------------------------
        // Create srs reciever object

        cuphySrsRxHndl_t srsRxHndl;
        
        cuphyStatus_t statusCreate = cuphyCreateSrsRx(&srsRxHndl, &srsStaticApiDataset.srsStatPrms, cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // Setup srs reciever object

        cuphySrsBatchPrmHndl_t const batchPrmHndl = nullptr;  // batchPrms currently un-used

        cuphyStatus_t statusSetup  = cuphySetupSrsRx(srsRxHndl, &srsDynApiDataset.srsDynPrm, batchPrmHndl);
        if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);

        //------------------------------------------------------------------
        // Run srs reciever object

        cuphyStatus_t statusRun = cuphyRunSrsRx(srsRxHndl, procModeBmsk);
        if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

        //------------------------------------------------------------------
        // Evaluate results    

        cudaStreamSynchronize(cuStrmMain.handle());
        srsEvalDataset.evalSrsRx(srsDynApiDataset.srsDynPrm, srsDynApiDataset.tSrsChEstVec, srsDynApiDataset.dataOut.pRbSnrBuffer, srsDynApiDataset.dataOut.pSrsReports, cuStrmMain.handle());
    
        //------------------------------------------------------------------
        // Write debug output

        if(!outputFilename.empty())
        {
            cuphyStatus_t statusDebugWrite = cuphyWriteDbgBufSynchSrs(srsRxHndl, cuStrmMain.handle());
            cuStrmMain.synchronize();
            if(CUPHY_STATUS_SUCCESS != statusDebugWrite) throw cuphy::cuphy_exception(statusDebugWrite);
        }

        // --------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroySrsRx(srsRxHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
