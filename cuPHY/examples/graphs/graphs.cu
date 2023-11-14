/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


////////////////////////////////////////////////////////////////////////
// usage()

#include <fstream>
#include <string>
#include <vector>
#include <list>
#include "cuphy.h"
#include "cuphy.hpp"
#include "util.hpp"
#include "pusch_rx.hpp"


void usage()
{
    printf("streams [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -i  input_filename     Input HDF5 filename\n");
    printf("    -g  GPU Id             GPU Id used to run all the pipelines\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    -r  # of iterations    Number of run iterations to run\n");
    printf("    -s  # of streams       Number of streams to be run\n");
    printf("    --H                    Use half precision (FP16) for back-end\n");
}
////////////////////////////////////////////////////////////////////////

// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("graph.log");
    try
    {
    //     //------------------------------------------------------------------
    //     // Parse command line arguments
    //     int         iArg = 1;
    //     std::string inputFileName;
    //     std::string outputFileName;
    //     int32_t     gpuId               = 0;
    //     uint32_t    nIterations         = 1000;
    //     uint32_t    fp16Mode            = 1;
    //     uint32_t    nStreams            = 1;
    //     int32_t     nGPUs               = 0;
    //     bool        fileBased           = false;
    //     uint32_t    circularBufferSize  = 1;

    //     CUDA_CHECK(cudaGetDeviceCount(&nGPUs));

    //     while(iArg < argc)
    //     {
    //         if('-' == argv[iArg][0])
    //         {
    //             switch(argv[iArg][1])
    //             {
    //             case 'b':
    //                 circularBufferSize = std::stoi(argv[++iArg]);
    //                 ++iArg;
    //                 break;
    //             case 'f':
    //                 ++iArg;
    //                 fileBased = true;
    //                 break;
    //             case 'i':
    //                 if(++iArg >= argc)
    //                 {
    //                     NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
    //                 }
    //                 inputFileName.assign(argv[iArg++]);
    //                 break;
    //             case 'h':
    //                 usage();
    //                 exit(0);
    //                 break;
    //             case 'r':
    //                 if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nIterations)) || ((nIterations <= 0)))
    //                 {
    //                     NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of run iterations");
    //                     exit(1);
    //                 }
    //                 ++iArg;
    //                 break;
    //             case 's':
    //                 nStreams = std::stoi(argv[++iArg]);
    //                 ++iArg;
    //                 break;
    //             case 'g':
    //                 if((++iArg >= argc) ||
    //                    (1 != sscanf(argv[iArg], "%i", &gpuId)) ||
    //                    ((gpuId < 0) || (gpuId >= nGPUs)))
    //                 {
    //                     NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid GPU Id (should be within [0,{}])", nGPUs - 1);
    //                     exit(1);
    //                 }
    //                 ++iArg;
    //                 break;
    //             case 'o':
    //                 if(++iArg >= argc)
    //                 {
    //                     NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: No output file name given");
    //                 }
    //                 outputFileName.assign(argv[iArg++]);
    //                 break;
    //             case '-':
    //                 switch(argv[iArg][2])
    //                 {
    //                 case 'H':
    //                     if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &fp16Mode)) || (1 < fp16Mode))
    //                     {
    //                         NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid FP16 mode {}", fp16Mode);
    //                         exit(1);
    //                     }
    //                     ++iArg;
    //                     break;
    //                 default:
    //                     NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
    //                     usage();
    //                     exit(1);
    //                     break;
    //                 }
    //                 break;
    //             default:
    //                 NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
    //                 usage();
    //                 exit(1);
    //                 break;
    //             }
    //         }
    //         else // if('-' == argv[iArg][0])
    //         {
    //             NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
    //             exit(1);
    //         }
    //     } // while (iArg < argc)
    //     if(inputFileName.empty())
    //     {
    //         usage();
    //         exit(1);
    //     }

    //     cudaSetDevice(gpuId);

    //     // input file names
    //     std::vector<std::string> inputFileNameVec;
    //     if(fileBased) 
    //     {
    //         nStreams = 0;
    //         std::fstream fl(inputFileName.c_str());
    //         std::string fName;

    //         while(fl >> fName)
    //         {
    //             inputFileNameVec.emplace_back(fName);
    //             ++nStreams;
    //         }
    //     }else
    //     {
    //         for(int i = 0; i < nStreams; ++i)
    //             inputFileNameVec.emplace_back(inputFileName);
    //     }

    //     // main stream
    //     cudaStream_t cuStream;
    //     cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking);

    //     //output file names (For now only 0'th pipeline)
    //     std::vector<std::string> outputFileNameVec(nStreams);


    //     //read datasets
    //     std::vector<EvalDataset>                 evalDatasetVec(nStreams);
    //     std::vector<StaticApiDataset>            staticApiDatatsetVec(nStreams);
    //     std::vector<std::vector<DynApiDataset>>  dynApiDatatsetVec(nStreams);
        
    //     for(int i = 0; i < nStreams; ++i)
    //     {
    //         evalDatasetVec[i] = std::move(EvalDataset(inputFileNameVec[i], cuStream));
    //         staticApiDatatsetVec[i] = std::move(StaticApiDataset(inputFileNameVec[i], cuStream));
    //         dynApiDatatsetVec[i].resize(circularBufferSize);
            
    //         for (int j = 0; j < circularBufferSize; ++j)
    //             dynApiDatatsetVec[i][j] = std::move(DynApiDataset(inputFileNameVec[i], cuStream));
    //     }

    //     // create/setup pipelines
    //     std::vector<std::vector<cuphyPuschRxHndl_t>> pushRxHndleVec(nStreams);
    //     for(int i = 0; i < nStreams; ++i)
    //     {
    //         pushRxHndleVec[i].resize(circularBufferSize);

    //         for(int j = 0; j < circularBufferSize; ++j)
    //         {
    //             cuphyCreatePuschRx(&pushRxHndleVec[i][j], &staticApiDatatsetVec[i].puschStatPrms, cuStream);
    
                
    //         cuphyPuschBatchPrmHndl_t     batchPrmHndl0 = nullptr; 
    //         cuphySetupPuschRx(pushRxHndleVec[i][j], &dynApiDatatsetVec[i][j].puschDynPrm, batchPrmHndl0, cuStream);
    //         }
    //     }

    //     // setup streams
    //     std::vector<cudaStream_t>   streamsArray(nStreams);
    //     std::vector<cudaEvent_t>    stopEventsArray(nStreams);

    //     for(int i = 0; i < nStreams; i++)
    //     {
    //         // create streams
    //         cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamNonBlocking);
    //         // create stopEvents
    //         cudaEventCreateWithFlags(&stopEventsArray[i], cudaEventDisableTiming);
    //     }
     

    //     cudaEvent_t stopTime, startTime;
    //     CUDA_CHECK(cudaEventCreateWithFlags(&stopTime, cudaEventBlockingSync));
    //     CUDA_CHECK(cudaEventCreateWithFlags(&startTime, cudaEventBlockingSync));
    //     cudaEvent_t startStreams;
    //     CUDA_CHECK(cudaEventCreateWithFlags(&startStreams, cudaEventDisableTiming));
    //     cudaGraph_t     graph;
    //     cudaGraphExec_t exeGraph;

    //     // capture stream
    //     cudaStreamBeginCapture(streamsArray[0], cudaStreamCaptureModeGlobal);

    //     gpu_ms_delay(0, gpuId, streamsArray[0]);

    //     CUDA_CHECK(cudaEventRecord(startStreams, streamsArray[0]));
    //     // Create start and stop events
    //     // Wait for start event and enqueue Pusch run calls
    //     for(int i = 0; i < nStreams; i++)
    //     {
    //         if(i != 0)
    //             CUDA_CHECK(cudaStreamWaitEvent(streamsArray[i], startStreams, 0));

    //             cuphyRunPuschRx(pushRxHndleVec[i][0],  0, streamsArray[i]);

    //         if(i != 0)
    //         {
    //             CUDA_CHECK(cudaEventRecord(stopEventsArray[i], streamsArray[i]));
    //             CUDA_CHECK(cudaStreamWaitEvent(streamsArray[0], stopEventsArray[i], 0));
    //         }

    //     }

    //     // Wait for start event and enqueue Pusch run calls

    //     CUDA_CHECK(cudaStreamEndCapture(streamsArray[0], &graph));

    //     cudaGraphInstantiate(&exeGraph, graph, NULL, NULL, 0);

    //     float totalTime = 0.0;

    //     TimePoint startCPUTime = Clock::now();

    //     CUDA_CHECK(cudaEventRecord(startTime, streamsArray[0]));

    //     // CUDA_CHECK(cudaGraphLaunch(exeGraph, streamsArray[0]));
    //     for(uint32_t it = 0; it < nIterations; ++it)
    //     {
    //         CUDA_CHECK(cudaGraphLaunch(exeGraph, streamsArray[0]));
    //     }
    //     CUDA_CHECK(cudaEventRecord(stopTime, streamsArray[0]));
    //     CUDA_CHECK(cudaDeviceSynchronize());

    //     TimePoint                   stopCPUTime = Clock::now();
    //     duration<float, std::micro> diff        = stopCPUTime - startCPUTime;

    //     float elapsedMs = 0.0f;
    //     cudaEventElapsedTime(&elapsedMs, startTime, stopTime);
    //     totalTime += elapsedMs;
    //     printf("Average GPU execution time: %4.4f usec (over %d runs, using CUDA events)\nAverage CPU execution time: %4.4f\n",
    //            totalTime * 1000 / nIterations,
    //            nIterations,
    //            diff.count() / nIterations);

    //     // copy output buffers to CPU
    //     for(int i = 0; i < nStreams; i++)
    //     {
    //         cuphyWriteDbgBufSynch(pushRxHndleVec[i][0], cuStream);
    //     }

 

    //     CUDA_CHECK(cudaDeviceSynchronize());

    //     for (int i = 0; i < nStreams; i++)
    //     {
    //          DisplayBler(evalDatasetVec[i], dynApiDatatsetVec[i][0]);
    //     }

        
    //     CUDA_CHECK(cudaEventDestroy(startTime));
    //     CUDA_CHECK(cudaEventDestroy(stopTime));
    //     CUDA_CHECK(cudaEventDestroy(startStreams));

    //     for(int i = 0; i < nStreams; i++)
    //     {
    //         CUDA_CHECK(cudaEventDestroy(stopEventsArray[i]));
    //         CUDA_CHECK(cudaStreamDestroy(streamsArray[i]));
    //     }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }

    return returnValue;
}
////////////////////////////////////////////////////////////////////////
