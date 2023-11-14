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
#include "cuphy_channels.hpp"
#include "pucch_rx.hpp"
#include "gen_pucch_perf_curve.hpp"

#include <fstream>
#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir
using namespace std;

/////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("pucch perf curve [options]\n");
    printf("  Options:\n");
    printf("  -i  Pucch Input HDF5 filename\n");
    printf("  -a  start Snr \n");
    printf("  -b  end Snr \n");
    printf("  -s  Snr step size \n");
    printf("  -m  test mode \n");
    printf("  -y  Pucch format \n");
    printf("  -r  number of iterations per snr step \n");
    printf("  -t  HDF5 filename to store perf curve \n");
    printf("  -o  HDF5 filename to pucch debug buffer \n");
    printf("  -f  input text file \n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    
    cuphyNvlogFmtHelper nvlog_fmt("pucch_bler_curve.log");
    
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int          iArg = 1;
        std::string  pucchInputFilename  = std::string();
        std::string  pucchOutputFilename = std::string();
        std::string  perfOutFilename     = std::string();
        std::string  textFilename        = std::string();
        uint32_t     nItrPerSnr          = 1000;
        uint8_t      formatType          = 0;
        uint8_t      mode                = 1;
        uint8_t      seed                = 1;
        float        startSnr            = -10;
        float        endSnr              = 40;
        float        snrStepSize         = 1;
        bool         txtInputFlag        = false;

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    pucchInputFilename.assign(argv[iArg++]);
                    break;
                case 'o':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: No output file name given");
                    }
                    pucchOutputFilename.assign(argv[iArg++]);
                    break;
                case 't':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: No perf output file name given");
                    }
                    perfOutFilename.assign(argv[iArg++]);
                    break;
                case 'f':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: No text file name given");
                    }else
                    {
                        textFilename.assign(argv[iArg++]);
                        txtInputFlag = true;
                    }
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nItrPerSnr)) || ((nItrPerSnr <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of run iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'a':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%f", &startSnr)))
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid start SNR");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'b':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%f", &endSnr)))
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid end SNR");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'm':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%c", &mode)) || ((mode < 1)))
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid PUCCH test mode");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'y':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%c", &formatType)) || ((formatType < 0)))
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid PUCCH format type");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 's':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%f", &snrStepSize)) || ((snrStepSize <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid snr step size SNR");
                        exit(1);
                    }
                    ++iArg;
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }

        if(txtInputFlag)
        {
            std::ifstream perfCurveRequestFile(textFilename);
            std::string                     currentLine;
            std::vector<std::string>        tvFolderNames;
            std::vector<std::string>        tvFormat;
            std::vector<uint8_t>            tvMode;
            std::vector<uint32_t>           tvNumItr;
            std::vector<uint8_t>            tvSeed;
            std::vector<std::vector<float>> snrValues;

            if ( perfCurveRequestFile.is_open() ) {
                while ( perfCurveRequestFile ) {
                    std::getline (perfCurveRequestFile, currentLine);
                    if(!currentLine.empty())
                    {
                        size_t firstF               = currentLine.find("pucchFormat");
                        std::string format          = currentLine.substr(firstF + 12, 1);

                        size_t firstM               = currentLine.find("pucchTestMode");
                        std::string modeTemp        = currentLine.substr(firstM + 14, 1);
                        
                        size_t firstS               = currentLine.find("seed");
                        std::string seedTemp1       = currentLine.substr(firstS + 5, 1);
                        std::string seedTemp2       = currentLine.substr(firstS + 6, 1);
                        uint8_t seedtemp;
                        if (seedTemp2.compare("_") == 0) {
                            seedtemp = std::stoi(seedTemp1);
                        } else {
                            seedtemp = std::stoi(seedTemp1)*10 + std::stoi(seedTemp2);
                        }

                        size_t firstI          = currentLine.find("N_slots");
                        uint32_t NumItrTemp    = 0;
                        // assumptions's that 1000<= N_slots <= 99999
                        for (int dIdx = 0; dIdx < 5; dIdx++) {
                            std::string digitTemp = currentLine.substr(firstI + 8 + dIdx, 1);
                            if (digitTemp.compare(";") == 0) {
                                break;
                            } else {
                                NumItrTemp *= 10;
                                NumItrTemp += std::stoi(digitTemp);
                            }
                        }

                        std::vector<float> snrVec;

                        size_t firstEqLoc      = currentLine.find("=");
                        size_t firstComLoc     = currentLine.find(",");

                        std::string folderName = currentLine.substr(firstEqLoc + 1, firstComLoc - firstEqLoc - 1);

                        size_t floatStartLoc   = currentLine.find("[") + 1;
                        size_t floatEndLoc     = currentLine.find_first_of (",]", floatStartLoc + 1) - 1;

                        while((floatEndLoc >= 0) && (floatStartLoc >= 0))
                        {
                            std::string snrStr = currentLine.substr(floatStartLoc, floatEndLoc - floatStartLoc + 1);
                            float       snr    = std::stof(snrStr);
                            snrVec.emplace_back(snr);
                            std::string endChar  = currentLine.substr(floatEndLoc + 1,1);
                            if((endChar.compare("]") == 0))
                            {
                                break;
                            }else
                            {
                                floatStartLoc = floatEndLoc + 2;
                                floatEndLoc   = currentLine.find_first_of (",]", floatStartLoc + 1) - 1;
                            }
                        }

                        tvFormat.emplace_back(format);
                        tvMode.emplace_back(std::stoi(modeTemp));
                        tvFolderNames.emplace_back(folderName);
                        snrValues.emplace_back(snrVec);
                        tvSeed.emplace_back(seedtemp);
                        tvNumItr.emplace_back(NumItrTemp);
                    }
                }
            }

            size_t nPerfCurveRequests = tvFolderNames.size();
            
            for(int perfSimIdx = 0; perfSimIdx < nPerfCurveRequests; ++perfSimIdx)
            {
                std::string pucchH5FileName;
                formatType = std::stoi(tvFormat[perfSimIdx]);
                switch(formatType) {
                    case 0:
                    case 1:
                        pucchH5FileName = tvFolderNames[perfSimIdx] + "/TV_PUCCH_F" + tvFormat[perfSimIdx] + "_gNB_CUPHY_s0p20.h5";
                        break;
                    case 2:
                    case 3:
                        pucchH5FileName = tvFolderNames[perfSimIdx] + "/TV_PUCCH_F" + tvFormat[perfSimIdx] + "_gNB_CUPHY_s0p5.h5";
                        break;
                }
                std::string perfOutFilename  = tvFolderNames[perfSimIdx] + "/cuphyPerfCurve.h5";
                std::string pucchDbgFilename;

                nItrPerSnr = tvNumItr[perfSimIdx];
                mode = tvMode[perfSimIdx];
                seed = tvSeed[perfSimIdx];

                gen_pucch_perf_curve(pucchH5FileName, pucchDbgFilename, perfOutFilename, formatType, snrValues[perfSimIdx], nItrPerSnr, mode, seed);
            }
        } else {
            uint32_t nSnrSteps  = static_cast<uint32_t>((endSnr - startSnr) / snrStepSize);
            std::vector<float> snrVec(nSnrSteps);
            for(int snrStepIdx = 0; snrStepIdx < nSnrSteps; ++snrStepIdx)
            {
                snrVec[snrStepIdx] = startSnr + snrStepIdx * snrStepSize;
            }
            gen_pucch_perf_curve(pucchInputFilename, pucchOutputFilename, perfOutFilename, formatType, snrVec, nItrPerSnr, mode, seed);
        }
        
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}



