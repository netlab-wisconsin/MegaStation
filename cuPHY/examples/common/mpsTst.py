#!/usr/bin/env python3
# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import os
import subprocess
#import re
from subprocess import check_output

# Parse arguments
argsParser = argparse.ArgumentParser(description='Process inputs.')
argsParser.add_argument('--gpuid' , type=int, dest='gpuId' , help='GPU ordinal', required=True)
argsParser.add_argument('--smpct' , type=int, dest='smPct' , help='SM allocation percentage per sub-context', required=True)
argsParser.add_argument('--tstcmd', type=str, dest='tstCmd', help='Test command', required=True)
args = argsParser.parse_args()

try:
    # Setup MPS pipe and log paths
    from pathlib import Path
    homeDirPath = str(Path.home())
    pipeDirPath = os.path.abspath(os.path.join(homeDirPath, 'mpsPipeDir'))
    logsDirPath = os.path.abspath(os.path.join(homeDirPath, 'cudaLogs'))
    
    if not os.path.exists(pipeDirPath):
        os.makedirs(pipeDirPath)
    
    if not os.path.exists(logsDirPath):
        os.makedirs(logsDirPath)
    
    # Extract GPU UUID
    cmd = f'sudo nvidia-smi -i {args.gpuId} -q'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    resultByteList = result.stdout.split()
    uuidPos = resultByteList.index(b'UUID')
    #print(resultByteList[uuidPos+2])
    gpuUUID = resultByteList[uuidPos+2].decode("utf-8")

    # Extract user id
    userid = os.getuid()
    #print(f'userid is {userid}')
    
     # Setup environment variables
    envVar = (f"CUDA_VISIBLE_DEVICES={gpuUUID} CUDA_MPS_PIPE_DIRECTORY={pipeDirPath} CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={args.smPct} "
              f"CUDA_LOG_DIRECTORY={logsDirPath} CUDA_MPS_LOG_DIRECTORY={logsDirPath}")

    #print(envVar)
    
    # Start MPS control daemon in background (it observes the value set by CUDA_MPS_ACTIVE_THREAD_PERCENTAGE)
    os.system(f'sudo {envVar} nvidia-smi -i {args.gpuId} -c EXCLUSIVE_PROCESS')
    os.system(f'sudo {envVar} nvidia-cuda-mps-control -d')
    os.system(f'echo start_server -uid {userid} | sudo {envVar} nvidia-cuda-mps-control')
    
    # Read MPS server pid
    cmd = f'pidof nvidia-cuda-mps-server'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    resultByteList = result.stdout.split()
    #print(resultByteList)
    mpsServerPid = resultByteList[0].decode("utf-8")

    """mpsServerPidByteList = check_output(["pidof", "nvidia-cuda-mps-server"])
    mpsServerPid = mpsServerPidByteList.decode("utf-8")
    print(mpsServerPidByteList)
    print(mpsServerPid)"""

    # Run test
    os.system(f'sudo {envVar} MPS_SERVER_PID={mpsServerPid} {args.tstCmd}')

finally:
    os.system(f'echo quit | sudo {envVar} nvidia-cuda-mps-control')
    os.system(f'sudo {envVar} nvidia-smi -i {args.gpuId} -c DEFAULT')