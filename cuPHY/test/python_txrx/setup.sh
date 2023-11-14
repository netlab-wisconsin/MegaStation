#!/bin/bash
#  Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

# NOTE: This script should be called via "source setup.sh"

if [ ! -v AERIAL_PYTHON ]; then
    export AERIAL_PYTHON=1
    export PS1="[Aerial Python]$PS1 "
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/MATLAB/MATLAB_Runtime/v911/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v911/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v911/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v911/extern/bin/glnxa64
fi
