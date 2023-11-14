#!/bin/bash

#  Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

# Exit on first error
set -e

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
echo $SCRIPT_DIR
CUPHY_ROOT=$(realpath $PROJECT_ROOT/../..)
echo $SCRIPT starting...
cd $CUPHY_ROOT/src

# Install the pycuphy package in developer mode
pip3 install --prefix ~/.local -e pycuphy

# Finished
