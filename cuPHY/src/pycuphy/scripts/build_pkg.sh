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
CUPHY_ROOT=$(realpath $PROJECT_ROOT/../..)
echo $SCRIPT starting...
cd $PROJECT_ROOT

# Define BUILD_ID if it does not exist yet.
if [ -z "${BUILD_ID}" ]; then
    export BUILD_ID=1
fi

# Copy the .so files so we can include them in the dist.
rm -f $PROJECT_ROOT/src/pycuphy/*.so
cp $CUPHY_ROOT/build/src/pycuphy/_pycuphy*.so $PROJECT_ROOT/src/pycuphy
cp $CUPHY_ROOT/build/src/cuphy/libcuphy.so $PROJECT_ROOT/src/pycuphy
cp $CUPHY_ROOT/build/external/fmtlog_flat/fmt/libfmt.so.9 $PROJECT_ROOT/src/pycuphy

# Build the pycuphy package.
rm -rf dist
python3 -m build

# Finished
echo $SCRIPT finished.
