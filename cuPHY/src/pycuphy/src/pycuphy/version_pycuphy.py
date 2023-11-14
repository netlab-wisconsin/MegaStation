# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Provides pycuphy version and release numbering."""
import os

# Check if BUILD_ID environment variable exists - from Jenkins
BUILD_ID = os.environ.get("BUILD_ID")
if BUILD_ID is None:
    raise RuntimeError("Environment variable BUILD_ID must be set")

BUILD_TYPE = os.environ.get("BUILD_TYPE")
if BUILD_TYPE is None:
    BUILD_TYPE = "dev"

if BUILD_TYPE == "rel":
    BUILD_TYPE = ""  # don't use 'rel' in the build string

# The short X.Y version
version = "0.20233"  # pylint: disable=invalid-name

# Create release version according to https://www.python.org/dev/peps/pep-0440/
# The full version, including alpha/beta/rc tags
release = f"{version}.{BUILD_ID}.{BUILD_TYPE}{BUILD_ID}"
