# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Python bindings for cuPHY."""
# Load libcuphy.so.
import ctypes
import os
libcuphy_path = os.path.dirname(os.path.realpath(__file__))
ctypes.cdll.LoadLibrary(os.path.join(libcuphy_path, "libfmt.so.9"))
ctypes.cdll.LoadLibrary(os.path.join(libcuphy_path, "libcuphy.so"))

# Import everything here to they can be imported directly from pycuphy.
from ._pycuphy import *  # type: ignore
from .types import *
from .util import *
from .params import *
from .chest_filters import *
