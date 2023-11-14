#  Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Setup file for pycuphy package."""
import os
import sys
import setuptools

sys.path.insert(0, os.path.abspath("./src/pycuphy"))
import version_pycuphy

version = version_pycuphy.release

# Note: The .so files need to be placed under src/pycuphy/lib.
setuptools.setup(
    name="pycuphy",
    version=version,
    author="NVIDIA",
    description="Aerial cuPHY Python API",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={"": [
        "version_pycuphy.py",
        "_pycuphy.cpython-38-x86_64-linux-gnu.so",
        "_pycuphy.cpython-310-x86_64-linux-gnu.so",
        "libcuphy.so",
        "libfmt.so.9"
    ]},
    python_requires=">=3.7",
    zip_safe=False,
)
