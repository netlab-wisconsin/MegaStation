# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import pycuda.driver as drv

from .FDD.run import run_FDD
from .TDD.run import run_TDD


def measure(base, args):

    if os.geteuid() == 0:
        sudo = ""
    else:
        sudo = "sudo "

    drv.init()
    buffer = list(drv.Device(args.gpu).get_attributes().items())

    attributes = {}
    for k, v in buffer:
        attributes[str(k)] = v

    sms = attributes["MULTIPROCESSOR_COUNT"]

    if args.is_power:
        if os.path.exists("power.txt"):
            os.remove("power.txt")

    try:
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -lgc {args.freq}")

        if args.power is not None:
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -pl {args.power}")

        if "FDD" in args.uc:
            run_FDD(args, sms)
        else:
            run_TDD(args, sms)

    finally:
        if not args.is_no_mps and args.debug_mode not in ["ncu"]:
            os.system(
                f"echo quit | CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control"
            )
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -rgc")
