# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import glob
from functools import partial

import shutil
from multiprocessing import Pool

from .FDD.run import run_FDD
from .TDD.run import run_TDD


def measure(base, args):

    if os.geteuid() == 0:
        sudo = ""
    else:
        sudo = "sudo "

    os.system(f"{sudo}nvidia-smi -i {args.gpu} -lgc {args.freq}")

    if args.power is not None:
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -pl {args.power}")

    os.system(f"{sudo}nvidia-smi mig -i {args.gpu} -lgip >buffer.txt")

    ifile = open("buffer.txt", "r")
    lines = ifile.readlines()
    ifile.close()
    os.remove("buffer.txt")

    mig_ids = {}

    if args.mig_instances > 1:
        if args.is_debug:
            base.error("Debug mode cannot be run with maximal MIG parallelization")

    for line in lines:
        lst = line.split()

        if len(lst) > 2 and lst[2] == "MIG" and int(lst[1]) == args.gpu:
            mig_ids[int(lst[4])] = (int(lst[5].split("/")[0]), int(lst[8]))

    sms = mig_ids[args.mig][1]

    if args.mig_instances > mig_ids[args.mig][0]:
        print(
            f"Warning: the HW can provide at most {mig_ids[args.mig][0]} GPU instances for the desired configuration"
        )

    number_of_instances = np.min([mig_ids[args.mig][0], args.mig_instances])

    if number_of_instances == 0:
        base.error("no HW resources available for the desidered configuration")

    instances = [str(args.mig)] * number_of_instances

    for instance in instances:

        os.system(f"{sudo}nvidia-smi mig -i {args.gpu} -cgi {instance} > buffer.txt")

        ifile = open("buffer.txt")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        if len(lines) != 1:
            raise SystemError

        if "Successfully created GPU instance ID" not in lines[0]:
            raise SystemError

        lst = lines[0].split()

        gi_id = int(lst[5])

        os.system(
            f"{sudo}nvidia-smi mig -i {args.gpu} -gi {gi_id} --list-compute-instance-profiles >buffer.txt"
        )

        ifile = open("buffer.txt", "r")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        gi_ids = {}

        for line in lines:
            lst = line.split()

            if len(lst) > 2 and lst[3] == "MIG" and int(lst[1]) == args.gpu:
                gi_ids[int(lst[7])] = lst[5].replace("*", "")

        gi_profile = gi_ids[sms]

        os.system(
            f"{sudo}nvidia-smi mig -i {args.gpu} -cci {gi_profile} -gi {gi_id} >buffer.txt"
        )

        ifile = open("buffer.txt")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        if len(lines) != 1:
            raise SystemError

        if "Successfully created compute instance ID" not in lines[0]:
            raise SystemError

    os.system(f"{sudo}nvidia-smi -L >buffer.txt")
    ifile = open("buffer.txt", "r")
    lines = ifile.readlines()
    ifile.close()
    os.remove("buffer.txt")

    uuid = []

    for odx, line in enumerate(lines):
        lst = line.split()
        if lst[0] == "GPU" and int(lst[1].replace(":", "")) == args.gpu:
            for idx in range(1, number_of_instances + 1):
                uuid.append(lines[odx + idx].split()[-1].replace(")", ""))
            break

    if args.is_power:
        if os.path.exists("power.txt"):
            os.remove("power.txt")

    pMIGS = glob.glob("MIG*")

    for pMIG in pMIGS:
        if os.path.isdir(pMIG):
            shutil.rmtree(pMIG, ignore_errors=True)

    try:
        if "FDD" in args.uc:
            pool = Pool(number_of_instances)
            functor = partial(run_FDD, args, sms)
            pool.map(functor, uuid)
        else:
            if number_of_instances == 1:
                run_TDD(args, sms, uuid[0])
            else:
                raise NotImplementedError
    finally:
        if not args.is_no_mps and args.debug_mode not in ["ncu"]:
            for gpu_instance in uuid:
                gpu_instance_folder = gpu_instance.replace("/", "-")
                os.system(
                    f"echo quit | CUDA_VISIBLE_DEVICES={gpu_instance} CUDA_MPS_PIPE_DIRECTORY={gpu_instance_folder} CUDA_LOG_DIRECTORY={gpu_instance_folder} nvidia-cuda-mps-control"
                )
                shutil.rmtree(gpu_instance_folder, ignore_errors=True)
                if args.is_test:
                    print(f"Removed: {gpu_instance_folder}")

        result = -1
        counter = 1

        while result != 0:
            print(f"Attempts to disable MIG: {counter}")
            result = os.system(f"{sudo}nvidia-smi -i {args.gpu} -mig 0 >/dev/null")
            counter += 1

        os.system(f"{sudo}nvidia-smi -i {args.gpu} -rgc")
