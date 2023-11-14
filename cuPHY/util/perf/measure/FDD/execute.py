# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import subprocess
from .parse_power import parse_power


def run(args, designated, mig, mig_gpu, command, vectors, mode, target, k):

    results = None

    subs, streams, steps = designated

    if args.force is not None:
        if args.force == 0:
            connections = streams
        else:
            connections = args.force
    else:
        connections = np.min([32, int(np.power(2, np.floor(np.log2(96 / (subs + 1)))))])

    if args.is_power:
        from .configure_power import configure

        system = configure(
            args,
            designated,
            mig,
            mig_gpu,
            connections,
            command,
            vectors,
            mode,
            k,
            target,
        )
        ofile = None

        if not os.path.exists("power.txt"):

            if args.is_test:
                print(
                    " ".join(
                        [
                            "nvidia-smi",
                            "-i",
                            f"{args.gpu}",
                            "--query-gpu=clocks.sm,clocks.mem,power.draw,memory.used",
                            "-lms",
                            "10",
                            "--format=csv",
                        ]
                    )
                )
            else:
                ofile = open("power.txt", "w")
                proc = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "-i",
                        f"{args.gpu}",
                        "--query-gpu=clocks.sm,clocks.mem,power.draw,memory.used",
                        "-lms",
                        "10",
                        "--format=csv",
                    ],
                    stdout=ofile,
                )

        if args.is_test:
            print(system)
            if not args.is_save_buffers:
                os.remove(vectors)
        else:
            if args.is_unsafe:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)
            else:
                buffer = system.split(args.cfld)[0].strip().split()
                env = {}

                for itm in buffer:
                    mapping = itm.split("=")
                    env[mapping[0]] = mapping[1]

                cmd = args.cfld + system.split(args.cfld)[-1].strip()
                cmd, stdout = cmd.split(">")

                ofile = open(stdout, "w")

                try:
                    mproc = subprocess.Popen(cmd.split(), env=env, stdout=ofile)
                    mproc.wait(100)
                except subprocess.TimeoutExpired:
                    mproc.kill()
                finally:
                    ofile.close()
                    if not args.is_save_buffers:
                        os.remove(vectors)

        if ofile is not None:
            proc.kill()
            ofile.close()

            ifile = open("power.txt", "r")
            lines = ifile.readlines()
            ifile.close()
            os.remove("power.txt")

            if mig is None:
                os.remove(f"buffer-{str(k).zfill(2)}.txt")
            else:
                os.remove(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt")

            results = parse_power(lines)

    else:
        if args.is_debug:
            from .configure_debug import configure

            system = configure(
                args,
                designated,
                mig,
                mig_gpu,
                connections,
                command,
                vectors,
                mode,
                k,
                target,
            )
            if args.is_test:
                print(system)
                if not args.is_save_buffers:
                    os.remove(vectors)
            else:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)

        else:
            from .configure import configure

            system = configure(
                args,
                designated,
                mig,
                mig_gpu,
                connections,
                command,
                vectors,
                mode,
                k,
                target,
            )
            if args.is_test:
                print(system)
                if not args.is_save_buffers:
                    os.remove(vectors)
            else:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)

                if mig is None:
                    ifile = open(f"buffer-{str(k).zfill(2)}.txt", "r")
                    lines = ifile.readlines()
                    ifile.close()
                    if not args.is_save_buffers:
                        os.remove(f"buffer-{str(k).zfill(2)}.txt")
                else:
                    ifile = open(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt", "r")
                    lines = ifile.readlines()
                    ifile.close()
                    if not args.is_save_buffers:
                        os.remove(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt")

                if args.is_check_traffic:
                    from ..error import parse
                else:
                    from .sweep import parse

                results = parse(args, lines)

    return results
