# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import numpy as np
import os
import uuid
import sys

from .traffic import traffic_avg, traffic_het
from .execute import run
from .properties import auto_het_subs, auto_avg_subs


def run_FDD(args, sms, mig=None):

    ifile = open(args.config)
    config = json.load(ifile)
    ifile.close()

    ifile = open(args.uc)
    uc = json.load(ifile)
    ifile.close()

    if args.is_graph:
        output = "sweep_graphs_" + args.uc.replace("uc_", "").replace(
            "_TDD.json", ""
        ).replace("_FDD.json", "")
        mode = 1
    else:
        output = "sweep_streams_" + args.uc.replace("uc_", "").replace(
            "_TDD.json", ""
        ).replace("_FDD.json", "")
        mode = 0

    data_targets = None
    target = None

    if not args.is_no_mps:
        if args.target is not None:
            if args.target[0].isnumeric():
                target = int(np.round(100 * float(args.target[0]) / sms))
                output = args.target[0].zfill(3) + "_" + output
            else:
                ifile = open(args.target[0], "r")
                data_targets = json.load(ifile)
                ifile.close()
        else:
            raise NotImplementedError

    if mig is not None:
        mig_gpu = mig.replace("/", "-")
        output = output + "_" + mig_gpu
    else:
        mig_gpu = None

    if args.seed is not None:
        output = output + "_s" + str(args.seed) + "_" + str(uuid.uuid4())

    if not args.is_no_mps:
        if mig is None:
            system = f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=."
        else:
            os.mkdir(mig_gpu)

            if args.is_test:
                print(f"Created: {mig_gpu}")

            system = f"CUDA_VISIBLE_DEVICES={mig} CUDA_MPS_PIPE_DIRECTORY={mig_gpu} CUDA_LOG_DIRECTORY={mig_gpu}"

        system = " ".join([system, "nvidia-cuda-mps-control -d"])

        os.system(system)

    k = args.start

    sweeps = {}
    powers = {}

    if args.seed is not None:
        np.random.seed(args.seed)

    command = os.path.join(args.cfld, "examples/psch_rx_tx/cuphy_ex_psch_rx_tx")

    is_scan = args.subs is None

    while k <= args.cap:

        if mig is None:
            vectors = os.path.join(os.getcwd(), "vectors-" + str(k).zfill(2) + ".yaml")
        else:
            vectors = os.path.join(
                os.getcwd(), "vectors-" + mig_gpu + "-" + str(k).zfill(2) + ".yaml"
            )

        if "_het_" in args.uc:

            designated_subs = None
            designated_streams = None
            designated_steps = None

            if is_scan:

                avail_subs = sorted(auto_het_subs, reverse=True)

                for subs in avail_subs:

                    if k % subs == 0 and k // subs <= 32:
                        designated_subs = subs
                        designated_streams = k // subs
                        designated_steps = 1
                        break

            else:

                if k % args.subs == 0:
                    designated_subs = args.subs
                    designated_streams = k // args.subs
                    designated_steps = 1

            if designated_subs is None:
                k += 1
                continue
            else:

                if not args.is_no_mps and data_targets is not None:

                    key = str(k).zfill(2)
                    target = data_targets.get(key, None)

                    if target is None:
                        k += 1
                        continue
                    else:
                        target = int(np.round(100 * float(target) / sms))

                uc_keys = list(uc.keys())

                uc_dl = [x for x in uc_keys if "PDSCH" in x]

                if len(uc_dl) != 1:
                    sys.exit("error: use case file exhibits an unexpected structure")
                else:
                    uc_dl = uc_dl[0]

                uc_ul = [x for x in uc_keys if "PUSCH" in x]

                if len(uc_ul) != 1:
                    sys.exit("error: use case file exhibits an unexpected structure")
                else:
                    uc_ul = uc_ul[0]

                testcases_dl = uc[uc_dl]
                testcases_ul = uc[uc_ul]

                filenames_dl = config[uc_dl]
                filenames_ul = config[uc_ul]

                message = "Number of active cells: " + str(k)

                if target is not None:
                    message += (
                        "("
                        + str(2 * int(np.round(target * sms / 200)))
                        + ","
                        + str(designated_subs)
                        + ")"
                    )

                print(message)

                traffic_het(
                    args,
                    vectors,
                    k,
                    (testcases_dl, testcases_ul),
                    (filenames_dl, filenames_ul),
                )

                designated = (designated_subs, designated_streams, designated_steps)

                if args.is_power:
                    powers[str(k).zfill(2)] = run(
                        args,
                        designated,
                        mig,
                        mig_gpu,
                        command,
                        vectors,
                        mode,
                        target,
                        k,
                    )
                else:
                    sweeps[str(k).zfill(2)] = run(
                        args,
                        designated,
                        mig,
                        mig_gpu,
                        command,
                        vectors,
                        mode,
                        target,
                        k,
                    )

        elif "_avg_":

            interval = uc["Peak: " + str(k)]

            for subcase in interval.keys():

                uc_keys = list(interval[subcase].keys())

                uc_dl = [x for x in uc_keys if "PDSCH" in x]

                if len(uc_dl) != 1:
                    sys.exit("error: use case file exhibits an unexpected struture")
                else:
                    uc_dl = uc_dl[0]

                uc_ul = [x for x in uc_keys if "PUSCH" in x]

                if len(uc_ul) != 1:
                    sys.exit("error: use case file exhibits an unexpected struture")
                else:
                    uc_ul = uc_ul[0]

                testcases_dl = interval[subcase][uc_dl]
                testcases_ul = interval[subcase][uc_ul]

                filenames_dl = config[uc_dl]
                filenames_ul = config[uc_ul]

                designated_subs = None
                designated_streams = None
                designated_steps = None

                # The assumption here is that the peak == 0

                if is_scan:

                    avail_subs = sorted(auto_avg_subs, reverse=True)

                    for subs in avail_subs:

                        if (
                            len(testcases_dl) % subs == 0
                            and len(testcases_dl) // subs <= 32
                        ):
                            designated_subs = subs
                            designated_streams = len(testcases_dl) // subs
                            designated_steps = 1
                            break

                else:

                    if len(testcases_dl) % args.subs == 0:
                        designated_subs = args.subs
                        designated_streams = len(testcases_dl) // args.subs
                        designated_steps = 1

                if designated_subs is None:
                    k += 1
                    continue
                else:
                    label = int(subcase.replace("Average: ", ""))

                    if not args.is_no_mps and data_targets is not None:

                        key = "+".join([str(k).zfill(2), str(label).zfill(2)])

                        target = data_targets.get(key, None)

                        if target is None:
                            k += 1
                            continue
                        else:
                            target = int(np.round(100 * float(target) / sms))

                    message = "Number of active cells: " + str(k) + "+" + str(label)

                    if target is not None:
                        message += (
                            "("
                            + str(2 * int(np.round(target * sms / 200)))
                            + ","
                            + str(designated_subs)
                            + ")"
                        )

                    print(message)

                    traffic_avg(
                        args,
                        vectors,
                        (testcases_dl, testcases_ul),
                        (filenames_dl, filenames_ul),
                    )

                    designated = (designated_subs, designated_streams, designated_steps)

                    if args.is_power:
                        powers["+".join([str(k).zfill(2), str(label).zfill(2)])] = run(
                            args,
                            designated,
                            mig,
                            mig_gpu,
                            command,
                            vectors,
                            mode,
                            target,
                            label,
                        )
                    else:
                        sweeps["+".join([str(k).zfill(2), str(label).zfill(2)])] = run(
                            args,
                            designated,
                            mig,
                            mig_gpu,
                            command,
                            vectors,
                            mode,
                            target,
                            label,
                        )
        else:
            raise NotImplementedError

        k += 1

    if args.is_power:
        if len(list(powers.keys())) > 0:
            ofile = open(output.replace("sweep", "power") + ".json", "w")
            json.dump(powers, ofile, indent=2)
            ofile.close()
    else:
        if not args.is_debug:
            if args.is_check_traffic:
                output_file = output.replace("sweep", "error") + ".json"
            else:
                output_file = output + ".json"

            ofile = open(output_file, "w")
            json.dump(sweeps, ofile, indent=2)
            ofile.close()
