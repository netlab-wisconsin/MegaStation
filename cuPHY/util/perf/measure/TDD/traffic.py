# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import yaml


def traffic_het(args, vectors, k, testcases, filenames):

    testcases_dl, testcases_ul = testcases
    filenames_dl, filenames_ul = filenames

    ofile = open(vectors, "w")

    payload = {}
    payload["cells"] = k

    channels = []

    for sweep_idx in range(args.sweeps):

        channel = {}
        tidxs_dl = np.random.randint(0, len(testcases_dl), k)
        tidxs_ul = np.random.randint(0, len(testcases_ul), k)

        if not args.is_no_pdsch:
            channel["PDSCH"] = [
                os.path.join(args.vfld, filenames_dl[testcases_dl[tidx_dl]])
                for tidx_dl in tidxs_dl
            ]
        if not args.is_no_pusch:
            channel["PUSCH"] = [
                os.path.join(args.vfld, filenames_ul[testcases_ul[tidx_ul]])
                for tidx_ul in tidxs_ul
            ]

        channels.append(channel)

    payload["slots"] = channels

    ofile = open(vectors, "w")
    yaml.dump(payload, ofile)
    ofile.close()


def traffic_avg(args, vectors, testcases, filenames):

    if args.pattern == "dddsu":
        from .DDDSU.traffic.generate import run

        run(args, vectors, testcases, filenames)
    else:
        from .DDDSUUDDDD.traffic.generate import run

        run(args, vectors, testcases, filenames)
