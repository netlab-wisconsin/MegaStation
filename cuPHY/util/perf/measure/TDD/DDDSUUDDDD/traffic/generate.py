# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import yaml
import json

from ....analyze import extract


def run(args, vectors, testcases, filenames):

    (
        testcases_dl,
        testcases_ul,
        testcases_bf,
        testcases_sr,
        testcases_ra,
        testcases_cdl,
        testcases_cul,
        testcases_ssb,
        testcases_cr,
    ) = testcases
    (
        filenames_dl,
        filenames_ul,
        filenames_bf,
        filenames_sr,
        filenames_ra,
        filenames_cdl,
        filenames_cul,
        filenames_ssb,
        filenames_cr,
    ) = filenames

    ofile = open(vectors, "w")

    payload = {}
    payload["cells"] = len(testcases_dl)

    ifile = open("measure/TDD/DDDSUUDDDD/traffic/priorities.json", "r")
    priorities = json.load(ifile)
    ifile.close()

    buffer = {}

    for key in priorities.keys():
        label = "_".join([key, "PRIO"])
        buffer[label] = priorities[key]

    payload.update(buffer)

    channels = []

    for sweep_idx in range(args.sweeps):

        channel = {}

        if not args.is_no_pdsch:
            channel["PDSCH"] = [
                os.path.join(args.vfld, filenames_dl[testcase])
                for testcase in testcases_dl
            ]

        if testcases_cdl is not None:
            channel["PDCCH"] = [
                os.path.join(args.vfld, filenames_cdl[testcase])
                for testcase in testcases_cdl
            ]

        if testcases_cr is not None:
            channel["CSIRS"] = [
                os.path.join(args.vfld, filenames_cr[testcase])
                for testcase in testcases_cr
            ]

        if testcases_bf is not None:
            channel["BWC"] = [
                os.path.join(args.vfld, filenames_bf[testcase])
                for testcase in testcases_bf
            ]

        if sweep_idx % 8 == 0:

            if not args.is_no_pusch:

                channel["PUSCH"] = [
                    os.path.join(args.vfld, filenames_ul[testcase])
                    for testcase in testcases_ul
                ]

            if testcases_cul is not None:
                channel["PUCCH"] = [
                    os.path.join(args.vfld, filenames_cul[testcase])
                    for testcase in testcases_cul
                ]

            if testcases_ra is not None:
                channel["PRACH"] = [
                    os.path.join(args.vfld, filenames_ra[testcase])
                    for testcase in testcases_ra
                ]

            if testcases_ssb is not None:
                channel["SSB"] = [
                    os.path.join(args.vfld, filenames_ssb[testcase])
                    for testcase in testcases_ssb
                ]

            if testcases_sr is not None:
                channel["SRS"] = [
                    os.path.join(args.vfld, filenames_sr[testcase])
                    for testcase in testcases_sr
                ]

        if sweep_idx % 8 == 1:

            if not args.is_no_pusch:

                channel["PUSCH"] = [
                    os.path.join(args.vfld, filenames_ul[testcase])
                    for testcase in testcases_ul
                ]

            if testcases_cul is not None:
                channel["PUCCH"] = [
                    os.path.join(args.vfld, filenames_cul[testcase])
                    for testcase in testcases_cul
                ]

        channels.append(channel)

    payload["slots"] = channels
    payload["parameters"] = extract(args, channels)

    ofile = open(vectors, "w")
    yaml.dump(payload, ofile, sort_keys=False)
    ofile.close()
