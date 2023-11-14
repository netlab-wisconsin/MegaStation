# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import argparse
import numpy as np

base = argparse.ArgumentParser()
base.add_argument(
    "--peak",
    nargs="+",
    type=int,
    dest="peak",
    help="Specifies the list of peak cells",
    required=True,
)
base.add_argument(
    "--avg",
    nargs="+",
    type=int,
    dest="avg",
    help="Specifies the number of avg. cells",
    required=True,
)
base.add_argument(
    "--exact",
    action="store_true",
    dest="is_exact",
    help="Specified whether the avg. list is the exact target for a given peak cell, or the maximum cell count for it",
)
base.add_argument(
    "--fdm",
    action="store_true",
    dest="is_fdm",
    help="Specified whether to use the FDM description for the avg. cells",
)
base.add_argument(
    "--case",
    type=str,
    dest="case",
    choices=["F08", "F09", "F14"],
    default="F14",
    help="Specifies the use case",
)

args = base.parse_args()

peak = args.peak
average = []

if args.is_exact:

    for peak_item in peak:
        buffer = []

        for avg_item in args.avg:
            buffer.append(avg_item)

        average.append(buffer)
else:

    for peak_item in peak:

        for item in args.avg:

            average.append(np.arange(0, item + 1))

data = {}

for idx_peak, k_peak in enumerate(peak):

    buffer = {}

    for k_average in average[idx_peak]:

        repetitions = k_average // 7
        remainder = k_average % 7

        buffer_average = {}
        buffer_average[f"{args.case} - PDSCH"] = []
        buffer_average[f"{args.case} - PUSCH"] = []
        if args.case == "F14" or args.case == "F09":
            buffer_average[f"{args.case} - BWC"] = []
            buffer_average[f"{args.case} - SRS"] = []
        buffer_average[f"{args.case} - SSB"] = []
        buffer_average[f"{args.case} - PRACH"] = []
        buffer_average[f"{args.case} - PDCCH"] = []
        buffer_average[f"{args.case} - CSIRS"] = []
        buffer_average[f"{args.case} - PUCCH"] = []

        for k_rep_p in range(k_peak):
            buffer_average[f"{args.case} - PDSCH"].append(f"{args.case}-PP-00")
            buffer_average[f"{args.case} - PUSCH"].append(f"{args.case}-PP-00")
            if args.case == "F14" or args.case == "F09":
                buffer_average[f"{args.case} - BWC"].append(f"{args.case}-PP-00")
            buffer_average[f"{args.case} - PDCCH"].append(f"{args.case}-PP-00")
            buffer_average[f"{args.case} - CSIRS"].append(f"{args.case}-PP-00")
            buffer_average[f"{args.case} - PUCCH"].append(f"{args.case}-PP-00")

        if args.is_fdm:

            for k_direct in range(k_average):
                buffer_average[f"{args.case} - PDSCH"].append(f"{args.case}-AX-01")
                buffer_average[f"{args.case} - PUSCH"].append(f"{args.case}-AX-01")

            for k_rep in range(repetitions):
                if args.case == "F14":
                    buffer_average[f"{args.case} - BWC"].append(f"{args.case}-AC-07")
                    buffer_average[f"{args.case} - BWC"].append(f"{args.case}-AM-07")
                    buffer_average[f"{args.case} - BWC"].append(f"{args.case}-AE-07")

            if remainder > 0:
                if args.case == "F14" or args.case == "F09":
                    buffer_average[f"{args.case} - BWC"].append(
                        f"{args.case}-AC-0" + str(remainder)
                    )
                    buffer_average[f"{args.case} - BWC"].append(
                        f"{args.case}-AM-0" + str(remainder)
                    )
                    buffer_average[f"{args.case} - BWC"].append(
                        f"{args.case}-AE-0" + str(remainder)
                    )

        else:

            for k_rep in range(repetitions):
                buffer_average[f"{args.case} - PDSCH"].append(f"{args.case}-AC-07")
                buffer_average[f"{args.case} - PUSCH"].append(f"{args.case}-AC-07")
                if args.case == "F14":
                    buffer_average[f"{args.case} - BWC"].append(f"{args.case}-AC-07")
                buffer_average[f"{args.case} - PDSCH"].append(f"{args.case}-AM-07")
                buffer_average[f"{args.case} - PUSCH"].append(f"{args.case}-AM-07")
                if args.case == "F14":
                    buffer_average[f"{args.case} - BWC"].append(f"{args.case}-AM-07")
                buffer_average[f"{args.case} - PDSCH"].append(f"{args.case}-AE-07")
                buffer_average[f"{args.case} - PUSCH"].append(f"{args.case}-AE-07")
                if args.case == "F14":
                    buffer_average[f"{args.case} - BWC"].append(f"{args.case}-AE-07")

            if remainder > 0:
                buffer_average[f"{args.case} - PDSCH"].append(
                    f"{args.case}-AC-0" + str(remainder)
                )
                buffer_average[f"{args.case} - PUSCH"].append(
                    f"{args.case}-AC-0" + str(remainder)
                )
                if args.case == "F14" or args.case == "F09":
                    buffer_average[f"{args.case} - BWC"].append(
                        f"{args.case}-AC-0" + str(remainder)
                    )
                buffer_average[f"{args.case} - PDSCH"].append(
                    f"{args.case}-AM-0" + str(remainder)
                )
                buffer_average[f"{args.case} - PUSCH"].append(
                    f"{args.case}-AM-0" + str(remainder)
                )
                if args.case == "F14":
                    buffer_average[f"{args.case} - BWC"].append(
                        f"{args.case}-AM-0" + str(remainder)
                    )
                buffer_average[f"{args.case} - PDSCH"].append(
                    f"{args.case}-AE-0" + str(remainder)
                )
                buffer_average[f"{args.case} - PUSCH"].append(
                    f"{args.case}-AE-0" + str(remainder)
                )
                if args.case == "F14":
                    buffer_average[f"{args.case} - BWC"].append(
                        f"{args.case}-AE-0" + str(remainder)
                    )

        if args.case == "F14" or args.case == "F09":
            for k_srs in range(k_peak + k_average):
                buffer_average[f"{args.case} - SRS"].append(f"{args.case}-PP-00")

        for k_rach in range(int(np.ceil((k_peak + k_average)))):
            buffer_average[f"{args.case} - PRACH"].append(f"{args.case}-PP-00")

        for k_pdcch in range(k_average):
            buffer_average[f"{args.case} - PDCCH"].append(f"{args.case}-AX-01")

        for k_pucch in range(k_average):
            buffer_average[f"{args.case} - PUCCH"].append(f"{args.case}-AX-01")

        for k_ssb in range(k_peak + k_average):
            buffer_average[f"{args.case} - SSB"].append(f"{args.case}-PP-00")

        for k_ssb in range(k_average):
            buffer_average[f"{args.case} - CSIRS"].append(f"{args.case}-AX-01")

        buffer["Average: " + str(k_average)] = buffer_average

    data["Peak: " + str(k_peak)] = buffer

ofile = open(f"uc_avg_{args.case}_TDD.json", "w")
json.dump(data, ofile, indent=4)
ofile.close()
