# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import numpy as np
import itertools
import json
import sys

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
    type=int,
    dest="avg",
    help="Specifies the number of avg. cells",
    required=True,
)
base.add_argument(
    "--subs",
    type=int,
    dest="subs",
    default=4,
    help="Specifies the number of sub-CTX to use",
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
    choices=["F01"],
    default="F01",
    help="Specifies the use case",
)

args = base.parse_args()

if args.is_exact:
    if args.avg is None:
        base.error("when --exact is used, --avg needs to also be provided")

peak = args.peak

base = [4, 5, 6, 7, 8]
average = []
subs = args.subs

is_valid = False

if args.is_exact:
    if args.peak[0] > 0 and args.avg == 0:
        average.append([args.avg])
        is_valid = True
    else:
        for k in range(1, 5):
            buffer = subs * np.array(base) * k

            if args.avg in buffer:
                average.append([args.avg])
                is_valid = True
            else:
                average.append([])
else:
    is_valid = True
    for k in range(1, 5):
        check = itertools.chain.from_iterable(average)
        buffer = subs * np.array(base) * k
        buffer = list(set(buffer) - set(check))

        if args.avg is not None:
            buffer = [x for x in buffer if x <= args.avg]

        average.append(sorted(buffer))

if not is_valid:

    average = []

    for k in range(1, 5):
        check = itertools.chain.from_iterable(average)
        buffer = subs * np.array(base) * k
        buffer = list(set(buffer) - set(check))
        average.append(sorted(buffer))

    average = list(itertools.chain(*average))
    sys.exit(
        f"error: only {average} avg. cells are supported; for 0 avg. cells, the peak cells must be >0"
    )

buffer = average
average = []
for k in range(len(args.peak)):
    average.append(buffer)

data = {}

for idx_peak, k_peak in enumerate(peak):

    buffer = {}

    for k_mul in range(len(average[idx_peak])):

        for k_average in average[idx_peak][k_mul]:

            buffer_average = {}
            buffer_average[f"{args.case} - PDSCH"] = []
            buffer_average[f"{args.case} - PUSCH"] = []

            for k_rep_p in range(k_peak):
                buffer_average[f"{args.case} - PDSCH"].append(f"{args.case}-PP-00")
                buffer_average[f"{args.case} - PUSCH"].append(f"{args.case}-PP-00")

            if args.is_fdm:

                for k_direct in range(k_average):
                    buffer_average[f"{args.case} - PDSCH"].append(f"{args.case}-AX-01")
                    buffer_average[f"{args.case} - PUSCH"].append(f"{args.case}-AX-01")

            else:

                discriminant = k_average // subs // (k_mul + 1)

                if discriminant > 0:
                    for k_rep in range(subs * (k_mul + 1)):
                        buffer_average[f"{args.case} - PDSCH"].append(
                            f"{args.case}-AC-0" + str(discriminant)
                        )
                        buffer_average[f"{args.case} - PUSCH"].append(
                            f"{args.case}-AC-0" + str(discriminant)
                        )
                        buffer_average[f"{args.case} - PDSCH"].append(
                            f"{args.case}-AM-0" + str(discriminant)
                        )
                        buffer_average[f"{args.case} - PUSCH"].append(
                            f"{args.case}-AM-0" + str(discriminant)
                        )
                        buffer_average[f"{args.case} - PDSCH"].append(
                            f"{args.case}-AE-0" + str(discriminant)
                        )
                        buffer_average[f"{args.case} - PUSCH"].append(
                            f"{args.case}-AE-0" + str(discriminant)
                        )

            check = buffer.get("Average: " + str(k_average), {})

            if len(check.keys()) == 0:
                buffer["Average: " + str(k_average)] = buffer_average

    data["Peak: " + str(k_peak)] = buffer

ofile = open(f"uc_avg_{args.case}_FDD.json", "w")
json.dump(data, ofile, indent=4)
ofile.close()
