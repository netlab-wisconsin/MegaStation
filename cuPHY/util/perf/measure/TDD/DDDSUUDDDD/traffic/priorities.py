# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json

base = argparse.ArgumentParser()
base.add_argument(
    "--first",
    type=str,
    dest="priority",
    choices=["pdsch", "prach"],
    help="Specifies which channel has the highest priority",
    required=True,
)
args = base.parse_args()

standard = [
    ["pdsch", "pdcch", "csirs"],
    ["pucch", "pucch2"],
    ["pusch", "pusch2"],
    ["srs", "ssb"],
    ["prach"],
]
alternative = [
    ["prach"],
    ["pdsch", "pdcch"],
    ["pucch", "pucch2"],
    ["pusch", "pusch2"],
    ["srs", "ssb"],
]

if args.priority == "pdsch":
    choice = standard
elif args.priority == "prach":
    choice = alternative
else:
    raise NotImplementedError

buffer = {}

for idx, itm in enumerate(choice):

    for sub_itm in itm:

        buffer[sub_itm.upper()] = idx

ofile = open("measure/TDD/DDDSUUDDDD/traffic/priorities.json", "w")
json.dump(buffer, ofile, indent=2)
ofile.close()
