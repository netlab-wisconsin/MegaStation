# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


def parse(args, lines):

    clat_tot = []
    clat_ul = []
    clat_dl = []

    k = 0

    while k < len(lines):
        lst = lines[k].split()

        if len(lst) == 3 and lst[0] == "Slot" and lst[1] == "#":

            k += 1

            lst = lines[k].split()

            if len(lst) > 0 and lst[0] == "average" and lst[1] == "slot":
                clat_tot.append(float(lst[4]))

                k += 1

                while k < len(lines) and "----" not in lines[k]:
                    lst = lines[k].split()

                    if len(lst) > 3:

                        if lst[0] == "Ctx" and lst[4] == "PDSCH":
                            clat_dl.append(float(lst[7]))
                            k += 1
                            continue

                        if lst[0] == "Ctx" and lst[4] == "PUSCH":
                            clat_ul.append(float(lst[7]))
                            k += 1
                            continue

                    k += 1

        k += 1

    latencies = {}
    latencies["Total"] = clat_tot
    latencies["PDSCH"] = clat_dl
    latencies["PUSCH"] = clat_ul

    return latencies
