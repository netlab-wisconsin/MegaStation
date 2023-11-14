# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from .common import check, unpack


def run(args, lines):

    latencies = {}
    clat_ul = []
    clat_dl = []
    clat_bf = []
    clat_sr1 = []
    clat_sr2 = []
    clat_ra = []
    clat_cdl = []
    clat_cul = []
    clat_ssb = []
    clat_cr = []

    k = 0

    is_correct = False

    while k < len(lines):
        lst = lines[k].split()

        if len(lst) == 4 and lst[0] == "Slot" and lst[1] == "pattern" and lst[2] == "#":

            k += 1

            pusch = 0
            pdsch = []
            bwc = []
            srs1 = 0
            srs2 = 0
            rach = []
            pdcch = []
            pucch = 0
            ssb = 0
            csirs = []

            while k < len(lines) and "----" not in lines[k]:
                lst = lines[k].split()

                if len(lst) > 3:

                    if lst[0] == "Average" and lst[1] == "PUSCH" and lst[2] == "run":
                        pusch = float(lst[4])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUCCH" and lst[2] == "run":
                        pucch = float(lst[4])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "SRS1":
                        srs1 = float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "SRS2":
                        srs2 = float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "PRACH":
                        rach = [float(lst[10]), float(lst[7])]
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "PDSCH":
                        pdsch.append([float(lst[10]), float(lst[7])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "PDCCH":
                        pdcch.append(float(lst[7]))
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "CSIRS":
                        csirs.append(float(lst[7]))
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "SSB":
                        ssb = float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "BWC":
                        bwc.append([float(lst[10]), float(lst[7])])
                        k += 1
                        continue

                k += 1

            is_correct = is_correct or check(
                args, pusch, pdsch, bwc, srs1, srs2, rach, pdcch, pucch, ssb, csirs
            )
            pusch, pdsch, bwc, srs1, srs2, rach, pdcch, pucch, ssb, csirs = unpack(
                args, pusch, pdsch, bwc, srs1, srs2, rach, pdcch, pucch, ssb, csirs
            )

            if pusch > 0:
                clat_ul.append(pusch)
            if pucch > 0:
                clat_cul.append(pucch)

            if ssb > 0:
                clat_ssb.append(ssb)

            clat_dl.extend(pdsch)
            clat_cdl.extend(pdcch)
            clat_cr.extend(csirs)
            if args.is_rec_bf:
                clat_bf.extend(bwc)
                clat_sr1.append(srs1)
                if srs2 > 0:
                    clat_sr2.append(srs2)

            if args.is_prach:
                if rach > 0:
                    clat_ra.append(rach)

        k += 1

    latencies["Structure"] = is_correct
    latencies["PDSCH"] = clat_dl
    latencies["PUSCH"] = clat_ul
    if args.is_pdcch:
        latencies["PDCCH"] = clat_cdl

    if args.is_pucch:
        latencies["PUCCH"] = clat_cul

    if args.is_rec_bf:
        latencies["BWC"] = clat_bf
        latencies["SRS1"] = clat_sr1
        latencies["SRS2"] = clat_sr2

    if args.is_prach:
        latencies["PRACH"] = clat_ra

    if args.is_ssb:
        latencies["SSB"] = clat_ssb

    if args.is_csirs:
        latencies["CSI-RS"] = clat_cr

    return latencies
