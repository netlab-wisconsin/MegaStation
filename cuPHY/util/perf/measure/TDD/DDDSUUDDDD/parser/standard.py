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
    clat_ul1 = []
    clat_ul2 = []
    clat_dl = []
    clat_bf = []
    clat_sr1 = []
    clat_sr2 = []
    clat_ra = []
    clat_cdl = []
    clat_cul1 = []
    clat_cul2 = []
    clat_ssb = []
    clat_cr = []

    if args.is_pusch_cascaded:
        mode = "Sequential"
    else:
        mode = "Parallel"

    k = 0

    is_correct = False

    # -----------------------------------------------------------
    # Slot pattern # 0
    # average slot pattern run time: 4765.95 us (averaged over 1 iterations)
    # Average PUSCH run time: 1977.38 us (averaged over 1 iterations)
    # Average PUSCH2 run time: 3363.26 us (averaged over 1 iterations)
    # Slot # 0: average SRS1 run time: 279.17 us (averaged over 1 iterations)
    # Slot # 4: average SRS2 run time: 3397.86 us (averaged over 1 iterations)
    # Slot # 0: average PRACH run time: 4430.30 us (averaged over 1 iterations)
    # Slot # 0: average PDSCH run time: 806.85 us (averaged over 1 iterations)
    # Slot # 0: average BWC run time: 469.50 us (averaged over 1 iterations)
    # Slot # 1: average PDSCH run time: 1979.78 us (averaged over 1 iterations)
    # Slot # 1: average BWC run time: 1039.97 us (averaged over 1 iterations)
    # Slot # 2: average PDSCH run time: 2942.75 us (averaged over 1 iterations)
    # Slot # 2: average BWC run time: 2100.22 us (averaged over 1 iterations)
    # Slot # 3: average PDSCH run time: 3587.62 us (averaged over 1 iterations)
    # Slot # 3: average BWC run time: 3386.59 us (averaged over 1 iterations)
    # Slot # 4: average PDSCH run time: 3874.72 us (averaged over 1 iterations)
    # Slot # 4: average BWC run time: 3671.39 us (averaged over 1 iterations)
    # Slot # 5: average PDSCH run time: 4150.62 us (averaged over 1 iterations)
    # Slot # 5: average BWC run time: 3953.95 us (averaged over 1 iterations)
    # Slot # 6: average PDSCH run time: 4476.99 us (averaged over 1 iterations)
    # Slot # 6: average BWC run time: 4233.18 us (averaged over 1 iterations)
    # Slot # 7: average PDSCH run time: 4755.36 us (averaged over 1 iterations)

    # -----------------------------------------------------------

    while k < len(lines):
        lst = lines[k].split()

        if len(lst) == 4 and lst[0] == "Slot" and lst[1] == "pattern" and lst[2] == "#":

            k += 1

            pusch1 = 0
            pusch2 = 0
            pdsch = []
            bwc = []
            srs1 = 0
            srs2 = 0
            rach = []
            pdcch = []
            pucch1 = 0
            pucch2 = 0
            ssb = []
            csirs = []

            while k < len(lines) and "----" not in lines[k]:
                lst = lines[k].split()

                if len(lst) > 3:

                    if lst[0] == "Average" and lst[1] == "PUSCH" and lst[2] == "run":
                        pusch1 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUSCH2" and lst[2] == "run":
                        pusch2 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUCCH" and lst[2] == "run":
                        pucch1 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUCCH2" and lst[2] == "run":
                        pucch2 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "SRS1":
                        srs1 = float(lst[7]) - float(lst[10])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "SRS2":
                        srs2 = float(lst[7]) - float(lst[10])
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
                        ssb.append([float(lst[10]), float(lst[7])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "BWC":
                        bwc.append([float(lst[10]), float(lst[7])])
                        k += 1
                        continue

                k += 1

            is_correct = check(
                args,
                pusch1,
                pusch2,
                pdsch,
                bwc,
                srs1,
                srs2,
                rach,
                pdcch,
                pucch1,
                pucch2,
                ssb,
                csirs,
            )

            (
                pusch1,
                pusch2,
                pdsch,
                bwc,
                srs1,
                srs2,
                rach,
                pdcch,
                pucch1,
                pucch2,
                ssb,
                csirs,
            ) = unpack(
                args,
                pusch1,
                pusch2,
                pdsch,
                bwc,
                srs1,
                srs2,
                rach,
                pdcch,
                pucch1,
                pucch2,
                ssb,
                csirs,
            )

            if pusch1 > 0:
                clat_ul1.append(pusch1)
            if pusch2 > 0:
                clat_ul2.append(pusch2)

            if pucch1 > 0:
                clat_cul1.append(pucch1)
            if pucch2 > 0:
                clat_cul2.append(pucch2)

            clat_ssb.extend(ssb)

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
    latencies["Mode"] = mode
    latencies["PDSCH"] = clat_dl
    latencies["PUSCH1"] = clat_ul1
    latencies["PUSCH2"] = clat_ul2

    if args.is_pdcch:
        latencies["PDCCH"] = clat_cdl

    if args.is_pucch:
        latencies["PUCCH1"] = clat_cul1
        latencies["PUCCH2"] = clat_cul2

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
