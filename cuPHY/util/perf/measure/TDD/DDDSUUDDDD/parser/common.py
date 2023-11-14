# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np


def check(
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
):

    pdsch_slots = 8

    if len(pdsch) < pdsch_slots:
        return False

    offset = 0

    if args.is_rec_bf:
        offset = 1

    for idx, itm in enumerate(pdsch):

        if np.abs(itm[0] - (idx + offset) * 500) > 50:
            return False

    return True


def unpack(
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
):

    new_pdsch = []
    new_pdcch = []
    new_csirs = []
    new_bwc = []
    new_ssb = []

    if args.is_pusch_cascaded:

        if args.is_rec_bf:

            for idx, itm in enumerate(pdsch):

                new_pdsch.append(itm[1] - itm[0])

            for idx, itm in enumerate(pdcch):

                new_pdcch.append(itm - pdsch[idx][0])

            for idx, itm in enumerate(csirs):

                new_csirs.append(itm - pdcch[idx])

            for idx, itm in enumerate(bwc):

                new_bwc.append(itm[1] - itm[0]) # use BWC standalone time measurement

            for idx, itm in enumerate(ssb):
                new_ssb.append(itm[1] - itm[0]) # pdsch[idx + 4][0])

            if len(rach) > 0:
                rach = rach[1] - rach[0]
            else:
                rach = 0

        else:

            for idx, itm in enumerate(pdsch):

                new_pdsch.append(itm[1] - itm[0])

            for idx, itm in enumerate(pdcch):

                new_pdcch.append(itm - pdsch[idx][0])

            for idx, itm in enumerate(csirs):

                new_csirs.append(itm - pdcch[idx])

            if len(rach) > 0:
                rach = rach[1] - rach[0]
            else:
                rach = 0

            for idx, itm in enumerate(ssb):
                new_ssb.append(itm[1] - itm[0]) # pdsch[idx + 4][0])

    else:

        if args.is_rec_bf:

            for idx, itm in enumerate(pdsch):

                new_pdsch.append(itm[1] - itm[0])

            for idx, itm in enumerate(pdcch):

                new_pdcch.append(itm - pdsch[idx][0])

            for idx, itm in enumerate(csirs):

                new_csirs.append(itm - pdcch[idx])

            for idx, itm in enumerate(bwc):

                new_bwc.append(itm[1] - itm[0]) # use BWC standalone time measurement

            for idx, itm in enumerate(ssb):
                new_ssb.append(itm[1] - itm[0]) # pdsch[idx + 4][0])

            if pusch2 > 0:
                pusch2 -= 1000

            if pucch2 > 0:
                pucch2 -= 1000

            if pusch1 > 0:
                pusch1 -= 500

            if pucch1 > 0:
                pucch1 -= 500

            if len(rach) > 0:
                rach = rach[1] - rach[0]
            else:
                rach = 0

        else:

            for idx, itm in enumerate(pdsch):

                new_pdsch.append(itm[1] - itm[0])

            for idx, itm in enumerate(pdcch):

                new_pdcch.append(itm - pdsch[idx][0])

            for idx, itm in enumerate(csirs):

                new_csirs.append(itm - pdcch[idx])

            if pusch2 > 0:
                pusch2 -= 500

            if pucch2 > 0:
                pucch2 -= 500

            if len(rach) > 0:
                rach = rach[1] - rach[0]
            else:
                rach = 0

            for idx, itm in enumerate(ssb):
                new_ssb.append(itm[1] - itm[0]) # pdsch[idx + 4][0])

    return (
        pusch1,
        pusch2,
        new_pdsch,
        new_bwc,
        srs1,
        srs2,
        rach,
        new_pdcch,
        pucch1,
        pucch2,
        new_ssb,
        new_csirs,
    )
