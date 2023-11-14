# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np


def check(args, pusch, pdsch, bwc, srs1, srs2, rach, pdcch, pucch, ssb, csirs):

    pdsch_slots = 4

    if len(pdsch) < pdsch_slots:
        return False

    for idx, itm in enumerate(pdsch):

        if np.abs(itm[0] - idx * 500) > 50:
            return False

    return True


def unpack(args, pusch, pdsch, bwc, srs1, srs2, rach, pdcch, pucch, ssb, csirs):

    new_pdsch = []
    new_pdcch = []
    new_csirs = []
    new_bwc = []

    for idx, itm in enumerate(pdsch):
        new_pdsch.append(itm[1] - itm[0])

    for idx, itm in enumerate(pdcch):
        new_pdcch.append(itm - pdsch[idx][0])

    for idx, itm in enumerate(csirs):
        new_csirs.append(itm - pdcch[idx])

    if args.is_rec_bf:
        for idx, itm in enumerate(bwc):

            if bwc[idx][0] == pdsch[idx][0]:
                new_bwc.append(bwc[idx][1] - pdsch[idx][1])
            else:
                new_bwc.append(bwc[idx][1] - bwc[idx][0] - 375)

    if pusch > 0:
        pusch -= 500

    if pucch > 0:
        pucch -= 500

    if len(rach) > 0:
        rach = rach[1] - rach[0]
    else:
        rach = 0

    return (
        pusch,
        new_pdsch,
        new_bwc,
        srs1,
        srs2,
        rach,
        new_pdcch,
        pucch,
        ssb,
        new_csirs,
    )
