# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import h5py
import numpy as np


def extract(args, channels):

    pdsch = []
    pusch = []

    for slot in channels:

        buffer = slot.get("PDSCH", None)

        if buffer is not None:
            pdsch.append(buffer)

        buffer = slot.get("PUSCH", None)

        if buffer is not None:
            pusch.append(buffer)

    pdsch_flattened = []

    for slot in pdsch:
        pdsch_flattened.extend(slot)

    pdsch_flattened = list(set(pdsch_flattened))

    pusch_flattened = []

    for slot in pusch:
        pusch_flattened.extend(slot)

    pusch_flattened = list(set(pusch_flattened))

    pdsch_cell_params = {}

    pdsch_max_ntb_per_cell = 0
    pdsch_max_ncb_per_cell = 0
    pdsch_max_ncb_per_tb = 0
    pdsch_max_prb_per_cell = 0
    pdsch_max_ntx_per_cell = 0

    for vector in pdsch_flattened:

        ifile = h5py.File(vector)

        num_tb = int(ifile["cw_pars"].shape[0])

        num_cb = 0
        max_cb = np.zeros(num_tb, dtype=int)

        for k in range(num_tb):
            max_cb[k] = int(ifile[f"tb{k}_cbs"].shape[0])
            num_cb += int(ifile[f"tb{k}_cbs"].shape[0])

        max_cb_per_tb = int(np.max(max_cb))

        num_prb = int(ifile["cellStat_pars"]["nPrbDlBwp"][0])
        num_ntx = int(ifile["cellStat_pars"]["nTxAnt"][0])

        pdsch_cell_params[vector] = {}
        pdsch_cell_params[vector] = (num_tb, num_cb, max_cb_per_tb, num_ntx, num_prb)

        if pdsch_max_ntb_per_cell < num_tb:
            pdsch_max_ntb_per_cell = num_tb

        if pdsch_max_ncb_per_cell < num_cb:
            pdsch_max_ncb_per_cell = num_cb

        if pdsch_max_ncb_per_tb < max_cb_per_tb:
            pdsch_max_ncb_per_tb = max_cb_per_tb

        if pdsch_max_prb_per_cell < num_prb:
            pdsch_max_prb_per_cell = num_prb

        if pdsch_max_ntx_per_cell < num_ntx:
            pdsch_max_ntx_per_cell = num_ntx

    pusch_cell_params = {}

    pusch_max_ntb_per_cell = 0
    pusch_max_ncb_per_cell = 0
    pusch_max_ncb_per_tb = 0
    pusch_max_prb_per_cell = 0
    pusch_max_ntx_per_cell = 0

    for vector in pusch_flattened:

        ifile = h5py.File(vector)

        num_tb = int(ifile["tb_pars"].shape[0])

        num_cb = 0
        max_cb = np.zeros(num_tb, dtype=int)

        for k in range(num_tb):
            max_cb[k] = int(ifile["tb_pars"][k]["nCb"])
            num_cb += int(ifile["tb_pars"][k]["nCb"])

        max_cb_per_tb = int(np.max(max_cb))

        num_prb = int(ifile["gnb_pars"]["nPrb"][0])
        num_ntx = int(ifile["gnb_pars"]["nRx"][0])

        pusch_cell_params[vector] = {}
        pusch_cell_params[vector] = (num_tb, num_cb, max_cb_per_tb, num_ntx, num_prb)

        if pusch_max_ntb_per_cell < num_tb:
            pusch_max_ntb_per_cell = num_tb

        if pusch_max_ncb_per_cell < num_cb:
            pusch_max_ncb_per_cell = num_cb

        if pusch_max_ncb_per_tb < max_cb_per_tb:
            pusch_max_ncb_per_tb = max_cb_per_tb

        if pusch_max_prb_per_cell < num_prb:
            pusch_max_prb_per_cell = num_prb

        if pusch_max_ntx_per_cell < num_ntx:
            pusch_max_ntx_per_cell = num_ntx

    pdsch_max_ncb_per_slot = 0
    pdsch_max_ntb_per_slot = 0

    for slot in pdsch:

        cell_num_tb = 0
        cell_num_cb = 0

        for cell in slot:

            (num_tb, num_cb, max_cb_per_tb, num_ntx, num_prb) = pdsch_cell_params[cell]

            cell_num_tb += num_tb
            cell_num_cb += num_cb

        if pdsch_max_ncb_per_slot < cell_num_cb:
            pdsch_max_ncb_per_slot = cell_num_cb

        if pdsch_max_ntb_per_slot < cell_num_tb:
            pdsch_max_ntb_per_slot = cell_num_tb

    pusch_max_ncb_per_slot = 0
    pusch_max_ntb_per_slot = 0

    for slot in pusch:

        cell_num_tb = 0
        cell_num_cb = 0

        for cell in slot:

            (num_tb, num_cb, max_cb_per_tb, num_ntx, num_prb) = pusch_cell_params[cell]

            cell_num_tb += num_tb
            cell_num_cb += num_cb

        if pusch_max_ncb_per_slot < cell_num_cb:
            pusch_max_ncb_per_slot = cell_num_cb

        if pusch_max_ntb_per_slot < cell_num_tb:
            pusch_max_ntb_per_slot = cell_num_tb

    results = {}

    results["PDSCH"] = {}
    results["PDSCH"]["Max #TB per slot"] = pdsch_max_ntb_per_slot
    results["PDSCH"]["Max #CB per slot"] = pdsch_max_ncb_per_slot
    results["PDSCH"]["Max #TB per slot per cell"] = pdsch_max_ntb_per_cell
    results["PDSCH"]["Max #CB per slot per cell"] = pdsch_max_ncb_per_cell
    results["PDSCH"]["Max #CB per slot per cell per TB"] = pdsch_max_ncb_per_tb
    results["PDSCH"]["Max #TX per cell"] = pdsch_max_ntx_per_cell
    results["PDSCH"]["Max #PRB per cell"] = pdsch_max_prb_per_cell

    results["PUSCH"] = {}
    results["PUSCH"]["Max #TB per slot"] = pusch_max_ntb_per_slot
    results["PUSCH"]["Max #CB per slot"] = pusch_max_ncb_per_slot
    results["PUSCH"]["Max #TB per slot per cell"] = pusch_max_ntb_per_cell
    results["PUSCH"]["Max #CB per slot per cell"] = pusch_max_ncb_per_cell
    results["PUSCH"]["Max #CB per slot per cell per TB"] = pusch_max_ncb_per_tb
    results["PUSCH"]["Max #RX per cell"] = pusch_max_ntx_per_cell
    results["PUSCH"]["Max #PRB per cell"] = pusch_max_prb_per_cell

    return results
