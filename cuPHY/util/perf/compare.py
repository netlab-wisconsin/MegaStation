# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
import json
import os
import sys
import datetime


base = argparse.ArgumentParser()
base.add_argument(
    "--filenames",
    type=str,
    nargs="+",
    dest="filenames",
    help="Specifies the files containing the results",
)
base.add_argument(
    "--folder",
    type=str,
    dest="folder",
    help="Specifies the folder containing the results",
)
base.add_argument(
    "--cells",
    type=str,
    nargs="+",
    dest="cells",
    default=1,
    help="Specifies the number of cells to focus the comparison for",
)
base.add_argument(
    "--filter",
    action="store_true",
    dest="is_filter",
    help="Specifies whether to remove values > mean + 3 sigma",
)
base.add_argument(
    "--percentile",
    type=float,
    dest="threshold",
    default=99.5,
    help="Specifies whether to remove values > mean + 3 sigma",
)
base.add_argument(
    "--fdd_pusch",
    action="store_true",
    dest="is_fdd_pusch",
    help="Internal",
)
base.add_argument(
    "--disable_uc",
    action="store_true",
    dest="is_disable_uc",
    help="Internal",
)
args = base.parse_args()

if args.filenames is None and args.folder is None:
    base.error("please specify which files or folder to analyze")

if len(args.cells) > 1 and args.folder is not None:
    base.error("multiple cell counts can be only be specified when using --filenames")

data = {}
legend = []

offset = 0

usecase = None

filenames = []

if args.filenames is not None:

    for fn in args.filenames:
        if "F01" in fn:
            usecase = "F01"
            break
        elif "F14" in fn:
            usecase = "F14"
            break
        elif "F08" in fn:
            usecase = "F08"
            break
        elif "F09" in fn:
            usecase = "F09"
            break
        else:
            sys.exit(
                "error: currently only F01, F08, F09 and F14 use cases are supported"
            )

    filenames.extend(args.filenames)

    for idx, filename in enumerate(args.filenames):
        ifile = open(filename, "r")
        data[idx] = json.load(ifile)
        ifile.close()
        if len(filename.split("/")) > 1:
            legend.append(filename.split("/")[-2])
        else:
            if len(args.cells) > 1:
                legend.append(args.cells[idx])
            else:
                legend.append(f"Dataset #{idx+1}")
        offset += 1

if args.folder is not None:
    folder = args.folder

    folder_filenames = []

    raw = os.walk(folder)

    for item in raw:
        c_root, c_flds, c_files = item
        for c_file in c_files:
            if ".json" in c_file:
                filename = os.path.join(c_root, c_file)
                folder_filenames.append(filename)

    folder_filenames = sorted(folder_filenames)

    for ffn in folder_filenames:
        if "F01" in ffn:
            usecase = "F01"
            break
        elif "F14" in ffn:
            usecase = "F14"
            break
        elif "F08" in ffn:
            usecase = "F08"
            break
        elif "F09" in ffn:
            usecase = "F09"
            break
        else:
            sys.exit(
                "error: currently only F01, F08, F09 and F14 use cases are supported"
            )

    filenames.extend(folder_filenames)

    for idx, filename in enumerate(folder_filenames):
        ifile = open(filename, "r")
        data[idx + offset] = json.load(ifile)
        ifile.close()

        if os.path.split(filename)[0] == folder:
            legend.append(f"Dataset #{idx+1}")
        else:
            legend.append(os.path.split(os.path.split(filename)[0])[-1])

    offset += len(folder_filenames)

legend.append("Constraint")

if len(args.cells) > 1:

    key = []

    for cells in args.cells:
        if cells.isnumeric():
            key.append(cells.zfill(2))
        else:
            key.append("+".join([x.zfill(2) for x in cells.split("+")]))
else:
    if args.cells[0].isnumeric():
        key = [args.cells[0].zfill(2)] * len(data.keys())
    else:
        key = ["+".join([x.zfill(2) for x in args.cells[0].split("+")])] * len(
            data.keys()
        )

threshold = args.threshold / 100

# plt.subplots(1, number_of_plots, figsize=(7.2 * number_of_plots, 4.8))
# plt.subplot(1, number_of_plots, 1)

channels = {}

for kidx, ikey in enumerate(data.keys()):

    if usecase != "F01":

        trace_type = data[ikey][key[kidx]].get("Mode", None)

        if trace_type is not None:
            # currently PUSCH1 and PUSCH2 are separated
            # PUCCH1 and PUCCH2 are still combined
            ul_cells = data[ikey][key[kidx]].get("PUSCH1", [])
            
            ul_cells2 = data[ikey][key[kidx]].get("PUSCH2", [])

            cul_cells = data[ikey][key[kidx]].get("PUCCH1", [])
        
            cul_cells2 = data[ikey][key[kidx]].get("PUCCH2", [])

            cul_cells.extend(cul_cells2)
        else:
            ul_cells = data[ikey][key[kidx]].get("Total", [])
            if len(ul_cells) == 0:
                ul_cells = data[ikey][key[kidx]].get("PUSCH", [])

            cul_cells = data[ikey][key[kidx]].get("PUCCH", [])

    else:
        if args.is_fdd_pusch:

            raw = data[ikey][key[kidx]].get("PDSCH", [])

            if type(raw[0]) == list:
                dl_cells = list(itertools.chain.from_iterable(raw))
            else:
                dl_cells = raw

            ul_cells = list(
                np.array(data[ikey][key[kidx]].get("PUSCH", [])) - np.array(dl_cells)
            )
        else:
            ul_cells = data[ikey][key[kidx]].get("Total", [])
            if len(ul_cells) == 0:
                ul_cells = data[ikey][key[kidx]].get("PUSCH", [])

    if len(ul_cells) > 0:
        y, x = np.histogram(ul_cells, bins=10000)
        cy = np.cumsum(y) / len(ul_cells)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("PUSCH1", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["PUSCH1"] = store

    if len(ul_cells2) > 0:
        y, x = np.histogram(ul_cells2, bins=10000)
        cy = np.cumsum(y) / len(ul_cells2)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("PUSCH2", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["PUSCH2"] = store

    if len(cul_cells) > 0:

        y, x = np.histogram(cul_cells, bins=10000)
        cy = np.cumsum(y) / len(cul_cells)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("PUCCH", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["PUCCH"] = store

    raw = data[ikey][key[kidx]].get("PDSCH", [])

    if len(raw) > 0:

        if usecase == "F01":
            if type(raw[0]) == list:
                dl_cells = list(itertools.chain.from_iterable(raw))
            else:
                dl_cells = raw
        else:
            dl_cells = []

            if type(raw) == dict:
                for ikey in raw.keys():
                    dl_cells.extend(itertools.chain.from_iterable(raw[ikey]))
            elif type(raw) == list:
                dl_cells = raw

        y, x = np.histogram(dl_cells, bins=10000)
        cy = np.cumsum(y) / len(dl_cells)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("PDSCH", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["PDSCH"] = store

    if usecase != "F01":

        buffer = data[ikey][key[kidx]].get("BWC", [])

        if len(buffer) > 0:

            y, x = np.histogram(np.array(buffer), bins=10000)
            cy = np.cumsum(y) / len(buffer)

            if args.is_filter:
                for idx, item in enumerate(x):
                    if cy[idx] > threshold:
                        x = x[: idx + 1]
                        cy = cy[:idx]
                        break

            store = channels.get("BWC", {})

            store[ikey] = {}

            store[ikey]["x"] = x[1:]
            store[ikey]["y"] = cy

            channels["BWC"] = store

            # add PDSCH + BWC
            raw_pdsch = data[ikey][key[kidx]].get("PDSCH", [])
            pdsch_bwc = raw_pdsch + buffer
            
            y, x = np.histogram(np.array(pdsch_bwc), bins=10000)
            cy = np.cumsum(y) / len(pdsch_bwc)

            if args.is_filter:
                for idx, item in enumerate(x):
                    if cy[idx] > threshold:
                        x = x[: idx + 1]
                        cy = cy[:idx]
                        break

            store = channels.get("PDSCH+BWC", {})

            store[ikey] = {}

            store[ikey]["x"] = x[1:]
            store[ikey]["y"] = cy

            channels["PDSCH+BWC"] = store

    raw = data[ikey][key[kidx]].get("PDCCH", [])

    if len(raw) > 0:

        if usecase == "F01":
            if type(raw[0]) == list:
                cdl_cells = list(itertools.chain.from_iterable(raw))
            else:
                cdl_cells = raw
        else:
            cdl_cells = raw

        y, x = np.histogram(cdl_cells, bins=10000)
        cy = np.cumsum(y) / len(cdl_cells)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("PDCCH", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["PDCCH"] = store

    buffer = data[ikey][key[kidx]].get("CSI-RS", [])

    if len(buffer) > 0:

        y, x = np.histogram(np.array(buffer) + np.array(raw), bins=10000)
        cy = np.cumsum(y) / len(buffer)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("CSI-RS", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["CSI-RS"] = store

    buffer = data[ikey][key[kidx]].get("SRS1", [])

    if len(buffer) > 0:

        y, x = np.histogram(buffer, bins=10000)
        cy = np.cumsum(y) / len(buffer)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("SRS1", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["SRS1"] = store

    buffer = data[ikey][key[kidx]].get("SRS2", [])

    if len(buffer) > 0:

        y, x = np.histogram(buffer, bins=10000)
        cy = np.cumsum(y) / len(buffer)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("SRS2", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["SRS2"] = store

    buffer = data[ikey][key[kidx]].get("PRACH", [])

    if len(buffer) > 0:

        y, x = np.histogram(buffer, bins=10000)
        cy = np.cumsum(y) / len(buffer)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("PRACH", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["PRACH"] = store

    buffer = data[ikey][key[kidx]].get("SSB", [])

    if len(buffer) > 0:

        y, x = np.histogram(buffer, bins=10000)
        cy = np.cumsum(y) / len(buffer)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get("SSB", {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels["SSB"] = store

sz_channels = len(channels.keys())
if "CSI-RS" in channels.keys():
    sz_channels -= 1

cols = np.min([3, sz_channels])
rows = int(np.ceil(sz_channels / 3))

plt.subplots(rows, cols, figsize=(7.2 * cols, 4.8 * rows))

offset = 0

for idx, key in enumerate(list(channels.keys())):

    local_legend = []

    if "CSI-RS" in list(channels.keys()):
        if key == "PDCCH":
            offset = 1
            continue

    plt.subplot(rows, cols, idx - offset + 1)

    for iidx, ikey in enumerate(list(channels[key].keys())):

        x = channels[key][ikey]["x"]
        y = channels[key][ikey]["y"]

        label = legend[ikey]

        local_legend.append(label)

        plt.plot(x, y, color=f"C{ikey}")

    if key == "PUSCH1":

        if usecase == "F01":
            if args.is_fdd_pusch:
                if args.is_disable_uc:
                    plt.title("PUSCH1")
                else:
                    plt.title(f"{usecase}: PUSCH1")
            else:
                if args.is_disable_uc:
                    plt.title("PDSCH + PUSCH")
                else:
                    plt.title(f"{usecase}: PDSCH + PUSCH")
            plt.vlines(1000, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PUSCH1")
            else:
                plt.title(f"{usecase}: PUSCH1")
            plt.vlines(1500, 0, 1, color="k")

    elif key == "PUSCH2":

        if usecase == "F01":
            if args.is_fdd_pusch:
                if args.is_disable_uc:
                    plt.title("PUSCH2")
                else:
                    plt.title(f"{usecase}: PUSCH2")
            else:
                if args.is_disable_uc:
                    plt.title("PDSCH + PUSCH")
                else:
                    plt.title(f"{usecase}: PDSCH + PUSCH")
            plt.vlines(1000, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PUSCH2")
            else:
                plt.title(f"{usecase}: PUSCH2")
            plt.vlines(1500, 0, 1, color="k")
            
    elif key == "PDSCH":
        if usecase == "F01":
            if args.is_disable_uc:
                plt.title("PDSCH")
            else:
                plt.title(f"{usecase}: PDSCH")
            plt.vlines(750, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PDSCH")
            else:
                plt.title(f"{usecase}: PDSCH")

            if channels.get("BWC", None) is not None:
                plt.vlines(375, 0, 1, color="k")
            else:
                plt.vlines(375, 0, 1, color="k")

    elif key == "PDSCH+BWC":
        if args.is_disable_uc:
            plt.title("PDSCH+BWC")
        else:
            plt.title(f"{usecase}: PDSCH+BWC")
        plt.vlines(500, 0, 1, color="k")

    elif key == "PDCCH":
        if usecase == "F01":
            if args.is_disable_uc:
                plt.title("PDCCH")
            else:
                plt.title(f"{usecase}: PDCCH")
            plt.vlines(750, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PDCCH")
            else:
                plt.title(f"{usecase}: PDCCH")
            plt.vlines(375, 0, 1, color="k")

    elif key == "CSI-RS":
        if usecase == "F01":
            if args.is_disable_uc:
                plt.title("PDCCH+CSI-RS")
            else:
                plt.title(f"{usecase}: PDCCH+CSI-RS")
            plt.vlines(750, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PDCCH+CSI-RS")
            else:
                plt.title(f"{usecase}: PDCCH+CSI-RS")
            plt.vlines(375, 0, 1, color="k")

    elif key == "PUCCH":
        if usecase == "F01":
            raise NotImplementedError
        else:
            if args.is_disable_uc:
                plt.title("PUCCH")
            else:
                plt.title(f"{usecase}: PUCCH")
            plt.vlines(1500, 0, 1, color="k")

    elif key == "BWC":
        if args.is_disable_uc:
            plt.title("BWC")
        else:
            plt.title(f"{usecase}: BWC")
        plt.vlines(500, 0, 1, color="k")

    elif key == "SRS1":
        if args.is_disable_uc:
            plt.title("SRS1")
        else:
            plt.title(f"{usecase}: SRS1")
        plt.vlines(500, 0, 1, color="k")

    elif key == "SRS2":
        if args.is_disable_uc:
            plt.title("SRS1")
        else:
            plt.title(f"{usecase}: SRS1")
        plt.vlines(500, 0, 1, color="k")

    elif key == "SSB":
        if args.is_disable_uc:
            plt.title("SSB")
        else:
            plt.title(f"{usecase}: SSB")
        plt.vlines(375, 0, 1, color="k")

    elif key == "PRACH":
        if args.is_disable_uc:
            plt.title("PRACH")
        else:
            plt.title(f"{usecase}: PRACH")
        if trace_type == "Sequential":
            plt.vlines(1500, 0, 1, color="k")
        elif trace_type == "Parallel":
            plt.vlines(1250, 0, 1, color="k")
        else:
            plt.vlines(2000, 0, 1, color="k")
    else:
        raise NotImplementedError

    local_legend.append("Constraint")
    plt.legend(local_legend)

    plt.grid(True)
    plt.ylabel("CDF")
    plt.xlabel("Latency [us]")

plt.tight_layout()

time = datetime.datetime.now()
buffer = "_".join([str(time.year), str(time.month).zfill(2), str(time.day).zfill(2)])

plt.savefig(f"compare-{buffer}.png")
