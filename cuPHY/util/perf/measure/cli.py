# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from importlib import util

from .FDD.properties import avg_subs, het_subs


def arguments():

    base = argparse.ArgumentParser()
    base.add_argument(
        "--cuphy",
        type=str,
        dest="cfld",
        help="Specifies the folder where cuPHY has been built",
        required=True,
    )
    base.add_argument(
        "--vectors",
        type=str,
        dest="vfld",
        help="Specifies the folder for the test vectors",
        required=True,
    )
    base.add_argument(
        "--config",
        type=str,
        dest="config",
        help="Specifies the file contaning the test cases list",
        required=True,
    )
    base.add_argument(
        "--uc",
        type=str,
        dest="uc",
        help="Specifies the file contaning the use case config",
        required=True,
    )
    base.add_argument(
        "--target",
        type=str,
        nargs="+",
        dest="target",
        help="Specifies the SMs used by each sub-CTX",
    )
    base.add_argument(
        "--delay",
        type=int,
        dest="delay",
        default=1000,
        help="Specifies the duration of the delay kernel",
    )
    base.add_argument(
        "--gpu",
        type=int,
        dest="gpu",
        default=0,
        help="Specifies on which GPU to run the measurements",
    )
    base.add_argument(
        "--freq",
        type=int,
        dest="freq",
        help="Specifies the frequency at which the GPU will be set for the measurements",
        required=True,
    )
    base.add_argument(
        "--graph",
        action="store_true",
        dest="is_graph",
        default=False,
        help="Specifies whether to use graphs rather than streams for FDD use cases",
    )
    base.add_argument(
        "--start",
        type=int,
        dest="start",
        default=1,
        help="Specifies the minimum numbers of cells to try",
    )
    base.add_argument(
        "--cap",
        type=int,
        dest="cap",
        help="Specifies the maximum numbers of cells to try",
        required=True,
    )
    base.add_argument(
        "--iterations",
        type=int,
        dest="iterations",
        default=1,
        help="Specifies number of iterations to use in averaging latency results",
    )
    base.add_argument(
        "--slots",
        type=int,
        dest="sweeps",
        default=1,
        help="Specifies number of sweep iterations",
    )
    base.add_argument(
        "--power",
        type=int,
        dest="power",
        help="Specifies the maximum power draw for the GPU used for the measurements",
    )
    base.add_argument(
        "--fdd_subs",
        type=int,
        dest="subs",
        help="Specifies the number of sub-CTXs to use",
    )
    base.add_argument(
        "--force",
        type=int,
        dest="force",
        help="Specifies the number of connections to use",
    )
    base.add_argument(
        "--priority",
        action="store_true",
        dest="is_priority",
        help="Specifies whether PDSCH has higher priority over PUSCH",
    )
    base.add_argument(
        "--mig",
        type=int,
        dest="mig",
        help="Specifies the MIG ID to use to create the GIs",
    )
    base.add_argument(
        "--mig_instances",
        type=int,
        dest="mig_instances",
        default=1,
        help="Specifies the number of MIG GPU-instances to run with",
    )
    base.add_argument(
        "--disable_mps",
        action="store_true",
        dest="is_no_mps",
        help="Not supported any longer",
    )
    base.add_argument(
        "--seed",
        type=int,
        dest="seed",
        help="Specifies the seed to use in the simulations",
    )
    base.add_argument(
        "--measure_power",
        action="store_true",
        dest="is_power",
        default=False,
        help="Specifies whether to use the bench for measuring power rather than latency",
    )
    base.add_argument(
        "--test",
        action="store_true",
        dest="is_test",
        help="Specifies whether to enable test mode",
    )
    base.add_argument(
        "--debug",
        action="store_true",
        dest="is_debug",
        help="Specifies whether to enable debug mode",
    )
    base.add_argument(
        "--debug_mode",
        type=str,
        dest="debug_mode",
        choices=["cta", "triage", "nsys", "ncu", "incu"],
        default="cta",
        help="Specifies which debug mode to use",
    )
    base.add_argument(
        "--rec_bf",
        action="store_true",
        dest="is_rec_bf",
        help="Specifies whether the use case involves reciprocal beamforming",
    )
    base.add_argument(
        "--prach",
        action="store_true",
        dest="is_prach",
        help="Specifies whether the use case involves PRACH",
    )
    base.add_argument(
        "--prach_isolate",
        action="store_true",
        dest="is_isolated_prach",
        help="Specifies whether PRACH needs to run on its own sub-CTX",
    )
    base.add_argument(
        "--ssb",
        action="store_true",
        dest="is_ssb",
        help="Specifies whether the use case involves SSB",
    )
    base.add_argument(
        "--csirs",
        action="store_true",
        dest="is_csirs",
        help="Specifies whether the use case involves CSI-RS",
    )
    base.add_argument(
        "--prach_tgt",
        type=int,
        dest="prach_tgt",
        help="Internal",
    )
    base.add_argument(
        "--pdcch",
        action="store_true",
        dest="is_pdcch",
        help="Specifies whether the use case involves PDCCH",
    )
    base.add_argument(
        "--pdcch_isolate",
        action="store_true",
        dest="is_isolated_pdcch",
        help="Specifies whether PDCCH needs to run on its own sub-CTX",
    )
    base.add_argument(
        "--pucch",
        action="store_true",
        dest="is_pucch",
        help="Specifies whether the use case involves PUCCH",
    )
    base.add_argument(
        "--pucch_isolate",
        action="store_true",
        dest="is_isolated_pucch",
        help="Specifies whether PUCCH needs to run on its own sub-CTX",
    )
    base.add_argument(
        "--unsafe",
        action="store_true",
        dest="is_unsafe",
        help="Specifies whether to measure power without timeouts",
    )
    base.add_argument(
        "--groups_dl",
        action="store_true",
        dest="is_groups_pdsch",
        help="Specifies whether to use cell groups for PDSCH",
    )
    base.add_argument(
        "--pack_pdsch",
        action="store_true",
        dest="is_pack_pdsch",
        help="Specifies whether to use packed cell groups for PDSCH",
    )
    base.add_argument(
        "--groups_pusch",
        action="store_true",
        dest="is_groups_pusch",
        help="Specifies whether to use cell groups for PUSCH",
    )
    base.add_argument(
        "--disable_pusch",
        action="store_true",
        dest="is_no_pusch",
        help="Specifies whether to simulate PUSCH",
    )
    base.add_argument(
        "--disable_pdsch",
        action="store_true",
        dest="is_no_pdsch",
        help="Specifies whether to simulate PDSCH",
    )
    base.add_argument(
        "--check_traffic",
        action="store_true",
        dest="is_check_traffic",
        help="Specifies whether to check for functional error in the traffic",
    )
    base.add_argument(
        "--numa", type=int, dest="numa", help="Specifies the NUMA node to use"
    )
    base.add_argument(
        "--2cb_per_sm",
        action="store_true",
        dest="is_2_cb_per_sm",
        help="Specifies whether to enable 2CB/SM on GA100",
    )
    base.add_argument(
        "--tdd_pattern",
        type=str,
        dest="pattern",
        choices=["dddsu", "dddsuudddd", "dsuuu"],
        default="dddsu",
        help="Specifies the TDD pattern to run",
    )
    base.add_argument(
        "--save_buffers",
        action="store_true",
        dest="is_save_buffers",
        help="Specifies whether to save intermediate buffers (normally erased)",
    )
    base.add_argument(
        "--pusch_cascaded",
        action="store_true",
        dest="is_pusch_cascaded",
        help="Specifies whether for, DDDSUUDDDD, the second UL slot needs to be processed after the first",
    )
    base.add_argument("--triage_start", type=int, dest="triage_start", help="Internal")
    base.add_argument("--triage_end", type=int, dest="triage_end", help="Internal")
    base.add_argument(
        "--triage_sample", type=int, default=1024, dest="triage_sample", help="Internal"
    )

    base.add_argument(
        "--ldpc_parallel",
        action="store_true",
        dest="is_ldpc_parallel",
        help="Specifies whether for the PUSCH LDPC decoder runs the TB in parallel rather than serially",
    )

    base.add_argument(
        "--srs_isolate",
        action="store_true",
        dest="is_srs_isolate",
        help="Specifies whether SRS needs to run on its own sub-CTX",
    )

    args = base.parse_args()

    if args.start > args.cap:
        base.error(
            "The minimum number of cells to try cannot be higher than the maximum"
        )

    if args.is_no_mps:
        args.is_no_mps = False

    if "FDD" in args.uc:
        if "_avg_" in args.uc:
            if args.subs is not None:
                if args.subs not in avg_subs:
                    base.error(
                        f"At this stage, the number of sub-CTXs can only be chosen in {avg_subs}"
                    )
        else:
            if args.subs is not None:
                if args.subs not in het_subs:
                    base.error(
                        f"At this stage, the number of sub-CTXs can only be chosen in {het_subs}"
                    )

        if args.subs is not None:
            if args.subs > 1:
                if args.is_no_mps:
                    base.error(
                        "Disabling MPS is not supported from a number of sub-CTXs larger than 1"
                    )

    if "TDD" in args.uc:
        if args.subs is not None:
            base.error("The number of sub-CTXs to use cannot be set with TDD use cases")
        if "_het_" in args.uc:
            if args.pattern == "dddsuudddd":
                base.error(
                    "extended TDD pattern is not supported for heterogeneous traffic"
                )

    if args.is_rec_bf:
        if (
            "TDD" not in args.uc
            or "_avg_" not in args.uc
            or ("F14" not in args.uc and "F09" not in args.uc)
        ):
            base.error(
                "Reciprocal beamforming can only be activated with the F09/F14 use cases with avg. cells"
            )

    if args.is_pdcch:
        if "TDD" not in args.uc or "_avg_" not in args.uc or args.is_no_pdsch:
            base.error(
                "PDCCH can only be activated with avg. cells, when PDSCH is also present"
            )

    if args.is_pucch:
        if "TDD" not in args.uc or "_avg_" not in args.uc or args.is_no_pusch:
            base.error(
                "PUCCH can only be activated with avg. cells, when PUSCH is also present"
            )

    if args.is_debug:
        if args.mig is not None:
            if args.mig_instances > 1:
                base.error(
                    "Profiling with multiple CPU processes running in parallel is not supported"
                )

        if "FDD" in args.uc:
            buffer = util.find_spec("measure.FDD.configure_debug")
            if buffer is None:
                base.error(
                    "This is a released version of the performance scripts, and debug mode cannot be enabled."
                )

        else:
            buffer = util.find_spec("measure.TDD.configure_debug")
            if buffer is None:
                base.error(
                    "This is a released version of the performance scripts, and debug mode cannot be enabled."
                )

    if args.is_debug:
        if args.debug_mode == "triage":
            if (
                args.triage_start is None
                or args.triage_end is None
                or args.triage_sample is None
            ):
                base.error("Triage mode is not supported without all of its parameters")

    if args.is_power:
        if args.is_debug:
            if args.debug_mode not in ["nsys"]:
                if "TDD" not in args.uc:
                    base.error(
                        "For power measurements, only trace with Nsight System is supported, and exclusively for F14 use case"
                    )
    else:
        if args.is_unsafe:
            print("Warning: --unsafe is only applicable for power measurements")

    if args.is_rec_bf or args.is_prach:
        if "TDD" not in args.uc or "_avg_" not in args.uc:
            base.error(
                "PRACH and/or reciprocal beamforming can only be enabled for the TDD use case with the peak + avg traffic model"
            )

    if args.is_pdcch or args.is_pucch or args.is_ssb or args.is_csirs:
        if "TDD" not in args.uc or "_avg_" not in args.uc:
            base.error(
                "PDCCH, PUCCH, SSB and/or CSI-RS can only be activated for the TDD use case with the peak + avg traffic model"
            )

    selected_channels = sum(
        list(
            map(
                int,
                [
                    args.is_prach and args.is_isolated_prach,
                    args.is_pdcch and args.is_isolated_pdcch,
                    args.is_pucch and args.is_isolated_pucch,
                    not args.is_no_pdsch,
                    not args.is_no_pusch,
                    args.is_ssb,
                    args.is_srs_isolate,
                ],
            )
        )
    )

    if len(args.target) > 1 and len(args.target) != selected_channels:
        base.error(
            "The number of arguments for --target does no match the isolation strategy for the selected channels"
        )

    if args.prach_tgt is not None:
        if not args.is_prach or not args.is_isolated_prach:
            base.error(
                "--prach_tgt can only be enabled with --prach and --prach_isolate"
            )

    if args.is_pack_pdsch:
        if not args.is_groups_pdsch:
            base.error("--pack_pdsch can only be used with --groups_pdsch")

    # if args.is_groups_pdsch:
    #     if args.is_rec_bf or args.is_prach or args.is_pdcch:
    #         base.error("--groups_pdsch can only be used with PDSCH-only traffic for DL")

    if not args.is_no_mps:
        if args.target is None:
            base.error("Missing MPS target")

    if args.is_no_mps:
        if args.target is not None:
            print("Warning: with MPS disabled, the provided target will be ignored")

    if args.is_pusch_cascaded:
        if "TDD" not in args.uc or "_avg_" not in args.uc or args.pattern == "dddsu":
            base.error(
                "back-to-back UL workloads can only be enabled for TDD DDDSUUDDDD links and for the peak+avg. traffic model"
            )

    return (base, args)
