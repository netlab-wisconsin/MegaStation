# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# import numpy as np


def configure(args, mig, mig_gpu, connections, command, vectors, mode, k, target):

    if args.is_no_mps:
        if args.mig is None:
            system = f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
        else:
            system = (
                f"CUDA_VISIBLE_DEVICES={mig} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
            )
    else:
        if args.mig is None:
            system = f"CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS={connections}"
        else:
            system = f"CUDA_MPS_PIPE_DIRECTORY={mig_gpu} CUDA_LOG_DIRECTORY={mig_gpu} CUDA_DEVICE_MAX_CONNECTIONS={connections}"

    if args.pattern == "dddsu":
        if args.prach_tgt is not None:
            system = " ".join(
                [system, f"PRACH_TPC_MASK_LOW={format(args.prach_tgt,'#x')}"]
            )

    if args.numa is not None:
        system = " ".join(
            [system, f"numactl --cpunodebind={args.numa} --membind={args.numa}"]
        )

    if args.pattern == "dddsu":

        system = " ".join(
            [
                system,
                f"{command} -i {vectors} -r 1 -w 10000 -u 3 -d 0 -m {mode} -P {args.iterations} -W {args.delay}",
            ]
        )

    else:

        system = " ".join(
            [
                system,
                f"{command} -i {vectors} -r 1 -w 10000 -u 5 -d 0 -m {mode} -P {args.iterations} -W {args.delay}",
            ]
        )

        if args.is_pusch_cascaded:
            system = " ".join([system, "-B"])

    if args.is_ldpc_parallel:
        system = " ".join([system, "-K 1"])

    if not args.is_no_mps and args.is_prach and args.is_isolated_prach:
        system = " ".join([system, "--P"])

    if not args.is_no_mps and args.is_pdcch and args.is_isolated_pdcch:
        system = " ".join([system, "--Q"])

    if not args.is_no_mps and args.is_pucch and args.is_isolated_pucch:
        system = " ".join([system, "--X"])

    if args.is_groups_pdsch:
        system = " ".join([system, "--G"])
        if args.is_pack_pdsch:
            system = " ".join([system, "--b"])

    if args.is_groups_pusch:
        system = " ".join([system, "--g"])

    if not args.is_no_mps:
        flat_target = ",".join(map(str, target))
        system = " ".join([system, f"--M {flat_target}"])

    if args.is_2_cb_per_sm:
        system = " ".join([system, "-L"])

    if args.is_priority:
        system = " ".join([system, "-a"])

    if mig is None:
        system = " ".join([system, f">buffer-{str(k).zfill(2)}.txt"])
    else:
        system = " ".join([system, f">buffer-{mig_gpu}-{str(k).zfill(2)}.txt"])

    return system
