# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

if __name__ == "__main__":

    from measure.cli import arguments

    if os.geteuid() == 0:
        sudo = ""
    else:
        sudo = "sudo "

    base, args = arguments()

    if args.mig is not None:

        os.system(f"{sudo}nvidia-smi -i {args.gpu} -pm 1 >/dev/null")
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -mig 0 >/dev/null")
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -mig 1 >buffer.txt")

        ifile = open("buffer.txt", "r")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        if lines[0].split()[0] == "Enabled" and lines[0].split()[1] == "MIG":
            import measure.mig

            measure.mig.measure(base, args)
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -pm 0 >/dev/null")
        else:
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -pm 0 >/dev/null")
            base.error("encountered issues in enabling MIG on the selected GPU")

    else:
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -pm 1 >/dev/null")

        import measure.nomig

        measure.nomig.measure(base, args)

        os.system(f"{sudo}nvidia-smi -i {args.gpu} -pm 0 >/dev/null")
