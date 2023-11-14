# cuBB system checks

Script for dumping system configuration relevant for running cuBB.

The goal of `cuBB_system_checks.py` is to verify that a platform on which the script is executed is correctly configured to run the cuBB software stack.

`cuBB_system_checks.py` provides a standardized view of the system which makes finding any misconfigurations easier.


## Example commands

Show help message:
```
./cuBB_system_checks.py -h
```

Dump cuBB-related system configuration:
```
sudo -E ./cuBB_system_checks.py -bcegilmnps
```

Dump NIC configuration only:
```
sudo -E ./cuBB_system_checks.py --nic
```

## Reference setup

Reference configuration for 21-4 Aerial release:
```
-----General--------------------------------------
Hostname                           : dc6-aerial-devkit-07
IP address                         : 10.152.138.75
Linux distro                       : "Ubuntu 20.04.2 LTS"
Linux kernel version               : 5.4.0-65-lowlatency
-----Kernel Command Line--------------------------
Audit subsystem                    : audit=0
Clock source                       : clocksource=tsc
HugePage count                     : hugepages=16
HugePage size                      : hugepagesz=1G
CPU idle time management           : idle=poll
Max Intel C-state                  : intel_idle.max_cstate=0
Intel IOMMU                        : intel_iommu=off
IOMMU                              : iommu=off
Isolated CPUs                      : isolcpus=2-21
Corrected errors                   : mce=ignore_ce
Adaptive-tick CPUs                 : nohz_full=2-21
Soft-lockup detector disable       : nosoftlockup
Max processor C-state              : processor.max_cstate=0
RCU callback polling               : rcu_nocb_poll
No-RCU-callback CPUs               : rcu_nocbs=2-21
TSC stability checks               : tsc=reliable
-----CPU------------------------------------------
CPU cores                          : 24
Thread(s) per CPU core             : 1
CPU MHz:                           : 3200.000
CPU sockets                        : 1
-----Environment variables------------------------
CUDA_DEVICE_MAX_CONNECTIONS        : 16
cuBB_SDK                           : /home/kkoch/cuBB
-----Memory---------------------------------------
HugePage count                     : 16
Free HugePages                     : 15
HugePage size                      : 1048576 kB
Shared memory size                 : 47G
-----Nvidia GPUs----------------------------------
GPU driver version                 : 470.57.02
CUDA version                       : 11.4
GPU0
  GPU product name                 : NVIDIA A100-PCIE-40GB
  GPU persistence mode             : Enabled
  Current GPU temperature          : 29 C
  GPU clock frequency              : 1410 MHz
  Max GPU clock frequency          : 1410 MHz
  GPU PCIe bus id                  : 00000000:B6:00.0
-----GPUDirect topology---------------------------
        GPU0    mlx5_0  mlx5_1  CPU Affinity    NUMA Affinity
GPU0     X      PIX     PIX     0-23            N/A
mlx5_0  PIX      X      PIX
mlx5_1  PIX     PIX      X
 
Legend:
 
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
 
-----Mellanox NICs--------------------------------
NIC0
  NIC product name                 : ConnectX6DX
  NIC part number                  : MCX623106AE-CDA_Ax
  NIC PCIe bus id                  : 0000:b5:00.0
  NIC FW version                   : 22.31.1014
  FLEX_PARSER_PROFILE_ENABLE       : 4
  PROG_PARSE_GRAPH                 : True(1)
  ACCURATE_TX_SCHEDULER            : True(1)
  CQE_COMPRESSION                  : AGGRESSIVE(1)
  REAL_TIME_CLOCK_ENABLE           : True(1)
-----Mellanox NIC Interfaces----------------------
Interface0
  Name                             : ens6f0
  Network adapter                  : mlx5_0
  PCIe bus id                      : 0000:b5:00.0
  Ethernet address                 : b8:ce:f6:33:fe:0e
  Operstate                        : up
  MTU                              : 1536
  RX flow control                  : off
  TX flow control                  : off
  PTP hardware clock               : 3
  QoS Priority trust state         : dscp
  PCIe MRRS                        : 4096 bytes
Interface1
  Name                             : ens6f1
  Network adapter                  : mlx5_1
  PCIe bus id                      : 0000:b5:00.1
  Ethernet address                 : b8:ce:f6:33:fe:0f
  Operstate                        : up
  MTU                              : 1514
  RX flow control                  : off
  TX flow control                  : off
  PTP hardware clock               : 4
  QoS Priority trust state         : dscp
  PCIe MRRS                        : 4096 bytes
-----Software Packages----------------------------
cmake       /usr/local/bin         : 3.19.7
docker      /usr/bin               : 19.03.13
gcc         /usr/bin               : 9.3.0
git-lfs     /usr/bin               : 2.9.2
MOFED                              : 5.4-1.0.3.0
meson       /usr/local/bin         : 0.56.2
ninja       /usr/bin               : 1.10.0
ptp4l       /usr/sbin              : 1.9.2-1
-----Loaded Kernel Modules------------------------
GDRCopy                            : gdrdrv
GPUDirect RDMA                     : nvidia_peermem
Nvidia                             : nvidia
-----Non-persistent settings----------------------
VM swappiness                      : vm.swappiness = 0
VM zone reclaim mode               : vm.zone_reclaim_mode = 0
```
