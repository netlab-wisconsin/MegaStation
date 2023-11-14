#!/usr/bin/python3
# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import re
import subprocess
import logging
import psutil

KEY_LJUST_VALUE = 35
EXEC_PATH_LJUST_VALUE = 12
DELIMETER_CENTER_VALUE = 45
NOT_FOUND_STRING = 'N/A'

#--- Helper methods
def print_delimiter(name):
    print('-----' + name.ljust(DELIMETER_CENTER_VALUE, '-'))


def print_config(key, value):
    global KEY_LJUST_VALUE
    print(key.ljust(KEY_LJUST_VALUE) + ': ' + value)


def execute(command):
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True).stdout


def find_pattern_in_string(string, pattern, name, dump=True):
    value = re.search(pattern, string)
    if value:
        if dump: print_config(name, value.group(0))
    else:
        if dump: print_config(name, NOT_FOUND_STRING)


def find_value_after_pattern_in_string(string, pattern, name, dump=True):
    key = re.search(pattern, string)
    if key:
        end = key.end()
        value = string[end:string.index('\n', end)]
        if dump: print_config(name, value)
        return value
    else:
        if dump: print_config(name, NOT_FOUND_STRING)
        return None


def print_file_contents(filepath, name):
    with open(filepath, 'r') as f:
        print_config(name, f.read().rstrip())


def get_executable_path_dirname(executable):
    return os.path.dirname(execute(['which', executable]))


def dump_package_if_installed(command, regex, version_after_regex, name):
    try:
        result = execute(command)
        dirname = get_executable_path_dirname(name)
        if version_after_regex:
            find_value_after_pattern_in_string(result, regex, name.ljust(EXEC_PATH_LJUST_VALUE) + dirname)
        else:
            find_pattern_in_string(result, regex, name.ljust(EXEC_PATH_LJUST_VALUE) + dirname)
    except FileNotFoundError:
        print_config(name, NOT_FOUND_STRING)


#--- System checks
def dump_general_info():
    print_delimiter('General')
    print_file_contents('/etc/hostname', 'Hostname')
    result = execute(['hostname', '-I'])
    print_config('IP address', result.split()[0])
    result = execute(['cat', '/etc/os-release'])
    find_value_after_pattern_in_string(result, 'PRETTY_NAME=', 'Linux distro')
    result = execute(['uname', '-a'])
    find_pattern_in_string(result, '\d+\.\d+.\d+-\d+-[A-Za-z]*', 'Linux kernel version')

def dump_system_info():
    print_delimiter('System')
    result=execute(['dmidecode', '-t 1'])
    find_value_after_pattern_in_string(result, 'Manufacturer:\s+', 'Manufacturer')
    find_value_after_pattern_in_string(result, 'Product Name:\s+', 'Product Name')
    result=execute(['dmidecode', '-t 2'])
    find_value_after_pattern_in_string(result, 'Manufacturer:\s+', 'Base Board Manufacturer')
    find_value_after_pattern_in_string(result, 'Product Name:\s+', 'Base Board Product Name')
    result=execute(['dmidecode', '-t 3'])
    find_value_after_pattern_in_string(result, 'Manufacturer:\s+', 'Chassis Manufacturer')
    find_value_after_pattern_in_string(result, 'Type:\s+', 'Chassis Type')
    find_value_after_pattern_in_string(result, 'Height:\s+', 'Chassis Height')

    result=execute(['dmidecode', '-t 4'])
    find_value_after_pattern_in_string(result, 'Version:\s+', 'Processor')
    find_value_after_pattern_in_string(result, 'Max Speed:\s+', 'Max Speed')
    find_value_after_pattern_in_string(result, 'Current Speed:\s+', 'Current Speed')

def dump_kernel_cmdline():
    print_delimiter('Kernel Command Line')
    result = execute(['cat', '/proc/cmdline'])
    find_pattern_in_string(result, 'audit=(0|1|off|on)', 'Audit subsystem')
    find_pattern_in_string(result, 'clocksource=\S+', 'Clock source')
    find_pattern_in_string(result, 'hugepages=\d+', 'HugePage count')
    find_pattern_in_string(result, 'hugepagesz=\d+(G|M)', 'HugePage size')
    find_pattern_in_string(result, 'idle=[a-z]+', 'CPU idle time management')
    find_pattern_in_string(result, 'intel_idle.max_cstate=\d', 'Max Intel C-state')
    find_pattern_in_string(result, 'intel_iommu=[a-z\_]+', 'Intel IOMMU')
    find_pattern_in_string(result, 'iommu=[a-z]+', 'IOMMU')
    find_pattern_in_string(result, 'isolcpus=[0-9\-,]+', 'Isolated CPUs')
    find_pattern_in_string(result, 'mce=[a-z\_]+', 'Corrected errors')
    find_pattern_in_string(result, 'nohz_full=[0-9\-,]+', 'Adaptive-tick CPUs')
    find_pattern_in_string(result, 'nosoftlockup', 'Soft-lockup detector disable')
    find_pattern_in_string(result, 'processor.max_cstate=\d', 'Max processor C-state')
    find_pattern_in_string(result, 'rcu_nocb_poll', 'RCU callback polling')
    find_pattern_in_string(result, 'rcu_nocbs=[0-9\-,]+', 'No-RCU-callback CPUs')
    find_pattern_in_string(result, 'tsc=[a-z]+', 'TSC stability checks')


def dump_cpu_info():
    print_delimiter('CPU')
    result = execute(['lscpu'])
    find_value_after_pattern_in_string(result, 'CPU\(s\):\s+', 'CPU cores')
    find_value_after_pattern_in_string(result, 'Thread\(s\) per core:\s+', 'Thread(s) per CPU core')
    find_value_after_pattern_in_string(result, 'CPU MHz:\s+', 'CPU MHz:')
    find_value_after_pattern_in_string(result, 'Socket\(s\):\s+', 'CPU sockets')


def dump_memory_info():
    print_delimiter('Memory')
    result = execute(['cat', '/proc/meminfo'])
    find_value_after_pattern_in_string(result, 'HugePages_Total:\s+', 'HugePage count')
    find_value_after_pattern_in_string(result, 'HugePages_Free:\s+', 'Free HugePages')
    find_value_after_pattern_in_string(result, 'Hugepagesize:\s+', 'HugePage size')
    result = execute(['df', '-h', '|', 'grep', '/dev/shm'])
    find_pattern_in_string(result, '\d+[\.\d+](G|M)?', 'Shared memory size')


def dump_gpu_info():
    print_delimiter('Nvidia GPUs')

    try:
        result = execute(['nvidia-smi', '-q'])
    except FileNotFoundError:
        return

    find_value_after_pattern_in_string(result, 'Driver Version\s+:\s', 'GPU driver version')
    find_value_after_pattern_in_string(result, 'CUDA Version\s+:\s', 'CUDA version')

    gpu_idx = 0
    gpus = re.finditer('GPU [0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+', result)
    for gpu in gpus:
        print('GPU' + str(gpu_idx))
        gpu_info = result[gpu.end():]
        find_value_after_pattern_in_string(gpu_info, 'Product Name\s+:\s', '  GPU product name')
        find_value_after_pattern_in_string(gpu_info, 'Persistence Mode\s+:\s', '  GPU persistence mode')
        find_value_after_pattern_in_string(gpu_info, 'GPU Current Temp\s+:\s', '  Current GPU temperature')
        find_value_after_pattern_in_string(gpu_info, 'Clocks\s+Graphics\s+:\s', '  GPU clock frequency')
        find_value_after_pattern_in_string(gpu_info, 'Max Clocks\s+Graphics\s+:\s', '  Max GPU clock frequency')
        find_value_after_pattern_in_string(gpu_info, 'Bus Id\s+:\s', '  GPU PCIe bus id')
        gpu_idx += 1

    print_delimiter('GPUDirect topology')
    print(execute(['nvidia-smi', 'topo', '-m']))


def dump_nic_info():
    print_delimiter('Mellanox NICs')
    try:
        result = execute(['mlxfwmanager'])

        nic_idx = 0
        nics = re.finditer('Device \#\d+', result)
        for nic in nics:
            print('NIC' + str(nic_idx))
            nic_info = result[nic.end():]
            nic_bdf = find_value_after_pattern_in_string(nic_info, 'PCI Device Name:\s+', 'NIC', dump=False)

            find_value_after_pattern_in_string(nic_info, 'Device Type:\s+', '  NIC product name')
            find_value_after_pattern_in_string(nic_info, 'Part Number:\s+', '  NIC part number')
            find_value_after_pattern_in_string(nic_info, 'PCI Device Name:\s+', '  NIC PCIe bus id')
            find_pattern_in_string(nic_info, '\d+\.\d+\.\d+', '  NIC FW version')

            mlxconfig_result = execute(['mlxconfig', '-d', nic_bdf, 'q'])
            find_value_after_pattern_in_string(mlxconfig_result, 'FLEX_PARSER_PROFILE_ENABLE\s+', '  FLEX_PARSER_PROFILE_ENABLE')
            find_value_after_pattern_in_string(mlxconfig_result, 'PROG_PARSE_GRAPH\s+', '  PROG_PARSE_GRAPH')
            find_value_after_pattern_in_string(mlxconfig_result, 'ACCURATE_TX_SCHEDULER\s+', '  ACCURATE_TX_SCHEDULER')
            find_value_after_pattern_in_string(mlxconfig_result, 'CQE_COMPRESSION\s+', '  CQE_COMPRESSION')
            find_value_after_pattern_in_string(mlxconfig_result, 'REAL_TIME_CLOCK_ENABLE\s+', '  REAL_TIME_CLOCK_ENABLE')
            nic_idx += 1
    except FileNotFoundError:
        return


def dump_net_interface_info():
    print_delimiter('Mellanox NIC Interfaces')
    try:
        net_idx = 0
        ibdev2netdev = execute(['ibdev2netdev', '-v'])
        ports = re.findall('[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+', ibdev2netdev)
        ibdevs = re.findall('mlx5_\d+', ibdev2netdev)
        interfaces = re.findall('==> [a-z0-9]+', ibdev2netdev)

        for port in ports:
            print('Interface' + str(net_idx))
            ifc = interfaces[net_idx].split()[1]
            port_bdf = ports[net_idx]
            ibdev = ibdevs[net_idx]
            print_config('  Name', ifc)
            print_config('  Network adapter', ibdev)
            print_config('  PCIe bus id', port_bdf)
            result = execute(['ifconfig', ifc])
            print_file_contents('/sys/class/net/' + ifc + '/address', '  Ethernet address')
            print_file_contents('/sys/class/net/' + ifc + '/operstate', '  Operstate')
            print_file_contents('/sys/class/net/' + ifc + '/mtu', '  MTU')
            result = execute(['ethtool', '-a', ifc])
            find_value_after_pattern_in_string(result, 'RX:\s+', '  RX flow control')
            find_value_after_pattern_in_string(result, 'TX:\s+', '  TX flow control')
            result = execute(['ethtool', '-T', ifc])
            find_value_after_pattern_in_string(result, 'PTP Hardware Clock:\s+', '  PTP hardware clock')
            result = execute(['mlnx_qos', '-i', ifc])
            find_value_after_pattern_in_string(result, 'Priority trust state:\s+', '  QoS Priority trust state')
            result = execute(['lspci', '-s', port_bdf, '-vvv'])
            find_value_after_pattern_in_string(result, 'MaxReadReq\s+', '  PCIe MRRS')
            net_idx += 1
    except FileNotFoundError:
        return


def dump_linux_ptp_deamons_output():
    print_delimiter('Linux PTP')
    print(execute(['systemctl', 'status', 'ptp4l.service']))
    print(execute(['systemctl', 'status', 'phc2sys.service']))


def dump_required_packages_info():
    print_delimiter('Software Packages')
    dump_package_if_installed(['cmake', '--version'], 'cmake version\s+', True, 'cmake')
    dump_package_if_installed(['docker', '--version'], '\d+(\.\d+)+', False, 'docker')
    dump_package_if_installed(['gcc', '--version'], '\d+(\.\d+)+', False, 'gcc')
    dump_package_if_installed(['git-lfs'], '\d+\.\d+\.\d+', False, 'git-lfs')
    dump_package_if_installed(['ofed_info', '-s'], '\d+\.\d+-\d+\.\d+\.\d+\.\d+', False, 'MOFED')
    dump_package_if_installed(['meson', '--version'], '^', True, 'meson')
    dump_package_if_installed(['ninja', '--version'], '^', True, 'ninja')
    dump_package_if_installed(['dpkg', '-l', 'linuxptp'], '\d+(\.\d+)*-\d+(\.\d+)*', False, 'ptp4l')


def dump_kernel_modules():
    print_delimiter('Loaded Kernel Modules')
    result = execute(['lsmod'])
    find_pattern_in_string(result, 'gdrdrv', 'GDRCopy')
    find_pattern_in_string(result, '(nv_peer_mem|nvidia_peermem)', 'GPUDirect RDMA')
    find_pattern_in_string(result, 'nvidia', 'Nvidia')


def dump_non_persistent_settings():
    print_delimiter('Non-persistent settings')
    result = execute(['sysctl', '-a'])
    find_pattern_in_string(result, 'vm\.swappiness = \d+', 'VM swappiness')
    find_pattern_in_string(result, 'vm\.zone_reclaim_mode = \d+', 'VM zone reclaim mode')


def dump_envvars():
    print_delimiter('Environment variables')
    result = execute(['env'])
    find_value_after_pattern_in_string(result, 'CUDA_DEVICE_MAX_CONNECTIONS=', 'CUDA_DEVICE_MAX_CONNECTIONS')
    find_value_after_pattern_in_string(result, 'cuBB_SDK=', 'cuBB_SDK')


def dump_docker_info():
    try:
        print_delimiter('Docker images')
        print(execute(['docker', 'image', 'ls']))
        print_delimiter('Docker containers')
        print(execute(['docker', 'ps']))
    except FileNotFoundError:
        return



MISFIT_ALLOWLIST = [
    'cpuhp\/*',
    'cuphycontroller',
    'idle_inject\/*',
    'irq\/*',
    'kworker\/*',
    'migration\/*',
    'ru-emulator',
    'test_mac'
]

ISOLCPUS = range(2,22)

def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def dump_check_affinity():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    logging.warning('Start scanning for processes running on Aerial-reserved CPU cores.  Hit Control-C to quit.')
    while True:
        misfits = []
        for proc in psutil.process_iter():
            cores = set()
            cores.add(proc.cpu_num())
            for t in proc.threads():
                pt = psutil.Process(pid=t.id)

                cores.add(pt.cpu_num())

            cores_isect = intersection(cores,ISOLCPUS)
            if len(cores_isect) > 0:
                is_allowed_misfit = False
                for allow in MISFIT_ALLOWLIST:
                    m = re.search(allow,proc.name())
                    if m is not None:
                        is_allowed_misfit = True
                        break # out of inner for loop

                if is_allowed_misfit == False:
                    misfits.append((proc.name(), cores))

        for misfit in misfits:
            logging.warning(f"Found name: {misfit[0]} cores: {misfit[1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cuBB_system_info', description='Dump system information for running cuBB')
    parser.add_argument('-a', '--check_affinity', help='Check for processes with incorrect CPU affinity', action='store_true')
    parser.add_argument('-b', '--boot', help='Kernel boot parameters', action='store_true')
    parser.add_argument('-c', '--cpu', help='CPU', action='store_true')
    parser.add_argument('-d', '--docker', help='Docker images and containers', action='store_true')
    parser.add_argument('-e', '--envvar', help='Environment variables', action='store_true')
    parser.add_argument('-g', '--gpu', help='GPU', action='store_true')
    parser.add_argument('-i', '--interface', help='Net interface', action='store_true')
    parser.add_argument('-l', '--lkm', help='Loaded Kernel Modules (LKMs)', action='store_true')
    parser.add_argument('-m', '--memory', help='Memory', action='store_true')
    parser.add_argument('-n', '--nic', help='NICs', action='store_true')
    parser.add_argument('-p', '--packages', help='Software packages', action='store_true')
    parser.add_argument('--ptp', help='linuxptp deamons', action='store_true')
    parser.add_argument('-s', '--sysctl', help='Non-persistent system settings', action='store_true')
    parser.add_argument('--sys', help='System Info', action='store_true')
    args = parser.parse_args()

    check_affinity = False
    boot = True
    cpu = True
    docker = True
    envvar = True
    gpu = True
    interface = True
    lkm = True
    memory = True
    nic = True
    packages = True
    ptp = True
    sysctl = True
    sys = False

    #if any([args.check_affinity, args.boot, args.cpu, args.docker, args.envvar, args.gpu, args.interface, args.lkm, args.memory, args.nic, args.packages, args.ptp, args.sysctl]):
    #    (check_affinity, boot, cpu, docker, envvar, gpu, interface, lkm, memory, nic, packages, ptp, sysctl) = (args.boot, args.cpu, args.docker, args.envvar, args.gpu, args.interface, args.lkm, args.memory, args.nic, args.packages, args.ptp, args.sysctl)
    if any([args.check_affinity, args.boot, args.cpu, args.docker, args.envvar, args.gpu, args.interface, args.lkm, args.memory, args.nic, args.packages, args.ptp, args.sysctl, args.sys]):
        (check_affinity, boot, cpu, docker, envvar, gpu, interface, lkm, memory, nic, packages, ptp, sysctl, sys) = (args.check_affinity, args.boot, args.cpu, args.docker, args.envvar, args.gpu, args.interface, args.lkm, args.memory, args.nic, args.packages, args.ptp, args.sysctl, args.sys)

    dump_general_info()
    if sys:             dump_system_info()
    if check_affinity:  dump_check_affinity()
    if boot:            dump_kernel_cmdline()
    if cpu:             dump_cpu_info()
    if envvar:          dump_envvars()
    if memory:          dump_memory_info()
    if gpu:             dump_gpu_info()
    if nic:             dump_nic_info()
    if interface:       dump_net_interface_info()
    if ptp:             dump_linux_ptp_deamons_output()
    if packages:        dump_required_packages_info()
    if lkm:             dump_kernel_modules()
    if sysctl:          dump_non_persistent_settings()
    if docker:          dump_docker_info()
