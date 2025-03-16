import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=str, help='The path to the `build` directory', default='./build/')
parser.add_argument('--ants', type=int, help='The number of antennas', default=256)
parser.add_argument('--users', type=int, help='The number of users', default=32)
parser.add_argument('--ofdm', type=int, help='The number of OFDM symbols', default=1216)
parser.add_argument('--sg', type=int, help='The number of subcarriers per group', default=32)
parser.add_argument('--dst', type=str, help='The destination directory', default='./')
args = parser.parse_args()

build_dir = args.build
dst_dir = args.dst
ants = args.ants
users = args.users
ofdm = args.ofdm
sg = args.sg

if build_dir[-1] != '/':
  build_dir += '/'

if dst_dir[-1] != '/':
  dst_dir += '/'

assert ofdm % sg == 0, "The number of OFDM symbols must be divisible by the number of subcarriers per group"

def generate_data(ants, users, ofdm, sg):
  global build_dir, dst_dir
  dir_ = build_dir
  dst_ = dst_dir
  dexec = "dgen"
  cexec = "convert"

  ps = (users + sg - 1) // sg
  us = 14 - ps
  ds = us

  os.system(f"{dir_}{dexec} -ue {users} -bs {ants} -ofdm_da {ofdm} -sc_group {sg} -num_pilots {ps} -num_uplinks {us} -num_downlinks {ds}")
  os.system(f"{dir_}{cexec} ant_data.data {dst_}tx_{ants}x{users}.data")
  os.system(f"rm -f ant_data.data")
  os.system(f"mv mac_data.data {dst_}mac_{ants}x{users}.data")

generate_data(ants, users, ofdm, sg)
