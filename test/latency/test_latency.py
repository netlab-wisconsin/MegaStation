import os

def run_latency_test(ants, users, ofdm, sg):
  dir_ = "~/MegaStation/build/"
  dexec = "dgen"
  cexec = "convert"
  texec = "test/latency/latency_breakdown"

  ps = (users + sg - 1) // sg
  us = 14 - ps
  ds = us

  os.system(f"{dir_}{dexec} -ue {users} -bs {ants} -ofdm_da {ofdm} -sc_group {sg} -num_pilots {ps} -num_uplinks {us} -num_downlinks {ds}")
  os.system(f"{dir_}{cexec} ant_data.data tx_{ants}x{users}.data")
  os.system(f"mv mac_data.data mac_{ants}x{users}.data")

  os.system(f"{dir_}{texec} -ants {ants} -users {users} -ofdm {ofdm} -sg {sg} -dir ./")

# Example usage
ants = 256
users = 32
ofdm = 1216
sg = 32

run_latency_test(ants, users, ofdm, sg)