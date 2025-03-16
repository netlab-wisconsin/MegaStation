#pragma once

#include <parallel_hashmap/phmap.h>

#include <array>
#include <atomic>
#include <cstdint>

#include "config/config.h"
#include "defs.h"

namespace mega {

struct InstInfo {
  static constexpr uint8_t kMaxOut = 1;
  static constexpr uint8_t kMaxIn = 2;
  static constexpr uint8_t kMaxArg = 3;

  std::array<TaskType, kMaxIn> in_jobs;
  MemOpType out_mem;
  MemOpType op;

  uint16_t sm_util;
};

struct InstTable {
 private:
  static inline const TaskId get_prev_id(const TaskId &task_id,
                                         const TaskType &prev_job) {
    TaskId prev_task = {{task_id}, prev_job};
    if ((prev_job != TaskType::kLoad && task_id.job_id != TaskType::kStore) &&
        TaskTypeProp::symbol_type(prev_job) !=
            TaskTypeProp::symbol_type(task_id.job_id)) {
      // Prev Task is Pilot Beam
      prev_task.sym_id = gconfig.frame.pilots.back();
    }
    return prev_task;
  }

  std::array<InstInfo, TaskTypeProp::kNumTaskType> inst_table;
  phmap::node_hash_map<int,
                       std::array<uint64_t, TaskTypeProp::kNumTaskType + 2>>
      latency_table;           //!< Latency Table (sm -> task_id -> latency)
  std::vector<int> device_sm;  //!< Device SM

 public:
  InstTable() : device_sm(gconfig.num_devices) {
    cudaDeviceProp props;
    for (uint16_t i = 0; i < gconfig.num_devices; i++) {
      cudaGetDeviceProperties(&props, i);
      device_sm[i] = props.multiProcessorCount;
      latency_table[device_sm[i]].fill(0);
    }
    // TODO: provide sm_util values
    uint32_t scg_count =
        static_cast<uint32_t>(gconfig.ofdm_data / gconfig.sc_group);
    inst_table[static_cast<uint8_t>(TaskType::kNullT)] = {
        {TaskType::kNullT, TaskType::kNullT},
        MemOpType::kNullMO,
        MemOpType::kNullMO,
        0};
    inst_table[static_cast<uint8_t>(TaskType::kPFFT)] = {
        {TaskType::kLoad, TaskType::kNullT},
        MemOpType::kCSI,
        MemOpType::kPFFTOp,
        static_cast<uint16_t>(std::lround(gconfig.antennas / 9.0f))};
    printf("kPFFT: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kPFFT)].sm_util);
    inst_table[static_cast<uint8_t>(TaskType::kBeam)] = {
        {TaskType::kPFFT, TaskType::kNullT},
        MemOpType::kCSI,
        MemOpType::kBeamOp,
        static_cast<uint16_t>(std::lround((gconfig.ofdm_data * 4) / 2.0f))};
    printf("kBeam: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kBeam)].sm_util);
    inst_table[static_cast<uint8_t>(TaskType::kBeamNM)] = {
        {TaskType::kBeam, TaskType::kNullT},
        MemOpType::kNmCSI,
        MemOpType::kNullMO,
        static_cast<uint16_t>(std::lround(scg_count / 2.0f))};
    inst_table[static_cast<uint8_t>(TaskType::kUFFT)] = {
        {TaskType::kLoad, TaskType::kNullT},
        MemOpType::kUFFT,
        MemOpType::kUFFTOp,
        static_cast<uint16_t>(std::lround(gconfig.antennas / 9.0f))};
    printf("kUFFT: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kUFFT)].sm_util);
    uint32_t grid_x = (gconfig.users + 127) / 128, grid_y;
    if (gconfig.users <= 8) {
      grid_y = (gconfig.sc_group + 7) / 8;
    } else if (gconfig.users <= 16) {
      grid_y = (gconfig.sc_group + 15) / 16;
    } else if (gconfig.users <= 32) {
      grid_y = (gconfig.sc_group + 15) / 16;
    } else if (gconfig.users <= 64) {
      grid_y = (gconfig.sc_group + 31) / 32;
    } else {
      grid_y = (gconfig.sc_group + 31) / 32;
    }
    inst_table[static_cast<uint8_t>(TaskType::kEqual)] = {
        {TaskType::kBeam, TaskType::kUFFT},
        MemOpType::kEqual,
        MemOpType::kNullMO,
        static_cast<uint16_t>(std::lround(grid_x * grid_y * scg_count /
                                          7.0f))};  // TODO: 7 not correct
    printf("kEqual: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kEqual)].sm_util);
    inst_table[static_cast<uint8_t>(TaskType::kDecode)] = {
        {TaskType::kEqual, TaskType::kNullT},
        MemOpType::kDecode,
        MemOpType::kNullMO,
        static_cast<uint16_t>(std::lround(gconfig.users / 3.0f))};
    printf("kDecode: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kDecode)].sm_util);
    inst_table[static_cast<uint8_t>(TaskType::kEncode)] = {
        {TaskType::kLoad, TaskType::kNullT},
        MemOpType::kEncode,
        MemOpType::kNullMO,
        static_cast<uint16_t>(std::lround(gconfig.users / 4.0f))};
    printf("kEncode: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kEncode)].sm_util);
    inst_table[static_cast<uint8_t>(TaskType::kModulate)] = {
        {TaskType::kEncode, TaskType::kNullT},
        MemOpType::kModulate,
        MemOpType::kNullMO,
        static_cast<uint16_t>(std::lround(
            static_cast<uint64_t>(gconfig.users *
                                  ((gconfig.ofdm_data + 256 - 1) / 256)) /
            32.0f))};
    printf("kModulate: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kModulate)].sm_util);
    grid_x = (gconfig.antennas + 127) / 128;
    if (gconfig.antennas <= 8) {
      grid_y = (gconfig.sc_group + 7) / 8;
    } else if (gconfig.antennas <= 16) {
      grid_y = (gconfig.sc_group + 7) / 8;
    } else if (gconfig.antennas <= 32) {
      grid_y = (gconfig.sc_group + 15) / 16;
    } else if (gconfig.antennas <= 64) {
      if (gconfig.sc_group <= 16) {
        grid_x = 2;
        grid_y = 1;
      } else {
        grid_y = (gconfig.sc_group + 31) / 32;
      }
    } else {
      grid_y = (gconfig.sc_group + 31) / 32;
    }
    inst_table[static_cast<uint8_t>(TaskType::kPrecode)] = {
        {TaskType::kBeamNM, TaskType::kModulate},
        MemOpType::kPrecode,
        MemOpType::kNullMO,
        static_cast<uint16_t>(std::lround(grid_x * grid_y * scg_count /
                                          7.0f))};  // TODO 2 not correct
    printf("kPrecode: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kPrecode)].sm_util);
    inst_table[static_cast<uint8_t>(TaskType::kDiFFT)] = {
        {TaskType::kPrecode, TaskType::kNullT},
        MemOpType::kDiFFT,
        MemOpType::kDiFFTOp,
        static_cast<uint16_t>(std::lround(gconfig.antennas / 9.0f))};
    printf("kDiFFT: %d\n",
           inst_table[static_cast<uint8_t>(TaskType::kDiFFT)].sm_util);
  }

  inline const InstInfo operator[](const TaskId &task_id) const {
    if (task_id.job_id == TaskType::kLoad) {
      switch (gconfig.frame.gidx_sym[task_id.sym_id]) {
        case Config::NLPilot:
        case Config::Pilot:
        case Config::Uplink:
          return {
              {TaskType::kNullT, TaskType::kNullT},
              MemOpType::kPreFFT,
              MemOpType::kNullMO,
              0,
          };
        case Config::Downlink:
          return {
              {TaskType::kNullT, TaskType::kNullT},
              MemOpType::kUncode,
              MemOpType::kNullMO,
              0,
          };
        default:
          return inst_table[static_cast<uint8_t>(TaskType::kNullT)];
      }
    } else if (task_id.job_id == TaskType::kStore) {
      switch (gconfig.frame.gidx_sym[task_id.sym_id]) {
        case Config::Uplink:
          return {
              {TaskType::kDecode, TaskType::kNullT},
              MemOpType::kNullMO,
              MemOpType::kNullMO,
              0,
          };
        case Config::Downlink:
          return {
              {TaskType::kDiFFT, TaskType::kNullT},
              MemOpType::kNullMO,
              MemOpType::kNullMO,
              0,
          };
        default:
          return inst_table[static_cast<uint8_t>(TaskType::kNullT)];
      }
    }
    return inst_table[static_cast<uint8_t>(task_id.job_id)];
  }

  inline const uint16_t operator[](const TaskType &job_id) const {
    return inst_table[static_cast<uint8_t>(job_id)].sm_util;
  }

  inline const std::array<TaskId, InstInfo::kMaxIn> prev_tasks(
      const TaskId &cur_id) const {
    const InstInfo &cur_info = operator[](cur_id);
    std::array<TaskId, InstInfo::kMaxIn> prev_ids{};
    for (uint8_t i = 0; i < InstInfo::kMaxIn; i++) {
      if (cur_info.in_jobs[i] != TaskType::kNullT)
        prev_ids[i] = get_prev_id(cur_id, cur_info.in_jobs[i]);
    }
    return prev_ids;
  }

  inline const TaskId prev_task(const TaskId &cur_id) const {
    const InstInfo &cur_info = operator[](cur_id);
    if (cur_info.in_jobs[1] != TaskType::kNullT)
      return get_prev_id(cur_id, cur_info.in_jobs[1]);
    else if (cur_info.in_jobs[0] != TaskType::kNullT)
      return get_prev_id(cur_id, cur_info.in_jobs[0]);
    else
      return TaskId();
  }

  inline void update_latency(const TaskId &task_id, const DeviceId dev_id,
                             const uint32_t latency) {
    auto inst_latency = std::atomic_ref<uint64_t>(
        latency_table.at(device_sm[dev_id])
            .at(static_cast<uint8_t>(task_id.job_id)));
    uint32_t prev_latency = inst_latency.load(std::memory_order_consume);

    prev_latency == 0 ? inst_latency.store(latency, std::memory_order_release)
                      : inst_latency.store((prev_latency + latency) / 2,
                                           std::memory_order_release);
  }

  inline void update_latency(const uint8_t ind, const DeviceId dev_id,
                             const uint32_t latency) {
    uint8_t real_ind = ind == 0 ? ind : TaskTypeProp::kNumTaskType + ind - 1;
    auto inst_latency = std::atomic_ref<uint64_t>(
        latency_table.at(device_sm[dev_id]).at(real_ind));
    uint32_t prev_latency = inst_latency.load(std::memory_order_consume);

    prev_latency == 0 ? inst_latency.store(latency, std::memory_order_release)
                      : inst_latency.store((prev_latency + latency) / 2,
                                           std::memory_order_release);
  }

  inline uint32_t get_latency(const TaskId &task_id,
                              const DeviceId dev_id) const {
    auto inst_latency = std::atomic_ref<const uint64_t>(
        latency_table.at(device_sm[dev_id])
            .at(static_cast<uint8_t>(task_id.job_id)));

    return inst_latency.load(std::memory_order_relaxed);
  }

  inline uint32_t get_latency(const uint8_t ind, const DeviceId dev_id) const {
    uint8_t real_ind = ind == 0 ? ind : TaskTypeProp::kNumTaskType + ind - 1;
    auto inst_latency = std::atomic_ref<const uint64_t>(
        latency_table.at(device_sm[dev_id]).at(real_ind));

    return inst_latency.load(std::memory_order_relaxed);
  }

  inline int get_sm(const DeviceId dev_id) const {
    return device_sm.at(dev_id);
  }
};

}  // namespace mega