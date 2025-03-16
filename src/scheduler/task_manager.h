#pragma once

#include "atomic_table.h"
#include "comm/range.h"
#include "defs.h"

namespace mega {

class DeviceTable : AtomicTable<DeviceId> {
 public:
  DeviceTable(const uint32_t vframe_counts)
      : AtomicTable<DeviceId>(vframe_counts * TaskTypeProp::jobs_per_frame()) {}

  inline DeviceId load(const TaskId &task_id) const {
    const uint32_t offset = TaskTypeProp::offset(task_id);
    return AtomicTable<DeviceId>::load(offset);
  }

  inline void store(const TaskId &task_id, const DeviceId &val) {
    const uint32_t offset = TaskTypeProp::offset(task_id);
    AtomicTable<DeviceId>::store(offset, val);
  }
};

class TimeTable : AtomicTable<uint64_t> {
 public:
  TimeTable(const uint32_t vframe_counts)
      : AtomicTable<uint64_t>(vframe_counts * TaskTypeProp::jobs_per_frame()) {}

  inline uint64_t load(const TaskId &task_id) const {
    const uint32_t offset = TaskTypeProp::offset(task_id);
    return AtomicTable<uint64_t>::load(offset);
  }

  inline void store(const TaskId &task_id, const uint64_t &val) {
    const uint32_t offset = TaskTypeProp::offset(task_id);
    AtomicTable<uint64_t>::store(offset, val);
  }
};

using RankDev = std::vector<DeviceId>;

class CSITable {
  std::vector<RankRange> csi_table;  //!< vframe_id -> rank_id -> Range
  std::vector<RankDev> dev_table;    //!< vframe_id -> rank_id -> dev_id

 public:
  CSITable(const uint32_t vframe_counts)
      : csi_table(vframe_counts, RankRange(gconfig.num_devices)),
        dev_table(vframe_counts, RankDev(gconfig.num_devices)) {}

  inline const RankRange &get(const FrameId &frame_id) const {
    return csi_table.at(frame_id.frm_id);
  }

  inline RankRange &ref_range(const FrameId &frame_id) {
    return csi_table.at(frame_id.frm_id);
  }

  inline const Range &get(const FrameId &frame_id,
                          const uint32_t &rank_id) const {
    return csi_table.at(frame_id.frm_id).at(rank_id);
  }

  inline const DeviceId &get_dev(const FrameId &frame_id,
                                 const uint32_t &rank_id) const {
    return dev_table.at(frame_id.frm_id).at(rank_id);
  }

  inline RankDev &ref_dev(const FrameId &frame_id) {
    return dev_table.at(frame_id.frm_id);
  }
};

class TaskManager {
 public:
  TaskManager(const uint32_t vframe_counts)
      : device_table(vframe_counts),
        time_table(vframe_counts),
        csi_table(vframe_counts) {}

  inline DeviceId load(const TaskId &task_id) const {
    return device_table.load(task_id);
  }

  inline void store(const TaskId &task_id, const DeviceId &val) {
    device_table.store(task_id, val);
  }

  inline uint64_t load_time(const TaskId &task_id) const {
    return time_table.load(task_id);
  }

  inline void store_time(const TaskId &task_id, const uint64_t &val) {
    time_table.store(task_id, val);
  }

 private:
  DeviceTable device_table;
  TimeTable time_table;

 public:
  CSITable csi_table;
};

}  // namespace mega