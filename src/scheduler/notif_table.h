#pragma once

#include "atomic_table.h"
#include "defs.h"

namespace mega {

class NotifTable : AtomicTable<TaskStatus> {
 public:
  NotifTable(const uint32_t vframe_counts)
      : AtomicTable<TaskStatus>(vframe_counts *
                                TaskTypeProp::jobs_per_frame()) {}

  inline TaskStatus load(const TaskId &task_id) const {
    const uint32_t offset = TaskTypeProp::offset(task_id);
    return AtomicTable<TaskStatus>::load(offset);
  }

  inline TaskStatus wait(const TaskId &task_id) const {
    const uint32_t offset = TaskTypeProp::offset(task_id);
    auto atomic_val =
        std::atomic_ref<TaskStatus const>(atomic_table.at(offset));
    return atomic_val.load(std::memory_order_relaxed);
  }

  inline void store(const TaskId &task_id, const TaskStatus &val) {
    const uint32_t offset = TaskTypeProp::offset(task_id);
    AtomicTable<TaskStatus>::store(offset, val);
  }

  inline void clear(const FrameId &frame_id) {
    const uint32_t start = frame_id.frm_id * TaskTypeProp::jobs_per_frame();
    const uint32_t end = (frame_id.frm_id + 1) * TaskTypeProp::jobs_per_frame();
    std::fill(atomic_table.begin() + start, atomic_table.begin() + end,
              TaskStatus::kIdle);
  }
};

}  // namespace mega