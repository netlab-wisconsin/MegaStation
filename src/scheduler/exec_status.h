#pragma once

#include <parallel_hashmap/phmap.h>

#include "defs.h"
#include "inst_table.h"
#include "notif_table.h"
#include "spin_lock.h"
#include "task_manager.h"

namespace mega {

class ExecStatus {
 private:
  using Weight = uint16_t;
  using TaskMap = phmap::node_hash_map<SymbolId, Weight>;

  SpinLock mutex;

  std::vector<TaskMap> task_status;  // dev_id -> sym_id -> weight

  const NotifTable &notif_table;
  const InstTable &inst_table;
  const TaskManager &task_table;

  DeviceId min_device(const TaskId &task_id);

 public:
  ExecStatus(const NotifTable &notif_table_, const InstTable &inst_table_,
             const TaskManager &task_table_)
      : mutex(),
        task_status(gconfig.num_devices),
        notif_table(notif_table_),
        inst_table(inst_table_),
        task_table(task_table_) {}

  void add_task(const DeviceId dev_id, const TaskId task_id);
  void apply_factor(const DeviceId dev_id, const TaskId task_id,
                    const float factor);
  void remove_sym(const SymbolId sym_id);

  DeviceId get_device(const TaskId &task_id);
  DeviceId get_minDevice(const TaskId &task_id);
};

}  // namespace mega