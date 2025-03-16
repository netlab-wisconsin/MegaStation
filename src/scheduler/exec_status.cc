#include "exec_status.h"

#include <cstdint>
#include <numeric>

#include "config/config.h"
#include "defs.h"
#include "scheduler/spin_lock.h"

using namespace mega;

DeviceId ExecStatus::get_device(const TaskId &task_id) {
  LockGuard lock(mutex);

  auto prev_tasks = inst_table.prev_tasks(task_id);
  if (prev_tasks[0].job_id == TaskType::kNullT &&
      gconfig.frame.gidx_sym[task_id.sym_id] > Config::Pilot)
    // first task of non-pilot symbol
    prev_tasks[0] = {task_id.frm_id, gconfig.frame.pilots.back(),
                     TaskType::kBeam};

  std::array<DeviceId, prev_tasks.size()> prev_devs;
  std::transform(prev_tasks.begin(), prev_tasks.end(), prev_devs.begin(),
                 [&](const TaskId &prev_task) {
                   return prev_task.job_id == TaskType::kNullT
                              ? -1
                              : task_table.load(prev_task);
                 });

  DeviceId prev_dev = -1;
  Weight min_util;

  for (uint8_t i = 0; i < prev_tasks.size(); i++) {
    if (prev_devs[i] == -1) continue;
    if (prev_devs[i] == prev_dev) continue;

    const auto &task_indev = task_status[prev_devs[i]];
    const Weight util = std::accumulate(
        task_indev.begin(), task_indev.end(), Weight(0),
        [&](const Weight &acc, const auto &task) {
          return acc + (task.first == task_id ? 0 : task.second);
        });

    if (prev_dev == -1) {
      prev_dev = prev_devs[i];
      min_util = util;
      continue;
    } else if (util < min_util) {
      prev_dev = prev_devs[i];
      min_util = util;
    }
  }

  if (prev_dev == -1) return min_device(task_id);

  if (min_util <= inst_table.get_sm(prev_dev) + 10) return prev_dev;

  DeviceId min_dev = -1;
  for (DeviceId dev_id = 0; dev_id < gconfig.num_devices; dev_id++) {
    if (dev_id == prev_devs[0] || dev_id == prev_devs[1]) continue;

    Weight util = 0;
    bool exist_prev = false;
    for (const auto &task : task_status[dev_id]) {
      if (task.first.frm_id == task_id.frm_id) exist_prev = true;
      util += task.second;
    }
    if (exist_prev && util <= inst_table.get_sm(prev_dev) + 10) return dev_id;

    if (util < min_util) {
      prev_dev = dev_id;
      min_util = util;
    }
    if (min_util == 0) break;
  }

  if (min_dev == -1)
    return prev_dev;
  else
    return min_dev;
}

DeviceId ExecStatus::min_device(const TaskId &task_id) {
  return (task_id.frm_id + 1) % gconfig.num_devices;

  DeviceId min_dev = 0;
  Weight min_util =
      std::accumulate(task_status[min_dev].begin(), task_status[min_dev].end(),
                      Weight(0), [&](const Weight &acc, const auto &task) {
                        return acc + (task.first == task_id ? 0 : task.second);
                      });

  for (DeviceId dev_id = 1; dev_id < gconfig.num_devices; dev_id++) {
    if (min_util == 0) break;

    const Weight util = std::accumulate(
        task_status[dev_id].begin(), task_status[dev_id].end(), Weight(0),
        [&](const Weight &acc, const auto &task) {
          return acc + (task.first == task_id ? 0 : task.second);
        });

    if (util < min_util) {
      min_dev = dev_id;
      min_util = util;
    }
  }

  if (min_util <= inst_table.get_sm(min_dev))
    return min_dev;
  else
    return -1;
}

DeviceId ExecStatus::get_minDevice(const TaskId &task_id) {
  LockGuard lock(mutex);

  return min_device(task_id);
}

void ExecStatus::add_task(const DeviceId dev_id, const TaskId task_id) {
  LockGuard lock(mutex);

  TaskMap &status = task_status[dev_id];
  if (status.contains(task_id)) return;

  if (gconfig.frame.gidx_sym[task_id.sym_id] == Config::Pilot)
    status[task_id] = 40;
  else
    status[task_id] = 10;
}

void ExecStatus::apply_factor(const DeviceId dev_id, const TaskId task_id,
                              const float factor) {
  LockGuard lock(mutex);

  auto &task_weight = task_status[dev_id][task_id];

  task_weight = static_cast<Weight>(task_weight * factor);
}

void ExecStatus::remove_sym(const SymbolId sym_id) {
  LockGuard lock(mutex);

  for (auto &status : task_status) status.erase(sym_id);
}