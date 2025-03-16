#include "scheduler.h"

#include <concurrentqueue.h>
#include <parallel_hashmap/btree.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <deque>

#include "comm/range.h"
#include "config/config.h"
#include "defs.h"
#include "frame_manager.h"
#include "utils/consts.h"
#include "utils/timer.h"

using namespace mega;

template <typename T, int N>
struct VarArray {
  std::array<T, N> data;
  uint32_t size = 0;

  VarArray() = default;

  void push_back(T val) { data[size++] = val; }
  T pop_back() { return data[--size]; }
  bool empty() const { return size == 0; }
};

void Scheduler::schedule() {
  std::thread sync_thread(&Scheduler::synchronize, this);

  DeviceId device_id = gconfig.num_devices - 1;
  // TODO: span multiple GPUs larger than 2
  constexpr uint16_t kMaxSpan = 1;
  // For tasks span multiple GPUs
  VarArray<DeviceId, kMaxSpan - 1> other_devices;

  moodycamel::ConsumerToken ctok(task_queue);
  moodycamel::ProducerToken ptok(complete_queue);

  phmap::btree_map<FrameId, std::deque<TaskId>> reschedule_queue;

  phmap::flat_hash_map<DeviceId, uint8_t> device_count(kMaxSpan);

  double total_time = 0.0;
  int total_tasks = 0;
  TimerCPU timer;

  int add = -1;

  while (gconfig.running) {
    TaskId new_task{};

    for (auto it = reschedule_queue.begin(); it != reschedule_queue.end();) {
      timer.start();
      if (it->second.empty()) {
        ++it;
        continue;
      }

      TaskId ntask = it->second.front();
      it->second.pop_front();

      if (TaskTypeProp::last_job(ntask)) {
        complete_queue.enqueue(ptok, ntask);
        if (ntask.sym_id == gconfig.symbols - 1) {
          it = reschedule_queue.erase(it);
          goto timer_stop;
        }
      }
      ++it;
    timer_stop:
      timer.stop();
      total_time += timer.get_duration_ms();
      total_tasks++;
    }

    if (task_queue.try_dequeue(ctok, new_task) == false) {
      __builtin_ia32_pause();
      continue;
    }

    // if (new_task.frm_id % 900 == 0 && new_task.frm_id != 0 &&
    //     new_task.sym_id == 0 && new_task.job_id == TaskType::kLoad) {
    //   if (gconfig.num_devices == 2) add = 1;
    //   gconfig.num_devices += add;
    //   spdlog::info("Task: {} {} {}, NUM Device {}", new_task.frm_id,
    //                new_task.sym_id, TaskTypeToString(new_task.job_id),
    //                gconfig.num_devices);
    // }

    timer.start();
    // constexpr int vsm = 7;

    // Get Notification
    uint32_t phy_frm_id = new_task.frm_id;
    if (new_task.sym_id == 0 && new_task.job_id == TaskType::kLoad)
      new_task.frm_id = frame_table.register_frame(new_task).frm_id;
    else
      new_task.frm_id = frame_table.get_v(new_task).frm_id;

    if (new_task == FrameTable::NFrameId) {
      // 1. no vframe available
      // 2. frame device = -1
      new_task.frm_id = phy_frm_id;
      reschedule_queue[new_task].push_back(new_task);
      continue;
    }
    if (new_task.sym_id == 0 && new_task.job_id == TaskType::kLoad)
      frame_table.init_count(new_task);

    if (gconfig.frame.pilot_syms > 1 &&
        gconfig.frame.gidx_sym[new_task.sym_id] <= Config::Symbol::Pilot &&
        new_task.sym_id > gconfig.frame.pilots.front()) {
      // Not first pilot symbol, push to the first pilot symbol's GPU
      TaskId pilot_task = new_task;
      pilot_task.sym_id = gconfig.frame.pilots.front();
      device_id = task_manager.load(pilot_task);

      if (policy == Policy::kByMega)
        exec_status.add_task(device_id, new_task);  //,
                                                    // exec_status.guess_end(
      //     new_task, epoch.load(std::memory_order_relaxed), device_id));

      goto skip_select;
    }

    // Select Device
    switch (policy) {
      case Policy::kByFrame:
        device_id = new_task.frm_id % gconfig.num_devices;
        // if (phy_frm_id < vsm)
        //   device_id = 0;
        // else
        //   device_id = 1;
        // if (phy_frm_id > 100) device_id = new_task.frm_id %
        // gconfig.num_devices;
        break;
      case Policy::kBySymbol:
        device_id = (new_task.frm_id * gconfig.symbols + new_task.sym_id) %
                    gconfig.num_devices;
        // if (new_task.sym_id < vsm)
        //   device_id = 0;
        // else
        //   device_id = 1;
        break;
      case Policy::kByTask:
        // if (new_task.sym_id < vsm)
        //   device_id = 0;
        // else
        //   device_id = 1;
        device_id = (new_task.frm_id * gconfig.symbols + new_task.sym_id) %
                    gconfig.num_devices;
        if (gconfig.frame.gidx_sym[new_task.sym_id] == Config::Symbol::Uplink &&
            (new_task.job_id > TaskType::kEqual ||
             new_task.job_id == TaskType::kStore))
          device_id = (device_id + 1) % gconfig.num_devices;
        else if (gconfig.frame.gidx_sym[new_task.sym_id] ==
                     Config::Symbol::Downlink &&
                 new_task.job_id < TaskType::kPrecode &&
                 new_task.job_id != TaskType::kStore)
          device_id = (device_id + 1) % gconfig.num_devices;
        break;
      case Policy::kByMega: {
        // auto good_id = exec_status.get_device(
        //     new_task, epoch.load(std::memory_order_relaxed));
        // device_id = good_id.first;
        device_id = exec_status.get_device(new_task);
        // spdlog::info("Task: {} {} {} is scheduled to Device {}",
        //              new_task.frm_id, new_task.sym_id,
        //              TaskTypeToString(new_task.job_id), device_id);

        if (device_id < 0) [[unlikely]] {
          // assert(new_task.job_id == TaskType::kLoad && new_task.sym_id == 0);
          frame_table.unregister_frame(new_task, {phy_frm_id});
          new_task.frm_id = phy_frm_id;
          reschedule_queue[new_task].push_front(new_task);
          break;
        } else {
          // exec_status.add_task(device_id, new_task, good_id.second);
          exec_status.add_task(device_id, new_task);
        }
      } break;
      default:
        device_id = 0;
    }

  skip_select:
    if (device_id < 0) continue;

    if (policy == Policy::kByMega && gconfig.frame.pilot_syms > 1 &&
        new_task.job_id == TaskType::kBeam) {
      device_count.clear();
      device_count[device_id] = 2;

      for (int i = 0; i < kMaxSpan - 1; i++) {
        // auto good_id = exec_status.get_device(
        //     new_task, epoch.load(std::memory_order_relaxed));
        // DeviceId min_dev = good_id.first;
        DeviceId min_dev = exec_status.get_minDevice(new_task);

        if (min_dev >= 0) {
          // exec_status.add_task(min_dev, new_task, good_id.second);
          exec_status.add_task(min_dev, new_task);
          device_count[min_dev]++;
        } else {
          device_count[device_id]++;
        }
      }

      RankRange &ranges = task_manager.csi_table.ref_range(new_task);
      ranges.resize(device_count.size());
      RankDev &devices = task_manager.csi_table.ref_dev(new_task);
      devices.resize(device_count.size());

      const uint32_t scg_count = gconfig.ofdm_data / gconfig.sc_group;
      uint16_t end = scg_count * device_count[device_id] / (kMaxSpan + 1);

      ranges[0] = {0, end};
      devices[0] = device_id;

      for (auto &[dev_id, count] : device_count) {
        exec_status.apply_factor(dev_id, new_task,
                                 count / static_cast<float>(kMaxSpan + 1));
        if (dev_id == device_id) continue;

        other_devices.push_back(dev_id);

        uint16_t next_end = end + scg_count * count / (kMaxSpan + 1);
        ranges[other_devices.size] = {end, next_end};
        devices[other_devices.size] = dev_id;

        end = next_end;
      }
      ranges[other_devices.size].end = scg_count;
    }
    if (policy == Policy::kByMega && gconfig.frame.pilot_syms > 1 &&
        new_task.job_id == TaskType::kBeamNM) {
      RankDev &devices = task_manager.csi_table.ref_dev(new_task);
      for (auto &dev_id : devices) {
        if (dev_id == device_id) continue;
        // exec_status.add_task(
        //     dev_id, new_task,
        //     exec_status.guess_end(new_task, epoch.load(), dev_id));
        exec_status.add_task(dev_id, new_task);
        other_devices.push_back(dev_id);
      }
    }

    task_manager.store(new_task, device_id);

    ExecInfo exec_info = {.task_id = new_task, .rank_id = 0};
    if (new_task.job_id == TaskType::kLoad ||
        new_task.job_id == TaskType::kStore)
      exec_info.circ_id = phy_frm_id % kFrameWindow;

    gpu_engine.enqueue(device_id, exec_info);

    while (!other_devices.empty()) {
      exec_info.rank_id = other_devices.size;
      gpu_engine.enqueue(other_devices.pop_back(), exec_info);
    }

    timer.stop();
    total_time += timer.get_duration_ms();
    total_tasks++;
  }

  sync_thread.join();

  spdlog::info("Average Task Scheduling Time: {} ms ({})",
               total_time / total_tasks, total_tasks);
}

std::vector<std::thread> Scheduler::run() {
  std::vector<std::thread> sched_threads = gpu_engine.run();
  sched_threads.emplace_back(std::thread(&Scheduler::timer, this));
  sched_threads.emplace_back(std::thread(&Scheduler::schedule, this));

  return sched_threads;
}

void Scheduler::synchronize() {
  auto &notif_table = gpu_engine.notif_table;

  auto &syncQ = gpu_engine.sync_queue;
  moodycamel::ConsumerToken ctok(syncQ);
  moodycamel::ProducerToken ptok(complete_queue);

  while (gconfig.running) {
    TaskId task_id;
    if (syncQ.try_dequeue(ctok, task_id) == false) {
      __builtin_ia32_pause();
      continue;
    }

    if (TaskTypeProp::last_job(task_id)) {
      TaskId ctask_id = task_id;
      ctask_id.frm_id = frame_table.get_p(task_id).frm_id;
      complete_queue.enqueue(ptok, ctask_id);
      exec_status.remove_sym(task_id);
    }

    if (frame_table.dec_count(task_id, notif_table))
      gpu_engine.enqueue(task_id);
  }
}

class TimerUS : public Timer {
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock>
      start_time;  //!< start time
  std::chrono::time_point<std::chrono::high_resolution_clock>
      end_time;  //!< end time

 public:
  TimerUS() {}
  void start(void * = nullptr) override {
    start_time = std::chrono::high_resolution_clock::now();
  }

  void stop(void * = nullptr) override {
    end_time = std::chrono::high_resolution_clock::now();
  }

  double get_duration_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                 start_time)
        .count();
  }

  double get_duration_ms() override { return get_duration_us() / 1000.0; }

  ~TimerUS() override = default;
};

void Scheduler::timer() {
  TimerUS timer;
  while (gconfig.running) {
    timer.start();
    // epoch++ every 10 us
    timer.stop();
    while (timer.get_duration_us() < 10) {
      timer.stop();
    }
    epoch.fetch_add(1, std::memory_order_relaxed);
  }
}