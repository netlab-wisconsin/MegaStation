#pragma once

#include <concurrentqueue.h>
#include <nccl.h>
#include <parallel_hashmap/phmap.h>
#include <readerwriterqueue.h>

#include <barrier>
#include <future>
#include <vector>

#include "config/config.h"
#include "cpu_buffer.h"
#include "defs.h"
#include "mem_manager.h"
#include "notif_table.h"
#include "scheduler/inst_table.h"
#include "task_manager.h"

namespace mega {

struct NotifPair {
  TaskStatus *host_notif;  //!< Host Notification Array
  int32_t *device_notif;   //!< Device Notification Array

  inline NotifPair offset(const TaskId &task_id) const {
    const uint32_t offset_ = TaskTypeProp::offset(task_id);
    return {host_notif + offset_, device_notif + offset_};
  }

  inline NotifPair offset(const uint32_t &frame_id) const {
    const uint32_t offset_ = frame_id * TaskTypeProp::jobs_per_frame();
    return {host_notif + offset_, device_notif + offset_};
  }
};

// TODO: write destructor for MemManager and EventManager
using EventMap = std::vector<cudaEvent_t>;

class GpuEngine {
 private:
  constexpr static uint8_t kMaxConnect = MAX_CONNECT;

  std::vector<moodycamel::ReaderWriterQueue<ExecInfo>>
      exec_queue;  //!< dev_id -> Execution Queue
  std::vector<moodycamel::ReaderWriterQueue<FrameId>>
      clear_queue;  //!< dev_id -> Clear Queue

  std::vector<MemManager *> mem_table;  //!< dev_id -> Memory Manager
  std::vector<std::array<cudaStream_t, kMaxConnect + 1>>
      stream_table;                    //!< dev_id -> Stream Table
  std::vector<ncclComm_t> comm_table;  //!< dev_id -> NCCL Communicator
  std::vector<std::vector<int *>>
      device_csi_flags;  //!< dev_id -> Device CSI Flags

  const Policy policy;                 //!< refernece to Policy
  TaskManager &task_table;             //!< refernece to Task Table
  const std::atomic<uint64_t> &epoch;  //!< refernece to Epoch
  const uint32_t vframe_counts;        //!< refernece to Frame Table

  CPUMemBuffer &cpu_buffer;  //!< CPU Buffer

  void sync_engine(const DeviceId device_id,
                   std::promise<NotifPair> notif_promise,
                   moodycamel::ReaderWriterQueue<TaskId> &task_queue);
  void execute_engine(const DeviceId device_id, std::barrier<> &init_barrier);
  uint8_t get_strm_id(const TaskId &task_id);

 public:
  GpuEngine(const Policy &policy_, TaskManager &task_table_,
            const std::atomic<uint64_t> &epoch_, CPUMemBuffer &cpu_buffer_,
            const uint32_t vframe_count_)
      : exec_queue(gconfig.num_devices),
        clear_queue(gconfig.num_devices),
        mem_table(gconfig.num_devices),
        stream_table(gconfig.num_devices),
        comm_table(gconfig.num_devices),
        device_csi_flags(gconfig.num_devices,
                         std::vector<int *>(vframe_count_)),
        policy(policy_),
        task_table(task_table_),
        epoch(epoch_),
        vframe_counts(vframe_count_),
        cpu_buffer(cpu_buffer_),
        notif_table(vframe_count_) {
    int nDev = gconfig.num_devices;
    std::vector<int> devs(nDev);
    for (int i = 0; i < nDev; ++i) devs[i] = i;

    ncclCommInitAll(comm_table.data(), nDev, devs.data());
  }

  inline bool enqueue(const uint16_t &device_id, const ExecInfo &exec_info) {
    return exec_queue[device_id].enqueue(exec_info);
  }

  inline void enqueue(const FrameId &frame_id) {
    for (uint16_t device_id = 0; device_id < gconfig.num_devices; ++device_id)
      clear_queue[device_id].enqueue(frame_id);
  }

  std::vector<std::thread> run();

  NotifTable notif_table;  //!< Notification Table
  InstTable inst_table;    //!< refernece to Instruction Table
  moodycamel::ConcurrentQueue<TaskId> sync_queue;  //!< Sync Queue
};

}  // namespace mega