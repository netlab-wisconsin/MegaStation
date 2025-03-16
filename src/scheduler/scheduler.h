#pragma once

#include <concurrentqueue.h>
#include <readerwriterqueue.h>

#include "bulk_enqueue.h"
#include "cpu_buffer.h"
#include "defs.h"
#include "exec_status.h"
#include "frame_manager.h"
#include "gpu_engine.h"
#include "task_manager.h"

namespace mega {

class Scheduler {
 public:
  moodycamel::ConcurrentQueue<TaskId> task_queue;
  moodycamel::ConcurrentQueue<SymbolId> complete_queue;

  const Policy policy;

  Scheduler(Policy policy_ = Policy::kByFrame)
      : task_queue(gconfig.symbols * 10),
        complete_queue(gconfig.symbols * 10),
        policy(policy_),
        frame_table(),
        task_manager(frame_table.size()),
        gpu_engine(policy_, task_manager, epoch, cpu_buffer,
                   frame_table.size()),
        exec_status(gpu_engine.notif_table, gpu_engine.inst_table,
                    task_manager) {}

 private:
  FrameTable frame_table;    //!< frame_table
  TaskManager task_manager;  //!< Task Table
  GpuEngine gpu_engine;      //!< GPU Engine
  ExecStatus exec_status;    //!< Execution Status

  void schedule();
  void synchronize();
  void timer();

 public:
  CPUMemBuffer cpu_buffer;
  std::atomic<uint64_t> epoch = 0;

  std::vector<std::thread> run();

  inline moodycamel::ProducerToken create_taskQ_token() {
    return moodycamel::ProducerToken(task_queue);
  }

  inline moodycamel::ConsumerToken create_completeQ_token() {
    return moodycamel::ConsumerToken(complete_queue);
  }

  inline bool enqueue_symbol(SymbolId frmsym,
                             moodycamel::ProducerToken &prod_token) {
    return enqueue_task(task_queue, frmsym, prod_token);
  }

  inline bool enqueue_symbol(SymbolId frmsym) {
    return enqueue_task(task_queue, frmsym);
  }
};

}  // namespace mega