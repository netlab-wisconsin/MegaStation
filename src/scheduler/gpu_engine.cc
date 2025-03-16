#include "gpu_engine.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <parallel_hashmap/phmap.h>

#include <atomic>
#include <cstdint>

#include "beamform/beamform.h"
#include "comm/ring_gather.h"
#include "concurrentqueue.h"
#include "config/config.h"
#include "defs.h"
#include "fft/fft_op.h"
#include "gdr_utils.h"
#include "gdrapi.h"
#include "inst_table.h"
#include "ldpc/decoder.h"
#include "ldpc/encoder.h"
#include "mem_manager.h"
#include "modulation/modulation.h"
#include "notif_table.h"
#include "sgemv/equalize.h"
#include "sgemv/precode.h"
#include "utils/consts.h"
#include "utils/types.h"

using namespace mega;

std::vector<std::thread> GpuEngine::run() {
  MemManager::init(vframe_counts);

  std::vector<std::thread> threads;

  std::barrier init_barrier(gconfig.num_devices + 1);

  for (uint16_t dev_id = 0; dev_id < gconfig.num_devices; dev_id++) {
    threads.emplace_back(&GpuEngine::execute_engine, this, dev_id,
                         std::ref(init_barrier));
  }

  init_barrier.arrive_and_wait();

  return threads;
}

struct SymTaskHash {
  std::size_t operator()(const TaskId &task_id) const {
    return phmap::HashState().combine(0, task_id.sym_id, task_id.job_id);
  }
};

struct SymTaskEqual {
  bool operator()(const TaskId &lhs, const TaskId &rhs) const {
    return lhs.sym_id == rhs.sym_id && lhs.job_id == rhs.job_id;
  }
};

using RevseTaskArray = phmap::flat_hash_set<TaskId, SymTaskHash, SymTaskEqual>;
using RevseTaskTable = std::vector<RevseTaskArray>;

uint8_t GpuEngine::get_strm_id(const TaskId &task_id) {
  switch (policy) {
    case Policy::kByFrame:
      return task_id.frm_id % kMaxConnect;
    case Policy::kBySymbol:
    case Policy::kByTask:
    case Policy::kByMega:
      return (task_id.frm_id * gconfig.symbols + task_id.sym_id) % kMaxConnect;
    default:
      return kMaxConnect;
  }
}

void GpuEngine::execute_engine(const DeviceId device_id,
                               std::barrier<> &init_barrier) {
  cudaSetDevice(device_id);

  // A. Notif Array initialized in sync engine for locality's sake
  std::promise<NotifPair> notif_promise;
  std::future<NotifPair> notif_future = notif_promise.get_future();

  // Instruction Queue to sync
  moodycamel::ReaderWriterQueue<TaskId> inst_queue;

  std::thread sync_thread(&GpuEngine::sync_engine, this, device_id,
                          std::move(notif_promise), std::ref(inst_queue));

  // B. Initialize the stream table (maybe increase the number of streams)
  std::array<cudaStream_t, kMaxConnect + 1> &streams = stream_table[device_id];
  for (uint8_t i = 0; i < kMaxConnect; i++) {
    cudaStreamCreate(&streams[i]);
  }
  streams[kMaxConnect] = nullptr;

  auto &local_execQ = exec_queue[device_id];

  NotifPair local_notif = notif_future.get();

  // C. Initialize the memory manager (allocate all the memory)
  MemManager mem_manager(vframe_counts);
  mem_table[device_id] = &mem_manager;

  // D. Wait other threads to finish initialization
  init_barrier.arrive_and_wait();

  TaskStatus cpstatus;

  // TODO: make sure all related vector/queue are initialized

  // E. Start the loop of getting the instructions to push and execute
  while (gconfig.running) {
    // 1. Get the instruction to execute
    ExecInfo exec_info;
    if (local_execQ.try_dequeue(exec_info) == false) {
      __builtin_ia32_pause();
      continue;
    }

    uint64_t start_epoch = epoch.load(std::memory_order_relaxed);

    // 2. Get the instruction related information
    const TaskId &task_id = exec_info.task_id;
    NotifPair task_notif = local_notif.offset(task_id);
    const InstInfo inst_info = inst_table[task_id];

    // 3. Get the stream id to push the instruction
    uint8_t strm_id = get_strm_id(task_id);

    // 4. Get the input operands of the instruction
    std::array<void *, InstInfo::kMaxIn> in_ptr{};

    const auto prev_task = inst_table.prev_tasks(task_id);
    for (uint8_t i = 0; i < InstInfo::kMaxIn; i++) {
      if (prev_task[i].job_id == TaskType::kNullT) {
        continue;
      }

      // a) Get previous task's output memory information
      const MemOpType prev_mem = inst_table[prev_task[i]].out_mem;
      const MemOpId prev_mem_id = {{prev_task[i]}, prev_mem};

      // b) Get the memory pointer of the previous task's output
      in_ptr[i] = mem_manager.query_memop(prev_mem_id);
      if (in_ptr[i] == nullptr) {
        // i) Nullptr when memory not in current device
        DeviceId src_did = task_table.load(prev_task[i]);
        if (src_did == device_id || prev_mem == MemOpType::kNullMO) {
          // should not happen
          throw std::runtime_error(
              fmt::format("Input memory {} not found in ",
                          MemOpTypeToString(prev_mem)) +
              std::to_string(device_id) +
              fmt::format(": Prev Task {} {} {}", prev_task[i].frm_id,
                          prev_task[i].sym_id,
                          TaskTypeToString(prev_task[i].job_id)) +
              fmt::format(", Current Task {} {} {}\n", task_id.frm_id,
                          task_id.sym_id, TaskTypeToString(task_id.job_id)));
        }

        // ii) Get the memory pointer from the source device
        void *src_ptr = mem_table[src_did]->query_memop(prev_mem_id);
        while (src_ptr == nullptr) {
          // Wait until the memory on the source device is ready
          __builtin_ia32_pause();
          src_ptr = mem_table[src_did]->wait_memop(prev_mem_id);
        }
        // iii) Allocate memory copy of current device for the input operand
        void *dst_ptr = mem_manager.get_memop(prev_mem_id);
        if (dst_ptr == nullptr) {
          throw std::runtime_error("There is no enough [dst] memory for " +
                                   MemOpTypeToString(prev_mem));
        }

        // iv) Get the input operand size
        uint64_t copy_sz;
        auto &wrapper = MemManager::get_mem_wrapper(prev_mem);
        if (std::holds_alternative<Matrix>(wrapper))
          copy_sz = std::get<Matrix>(wrapper).szBytes();
        else
          copy_sz = std::get<CuphyTensor>(wrapper).szBytes();

        if (task_id.job_id == TaskType::kBeam && policy == Policy::kByMega &&
            gconfig.frame.pilot_syms > 1) {
          const Range &range =
              task_table.csi_table.get(task_id)[exec_info.rank_id];
          auto &matrix = std::get<Matrix>(wrapper);
          uint64_t batch_sz = matrix.szBytes(matrix.nDim() - 1);

          src_ptr = static_cast<char *>(src_ptr) + range.start * batch_sz;
          dst_ptr = static_cast<char *>(dst_ptr) + range.start * batch_sz;
          copy_sz = (range.end - range.start) * batch_sz;
        }

        // v) Ensure previous instruction already completed on the source device
        TaskStatus status = notif_table.load(prev_task[i]);
        while (status != TaskStatus::kDone) {
          __builtin_ia32_pause();
          status = notif_table.wait(prev_task[i]);
        }
        cudaMemcpyAsync(dst_ptr, src_ptr, copy_sz, cudaMemcpyDeviceToDevice,
                        streams[strm_id]);
        if (prev_task[i].sym_id != task_id.sym_id && policy != Policy::kByFrame)
          // vi) when the previous task is not in the same stream (kCSI)
          // to synchronize the completion of the copy operation
          cuStreamWriteValue32(
              streams[strm_id],
              reinterpret_cast<CUdeviceptr>(
                  local_notif.offset(prev_task[i]).device_notif),
              static_cast<int32_t>(TaskStatus::kDone),
              CU_STREAM_WRITE_VALUE_DEFAULT);
        in_ptr[i] = dst_ptr;
      } else if (prev_task[i].sym_id != task_id.sym_id &&
                 get_strm_id(prev_task[i]) != strm_id) {
        // vi) ...
        // to synchronize the completion of the execute-(same device) /
        // copy-(different device) operation
        cuStreamWaitValue32(streams[strm_id],
                            reinterpret_cast<CUdeviceptr>(
                                local_notif.offset(prev_task[i]).device_notif),
                            static_cast<int32_t>(TaskStatus::kDone),
                            CU_STREAM_WAIT_VALUE_EQ);
      }
    }

    // 5. Get the output operand of the instruction
    MemOpId out_mem = {{task_id}, inst_info.out_mem};
    void *out_ptr = mem_manager.query_memop(out_mem);
    if (out_ptr == nullptr) {
      // a) Not nullptr only when out_mem is kCSI
      out_ptr = mem_manager.get_memop(out_mem);
      if (inst_info.out_mem != MemOpType::kNullMO && out_ptr == nullptr) {
        // TODO: handle this case
        throw std::runtime_error("There is no enough [out] memory for " +
                                 MemOpTypeToString(inst_info.out_mem));
      }

      // b) First time initialize this memory, initialize reference count
      if (out_mem.type_id != MemOpType::kNullMO) mem_manager.store_ref(out_mem);
    }

    // 6. Get the operator of the instruction
    void *op = mem_manager.get_memop({{task_id}, inst_info.op});
    if (inst_info.op != MemOpType::kNullMO && op == nullptr) {
      // TODO: handle this case
      throw std::runtime_error("There is no enough memory for " +
                               MemOpTypeToString(inst_info.op));
    }

    // 7. Execute the instruction (push the instruction to the device)
    switch (task_id.job_id) {
      case TaskType::kPFFT:
        (*reinterpret_cast<PilotFFT<QuanT> *>(op))(
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kPreFFT))
                .wrap(in_ptr[0]),
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kCSI))
                .wrap(out_ptr),
            (gconfig.sc_group * gconfig.frame.gidx_lidx[task_id.sym_id]),
            streams[strm_id]);
        if (gconfig.frame.pilot_syms > 1 &&
            task_id.sym_id == gconfig.frame.pilots.back() &&
            policy != Policy::kByFrame) {
          // last pilot symbol, synchronize the completion of all the previous
          // PFFT operation
          for (auto &pilot_symid : gconfig.frame.pilots) {
            if (pilot_symid == task_id.sym_id) continue;

            TaskId prev_pilot = task_id;
            prev_pilot.sym_id = pilot_symid;
            cuStreamWaitValue32(
                streams[strm_id],
                reinterpret_cast<CUdeviceptr>(
                    local_notif.offset(prev_pilot).device_notif),
                static_cast<int32_t>(TaskStatus::kDone),
                CU_STREAM_WAIT_VALUE_EQ);
          }
        }
        break;
      case TaskType::kBeam: {
        auto matrix =
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kCSI));
        const RankRange &rank_range = task_table.csi_table.get(task_id);
        if (policy == Policy::kByMega && gconfig.frame.pilot_syms > 1) {
          const Range &range = rank_range[exec_info.rank_id];
          matrix = matrix.sub(range.start, range.end);
        }
        reinterpret_cast<Beamform *>(op)->beamform_uplink(
            matrix.wrap(in_ptr[0]), streams[strm_id]);
        if (policy == Policy::kByMega && gconfig.frame.pilot_syms > 1 &&
            rank_range.size() > 1) {
          uint32_t next_rank_id = (exec_info.rank_id + 1) % rank_range.size();
          DeviceId next_dev_id =
              task_table.csi_table.get_dev(task_id, next_rank_id);
          MemOpId next_mem_id = {{task_id}, MemOpType::kCSI};
          void *next_ptr = mem_table[next_dev_id]->query_memop(next_mem_id);
          while (next_ptr == nullptr) {
            // Wait until the memory on the source device is ready
            __builtin_ia32_pause();
            next_ptr = mem_table[next_dev_id]->wait_memop(next_mem_id);
          }
          assert(device_csi_flags[device_id][task_id.frm_id] ==
                 task_notif.device_notif);  // TODO: remove this
          cudaMemcpyAsync(&cpstatus, task_notif.device_notif,
                          sizeof(TaskStatus), cudaMemcpyDeviceToHost,
                          streams[strm_id]);  // no this, no run
          ringGather(reinterpret_cast<char *>(next_ptr),
                     reinterpret_cast<char *>(out_ptr),
                     device_csi_flags[next_dev_id][task_id.frm_id],
                     task_notif.device_notif, exec_info.rank_id, rank_range,
                     matrix.szBytes(matrix.nDim() - 1), streams[strm_id]);
        }
      } break;
      case TaskType::kBeamNM:
        reinterpret_cast<Beamform *>(op)->beamform_downlink(
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kCSI))
                .wrap(in_ptr[0]),
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kNmCSI))
                .wrap(out_ptr),
            streams[strm_id]);
        break;
      case TaskType::kUFFT:
        (*reinterpret_cast<UplinkFFT<QuanT> *>(op))(
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kPreFFT))
                .wrap(in_ptr[0]),
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kUFFT))
                .wrap(out_ptr),
            streams[strm_id]);
        break;
      case TaskType::kEqual:
        Equalize::equalize(
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kCSI))
                .wrap(in_ptr[0]),
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kUFFT))
                .wrap(in_ptr[1]),
            std::get<CuphyTensor>(
                MemManager::get_mem_wrapper(MemOpType::kEqual))
                .wrap(out_ptr),
            gconfig.sc_group, gconfig.mod_order,
            gconfig.ldpc_uconfig.encoded_bits,
            gconfig.ldpc_uconfig.punctured_bits, kCbPerSymbol,
            streams[strm_id]);
        break;
      case TaskType::kDecode:
        DecoderFactory::decode(
            std::get<CuphyTensor>(
                MemManager::get_mem_wrapper(MemOpType::kEqual))
                .wrap(in_ptr[0]),
            std::get<CuphyTensor>(
                MemManager::get_mem_wrapper(MemOpType::kDecode))
                .wrap(out_ptr),
            streams[strm_id]);
        break;
      case TaskType::kEncode:
        EncoderFactory::encode(
            std::get<CuphyTensor>(
                MemManager::get_mem_wrapper(MemOpType::kUncode))
                .wrap(in_ptr[0]),
            std::get<CuphyTensor>(
                MemManager::get_mem_wrapper(MemOpType::kEncode))
                .wrap(out_ptr),
            streams[strm_id]);
        break;
      case TaskType::kModulate:
        Modulation::modulate(
            std::get<CuphyTensor>(
                MemManager::get_mem_wrapper(MemOpType::kEncode))
                .wrap(in_ptr[0]),
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kModulate))
                .wrap(out_ptr),
            streams[strm_id]);
        break;
      case TaskType::kPrecode:
        Precode::precode(
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kNmCSI))
                .wrap(in_ptr[0]),
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kModulate))
                .wrap(in_ptr[0]),
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kPrecode))
                .wrap(out_ptr),
            gconfig.sc_group, gconfig.ofdm_start, streams[strm_id]);
        break;
      case TaskType::kDiFFT:
        (*reinterpret_cast<DownlinkIFFT<QuanT> *>(op))(
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kPrecode))
                .wrap(in_ptr[0]),
            std::get<Matrix>(MemManager::get_mem_wrapper(MemOpType::kDiFFT))
                .wrap(out_ptr),
            streams[strm_id]);
        break;
      case TaskType::kLoad: {
        SymbolId circ_id = {exec_info.circ_id, task_id.sym_id};
        Matrix &cpu_in = cpu_buffer.buf_in[circ_id];
        cudaMemcpyAsync(cpu_in.ptr(), out_ptr, cpu_in.szBytes(),
                        cudaMemcpyHostToDevice, streams[strm_id]);
        exec_info.rank_id = 0;
      } break;
      case TaskType::kStore: {
        SymbolId circ_id = {exec_info.circ_id, task_id.sym_id};
        Matrix &cpu_out = cpu_buffer.buf_out[circ_id];
        cudaMemcpyAsync(in_ptr[0], cpu_out.ptr(), cpu_out.szBytes(),
                        cudaMemcpyHostToDevice, streams[strm_id]);
        exec_info.rank_id = 0;
      } break;
      default:
        throw std::runtime_error("Unknown Task Type %d" +
                                 TaskTypeToString(task_id.job_id));
    }
    cuStreamWriteValue32(streams[strm_id],
                         reinterpret_cast<CUdeviceptr>(task_notif.device_notif),
                         static_cast<int32_t>(TaskStatus::kDone),
                         CU_STREAM_WRITE_VALUE_DEFAULT);

    // 8. Set the task status to kSched to inform the scheduler that instruction
    // has pushed to device
    notif_table.store(task_id, TaskStatus::kSched);
    task_table.store_time(
        task_id, epoch.load(std::memory_order_relaxed));  // store kSched time

    // 9. Enqueue the task to the sync engine
    if (exec_info.rank_id == 0)  // only enqueue the first rank
      inst_queue.enqueue(task_id);
    else if (task_id.job_id == TaskType::kBeam)  // TODO: improve this
      mem_manager.free_memop({{task_id}, inst_info.op});

    uint64_t end_epoch = epoch.load(std::memory_order_relaxed);
    inst_table.update_latency({0, 0, TaskType::kNullT}, device_id,
                              end_epoch - start_epoch);
  }

  sync_thread.join();

  for (uint8_t i = 0; i < kMaxConnect; i++) {
    cudaStreamDestroy(streams[i]);
  }
}

void GpuEngine::sync_engine(const DeviceId device_id,
                            std::promise<NotifPair> notif_promise,
                            moodycamel::ReaderWriterQueue<TaskId> &inst_queue) {
  cudaSetDevice(device_id);

  // A. Allocate GDR memory for notif array
  GDRInfo local_gdr;
  TaskStatus *host_notif;

  local_gdr.alloc_map(reinterpret_cast<void *&>(host_notif));
  NotifPair local_notif = {host_notif, local_gdr.ptr()};
  std::vector<int *> &csi_flags = device_csi_flags[device_id];
  for (uint16_t i = 0; i < vframe_counts; i++) {
    uint16_t csi_symid = gconfig.frame.pilots.back();
    TaskId csi_task = {i, csi_symid, TaskType::kBeam};
    csi_flags[i] = local_notif.offset(csi_task).device_notif;
  }
  notif_promise.set_value(local_notif);

  // B. Initialize reverse task table (vframe -> working task_id)
  RevseTaskTable reverse_task_table(
      vframe_counts, RevseTaskArray(TaskTypeProp::jobs_per_frame()));

  moodycamel::ProducerToken ptok(sync_queue);  // producer token for syncQ

  auto &clearQ = clear_queue[device_id];

  // C. Start the loop of getting the working instructions and synchronizing
  while (gconfig.running) {
    __builtin_ia32_pause();

    // 1. Get the frame to free from clear_queue (if any)
    FrameId frame_id;
    while (clearQ.try_dequeue(frame_id)) {
      if (!reverse_task_table[frame_id.frm_id].empty()) {
        // should not happen
        throw std::runtime_error("There are still working instructions in " +
                                 std::to_string(frame_id.frm_id));
      }
      NotifPair notifs = local_notif.offset(frame_id.frm_id);
      memset(notifs.host_notif, 0,
             TaskTypeProp::jobs_per_frame() * sizeof(int32_t));
      gdr_copy_to_mapping(
          local_gdr.gdr_handle, local_gdr.offset(notifs.device_notif),
          notifs.host_notif, TaskTypeProp::jobs_per_frame() * sizeof(int32_t));
    }

    // 2. Get current working instructions from execute engine (if any)
    TaskId task_id;
    while (inst_queue.try_dequeue(task_id)) {
      reverse_task_table[task_id.frm_id].emplace(task_id);
    }

    // 3. Check if the task is done
    for (uint32_t vfrm_id = 0; vfrm_id < reverse_task_table.size(); vfrm_id++) {
      // a) If there is a working instruction in the vframe
      auto &task_array = reverse_task_table[vfrm_id];
      if (task_array.empty()) {
        continue;
      }

      // b) Synchronize the notif array of the vframe from GPU to CPU (gdrcopy)
      NotifPair frame_notifs = local_notif.offset(vfrm_id);

      gdr_copy_from_mapping(local_gdr.gdr_handle, frame_notifs.host_notif,
                            local_gdr.offset(frame_notifs.device_notif),
                            TaskTypeProp::jobs_per_frame() * sizeof(int32_t));

      // c) Check if each working instruction in this vframe is done
      for (const TaskId &task_id : task_array) {
        NotifPair notifs = local_notif.offset(task_id);
        if (*notifs.host_notif == TaskStatus::kDone) {
          // i) Set kDone to the global notif table to inform completion
          notif_table.store(task_id, TaskStatus::kDone);

          const InstInfo inst_info = inst_table[task_id];

          // ii) Free the operator of the instruction
          if (inst_info.op != MemOpType::kNullMO)
            mem_table[device_id]->free_memop({{task_id}, inst_info.op});

          // iii) decrement ref counts of input operands of the instruction
          const auto prev_task = inst_table.prev_tasks(task_id);
          uint64_t max_comt = 0;
          for (uint8_t i = 0; i < InstInfo::kMaxIn; i++) {
            if (prev_task[i].job_id == TaskType::kNullT) continue;

            MemOpId in_mem = {{prev_task[i]}, inst_table[prev_task[i]].out_mem};

            if (in_mem.type_id != MemOpType::kNullMO &&
                MemManager::dec_ref(in_mem) == 0) {
              // clear all the other copies in the other devices
              for (uint16_t dev_id = 0; dev_id < gconfig.num_devices; dev_id++)
                mem_table[dev_id]->free_memop(in_mem);
            }

            max_comt = std::max(task_table.load_time(prev_task[i]), max_comt);
          }

          // iv) Tell the scheduler that the instruction is done
          sync_queue.enqueue(ptok, task_id);

          // v) Update the latency of the instruction
          uint64_t cur_epoch = epoch.load(std::memory_order_relaxed);
          max_comt = std::max(task_table.load_time(task_id), max_comt);
          inst_table.update_latency(task_id, device_id, cur_epoch - max_comt);
          task_table.store_time(task_id, cur_epoch);  // Store kDone time
          if (TaskTypeProp::last_job(task_id) &&
              gconfig.frame.gidx_sym[task_id.sym_id] > Config::Pilot) {
            TaskId first_task = {task_id.frm_id, task_id.sym_id,
                                 TaskType::kLoad};
            uint64_t prev_comt = task_table.load_time(first_task) -
                                 inst_table.get_latency(first_task, device_id);
            // symbol time
            inst_table.update_latency(1, device_id, cur_epoch - prev_comt);
            if (task_id.sym_id == gconfig.symbols - 1) {
              first_task.sym_id = 0;
              prev_comt = task_table.load_time(first_task) -
                          inst_table.get_latency(first_task, device_id);
              // frame time
              inst_table.update_latency(2, device_id, cur_epoch - prev_comt);
            }
          }

          // vi) Remove the instruction from the working list
          task_array.erase(task_id);
        }
      }
    }
  }

  local_gdr.unmap_free(reinterpret_cast<void *&>(host_notif));
}