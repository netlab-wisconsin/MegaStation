#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <fstream>
#include <thread>
#include <vector>

#include "config/config.h"
#include "scheduler/bulk_enqueue.h"
#include "scheduler/defs.h"
#include "scheduler/scheduler.h"
#include "utils/consts.h"
#include "utils/pin_threads.h"
#include "utils/prints.h"
#include "utils/timer.h"

int main(int argc, char **argv) {
  int ants = 64;
  int users = 16;
  int ofdm = 1200;
  int sg = 16;
  char sym = 'U';
  mega::Policy policy = mega::Policy::kByFrame;
  std::string dir = "../../benchmark/";

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-ants") == 0) {
      ants = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-users") == 0) {
      users = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-ofdm") == 0) {
      ofdm = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-sg") == 0) {
      sg = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-dir") == 0) {
      dir = std::string(argv[++i]);
    } else if (strcmp(argv[i], "--up") == 0) {
      sym = 'U';
    } else if (strcmp(argv[i], "--dw") == 0) {
      sym = 'D';
    } else if (strcmp(argv[i], "-policy") == 0) {
      if (strcmp(argv[i + 1], "byframe") == 0) {
        policy = mega::Policy::kByFrame;
      } else if (strcmp(argv[i + 1], "bymega") == 0) {
        policy = mega::Policy::kByMega;
      } else if (strcmp(argv[i + 1], "bysymbol") == 0) {
        policy = mega::Policy::kBySymbol;
      } else if (strcmp(argv[i + 1], "bytask") == 0) {
        policy = mega::Policy::kByTask;
      } else {
        throw std::runtime_error("Invalid policy");
      }
      ++i;
    }
  }

  spdlog::init_thread_pool(mega::kLogQueueSize, mega::kLogThreads);
  spdlog::set_default_logger(
      spdlog::create_async_nb<spdlog::sinks::stdout_color_sink_mt>("console"));
  // +%i [level(color)] message
  spdlog::set_pattern("[+%i][%^%L%$] %v");

  std::string log_file = std::string(dir) + "throughput" +
                         std::to_string(ants) + "x" + std::to_string(users) +
                         sym + mega::PolicyToString(policy) + ".tsv";
  auto my_logger = spdlog::basic_logger_mt("basic_logger", log_file, true);
  my_logger->set_pattern("%v");
  my_logger->flush_on(spdlog::level::info);

  mega::PinThreads::init();

  int num_pilots = (users + sg - 1) / sg;
  std::string frame_str =
      std::string(num_pilots, 'P') + std::string(14 - num_pilots, sym);

  mega::gconfig = mega::Config(ants, users, ofdm, 2048, sg, frame_str, 6,
                               1. / 3., 1, 1, 16);
  mega::gconfig.open_peer_access();

  mega::Scheduler scheduler(policy);
  std::vector<std::thread> threads = scheduler.run();

  printf("Running\n");
  for (int i = 0; i < 2; i++) {
    for (uint16_t symbol_id = 0; symbol_id < mega::gconfig.symbols;
         symbol_id++) {
      mega::SymbolId frmsym = {static_cast<uint32_t>(~i), symbol_id};
      bool status = mega::enqueue_task(scheduler.task_queue, frmsym);
      if (status == false) {
        throw std::runtime_error("Failed to enqueue");
      }
    }
    for (uint16_t symbol_id = 0; symbol_id < mega::gconfig.symbols;
         symbol_id++) {
      mega::SymbolId frmsym;
      while (scheduler.complete_queue.try_dequeue(frmsym) == false) {
      }
    }
  }
  printf("After Cold Start\n");
  for (int i = 2; i < 4; i++) {
    for (uint16_t symbol_id = 0; symbol_id < mega::gconfig.symbols;
         symbol_id++) {
      mega::SymbolId frmsym = {static_cast<uint32_t>(~i), symbol_id};
      bool status = mega::enqueue_task(scheduler.task_queue, frmsym);
      if (status == false) {
        throw std::runtime_error("Failed to enqueue");
      }
    }
  }
  for (int i = 2; i < 4; i++) {
    for (uint16_t symbol_id = 0; symbol_id < mega::gconfig.symbols;
         symbol_id++) {
      mega::SymbolId frmsym;
      while (scheduler.complete_queue.try_dequeue(frmsym) == false) {
      }
    }
  }
  printf("After Warm Start\n");

  std::string upfile_name = std::string(dir) + "tx_" + std::to_string(ants) +
                            "x" + std::to_string(users) + ".data";
  std::ifstream input_ul(upfile_name, std::ios::binary);
  input_ul.seekg(0, std::ios::end);
  std::streampos fileSize_ul = input_ul.tellg();
  input_ul.seekg(0, std::ios::beg);

  std::string dlfile_name = std::string(dir) + "mac_" + std::to_string(ants) +
                            "x" + std::to_string(users) + ".data";
  std::ifstream input_dl(dlfile_name, std::ios::binary);
  input_dl.seekg(0, std::ios::end);
  std::streampos fileSize_dl = input_dl.tellg();
  input_dl.seekg(0, std::ios::beg);

  mega::SymbolId frmsym_in = {0, 0};
  auto h_recv = scheduler.cpu_buffer.buf_in[frmsym_in];
  if (fileSize_ul != h_recv.szBytes(h_recv.nDim()) * 14) {
    spdlog::error("UlFile size mismatch: {} != {}", fileSize_ul,
                  h_recv.szBytes(h_recv.nDim()) * 14);
    return -1;
  }

  size_t pilot_size = 0;
  for (int i = 0; i < num_pilots; i++) {
    size_t read_size = h_recv.szBytes(h_recv.nDim());
    pilot_size += read_size;

    input_ul.read(h_recv.ptr<char>(), read_size);
    frmsym_in.sym_id++;
    h_recv = scheduler.cpu_buffer.buf_in[frmsym_in];
  }

  if (sym == 'D' && fileSize_dl != h_recv.szBytes(h_recv.nDim()) *
                                       (mega::gconfig.frame.downlink_syms)) {
    spdlog::error(
        "DlFile size mismatch: {} != {}", fileSize_dl,
        h_recv.szBytes(h_recv.nDim()) * (mega::gconfig.frame.downlink_syms));
    return -1;
  }

  auto &normal_in = sym == 'D' ? input_dl : input_ul;
  size_t normal_size = 0;
  size_t read_size = h_recv.szBytes(h_recv.nDim());
  while (normal_in.read(h_recv.ptr<char>(), read_size)) {
    normal_size += read_size;

    frmsym_in.sym_id++;
    h_recv = scheduler.cpu_buffer.buf_in[frmsym_in];
    h_recv.szBytes(h_recv.nDim());
  }

  input_ul.close();
  input_dl.close();

  for (uint32_t i = 1; i < mega::kFrameWindow; i++) {
    memcpy(scheduler.cpu_buffer.buf_in[{i, 0}].ptr<char>(),
           scheduler.cpu_buffer.buf_in[{0, 0}].ptr<char>(), pilot_size);
    memcpy(scheduler.cpu_buffer.buf_in[{i, 1}].ptr<char>(),
           scheduler.cpu_buffer.buf_in[{0, 1}].ptr<char>(), normal_size);
  }

  std::mutex mtx;
  phmap::flat_hash_map<uint64_t, mega::TimerCPU> timer_cpu_frm;
  phmap::flat_hash_map<uint64_t, uint32_t> frm_counter;

  constexpr uint32_t kNumFrames = 1000;
  constexpr uint32_t kNumRrus = 1;

  bool start = false;
  std::vector<std::thread> producers;
  for (uint32_t rru_id = 0; rru_id < kNumRrus; rru_id++) {
    producers.push_back(std::thread([rru_id, &kNumRrus, &scheduler,
                                     &timer_cpu_frm, &start, &mtx]() {
      moodycamel::ProducerToken ptok(scheduler.task_queue);
      mega::TimerCPU timer_sleep;

      while (start == false) {
      }
      for (uint32_t rframe_id = 0; rframe_id < kNumFrames; rframe_id++) {
        uint32_t frame_id = rframe_id * kNumRrus + rru_id;

        // if (scheduler.task_queue.size_approx() > 5 * mega::gconfig.symbols) {
        //   printf(RED "Waiting: %u, %lu\n" RST, frame_id,
        //          scheduler.task_queue.size_approx());
        //   while (timer_cpu_frm.size() > 1) {
        //   }
        // }

        mtx.lock();
        timer_cpu_frm[frame_id].start();
        mtx.unlock();

        for (uint16_t symbol_id = 0; symbol_id < mega::gconfig.symbols;
             symbol_id++) {
          timer_sleep.start();
          // std::printf(GRN "Enqueueing: %d, %d\n" RST, frame_id, symbol_id);

          mega::SymbolId frmsym = {frame_id, symbol_id};
          bool status = enqueue_task(scheduler.task_queue, frmsym, ptok);

          if (status == false) throw std::runtime_error("Failed to enqueue");

          uint64_t sleep_time = uint64_t(1000.0 / mega::gconfig.symbols);
          timer_sleep.stop();
          while (uint64_t(timer_sleep.get_duration_ms() * 1000.0) < sleep_time)
            timer_sleep.stop();
        }
      }
    }));
  }

  mega::TimerCPU timer;
  start = true;

  timer.start();

  for (uint32_t frame_id = 0; frame_id < kNumFrames * kNumRrus; frame_id++) {
    for (uint16_t symbol_id = 0; symbol_id < mega::gconfig.symbols;
         symbol_id++) {
      mega::SymbolId frmsym;

      while (scheduler.complete_queue.try_dequeue(frmsym) == false) {
      }

      // std::printf(BLU "Completed: %d, %d: %f\n" RST, frmsym.frm_id,
      //             frmsym.sym_id, timer.checkpoint());

      frm_counter[frmsym.frm_id]++;
      if (frm_counter[frmsym.frm_id] == mega::gconfig.symbols) {
        mtx.lock();
        timer_cpu_frm[frmsym.frm_id].stop();

        my_logger->info("{}\t{}", frmsym.frm_id,
                        timer_cpu_frm[frmsym.frm_id].get_duration_ms());
        timer_cpu_frm.erase(frmsym.frm_id);
        mtx.unlock();

        frm_counter.erase(frmsym.frm_id);
      }
    }
  }

  timer.stop();
  spdlog::info("Total Throughput: {}",
               kNumFrames * kNumRrus / timer.get_duration_ms());

  mega::gconfig.running = false;

  for (auto &t : producers) t.join();
  for (auto &t : threads) t.join();

  spdlog::shutdown();

  return 0;
}