/**
 * @file pin_threads.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Implementation of pin threads to cores
 * @version 0.1
 * @date 2024-01-29
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "pin_threads.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>

#include "numa.h"

using namespace mega;

std::vector<std::vector<uint32_t>> PinThreads::cpu_layout;
std::mutex PinThreads::mtx;
std::unordered_set<uint32_t> PinThreads::assigned_cores;

void PinThreads::init(const std::unordered_set<uint32_t>& exclude_cores) {
  int numa_cpus = numa_num_configured_cpus();
  bitmask* cpus = numa_bitmask_alloc(numa_cpus);
  int numa_nodes = numa_num_configured_nodes();
  cpu_layout.resize(numa_nodes);
  for (int i = 0; i < numa_nodes; i++) {
    std::string node_str = "Numa Node " + std::to_string(i) + " CPUs:";
    numa_node_to_cpus(i, cpus);
    for (uint32_t j = 0; j < cpus->size; j++) {
      if (numa_bitmask_isbitset(cpus, j)) {
        if (exclude_cores.count(j) == 0) {
          cpu_layout[i].push_back(j);
          node_str += " " + std::to_string(j);
        }
      }
    }
    spdlog::info(node_str);
  }
  numa_bitmask_free(cpus);
}

void PinThreads::pin_thread(uint32_t node_id, uint32_t thread_id) {
  std::unique_lock lock(mtx);

  uint32_t num_cores = cpu_layout.at(node_id).size();
  uint32_t core_id = cpu_layout.at(node_id).at(thread_id % num_cores);
  if (assigned_cores.count(thread_id) > 0) {
    spdlog::error("Thread {} is already pinned to numa {} core {}", thread_id,
                  node_id, core_id);
    throw std::runtime_error("Thread is already pinned to core");
  }
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  pthread_t current_thread = pthread_self();
  int ret = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
  if (ret != 0) {
    spdlog::error("pthread_setaffinity_np error: {}", strerror(errno));
    throw std::runtime_error("pthread_setaffinity_np error");
  }
  assigned_cores.insert(thread_id);
  spdlog::info("Thread {} is pinned to numa {} core {}", thread_id, node_id,
               core_id);
}

void PinThreads::release_thread(uint32_t node_id, uint32_t thread_id) {
  std::unique_lock lock(mtx);

  uint32_t num_cores = cpu_layout.at(node_id).size();
  uint32_t core_id = cpu_layout.at(node_id).at(thread_id % num_cores);
  if (assigned_cores.count(thread_id) == 0) {
    spdlog::error("Thread {} is not pinned to numa {} core {}", thread_id,
                  node_id, core_id);
    throw std::runtime_error("Thread is not pinned to core");
  }

  assigned_cores.erase(thread_id);
  spdlog::info("Thread {} is released from numa {} core {}", thread_id, node_id,
               core_id);
}