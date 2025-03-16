/**
 * @file bulk_enqueue.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Bulk Enqueue A symbol (multiple tasks) to a FIFO Queue
 * @version 0.1
 * @date 2024-04-14
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "config/config.h"
#include "defs.h"

namespace mega {

inline bool enqueue_task(auto &fifo, SymbolId frmsym, auto &prod_token) {
  bool status = true;
  static constexpr uint8_t kMaxNumTasks = 6;
  std::array<TaskId, kMaxNumTasks> tasks;
  switch (gconfig.frame.gidx_sym[frmsym.sym_id]) {
    case Config::NLPilot:
      tasks = {{{frmsym.frm_id, frmsym.sym_id, TaskType::kLoad},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kPFFT}}};
      status = fifo.enqueue_bulk(prod_token, tasks.begin(), 2);
      break;
    case Config::Pilot:
      tasks = {{{frmsym.frm_id, frmsym.sym_id, TaskType::kLoad},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kPFFT},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kBeam},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kBeamNM}}};
      status = fifo.enqueue_bulk(prod_token, tasks.begin(),
                                 gconfig.frame.downlink_syms > 0 ? 4 : 3);
      break;
    case Config::Uplink:
      tasks = {{{frmsym.frm_id, frmsym.sym_id, TaskType::kLoad},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kUFFT},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kEqual},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kDecode},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kStore}}};
      status = fifo.enqueue_bulk(prod_token, tasks.begin(), 5);
      break;
    case Config::Downlink:
      tasks = {{{frmsym.frm_id, frmsym.sym_id, TaskType::kLoad},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kEncode},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kModulate},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kPrecode},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kDiFFT},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kStore}}};
      status = fifo.enqueue_bulk(prod_token, tasks.begin(), 6);
      break;
    default:
      return false;
  }
  return status;
}

inline bool enqueue_task(auto &fifo, SymbolId frmsym) {
  bool status = true;
  static constexpr uint8_t kMaxNumTasks = 6;
  std::array<TaskId, kMaxNumTasks> tasks;
  switch (gconfig.frame.gidx_sym[frmsym.sym_id]) {
    case Config::NLPilot:
      tasks = {{{frmsym.frm_id, frmsym.sym_id, TaskType::kLoad},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kPFFT}}};
      status = fifo.enqueue_bulk(tasks.begin(), 2);
      break;
    case Config::Pilot:
      tasks = {{{frmsym.frm_id, frmsym.sym_id, TaskType::kLoad},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kPFFT},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kBeam},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kBeamNM}}};
      status = fifo.enqueue_bulk(tasks.begin(),
                                 gconfig.frame.downlink_syms > 0 ? 4 : 3);
      break;
    case Config::Uplink:
      tasks = {{{frmsym.frm_id, frmsym.sym_id, TaskType::kLoad},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kUFFT},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kEqual},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kDecode},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kStore}}};
      status = fifo.enqueue_bulk(tasks.begin(), 5);
      break;
    case Config::Downlink:
      tasks = {{{frmsym.frm_id, frmsym.sym_id, TaskType::kLoad},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kEncode},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kModulate},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kPrecode},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kDiFFT},
                {frmsym.frm_id, frmsym.sym_id, TaskType::kStore}}};
      status = fifo.enqueue_bulk(tasks.begin(), 6);
      break;
    default:
      return false;
  }
  return status;
}

}  // namespace mega