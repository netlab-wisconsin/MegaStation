#pragma once

#include <parallel_hashmap/phmap_utils.h>

#include <cstdint>
#include <string>

#include "config/config.h"

namespace mega {

enum class TaskStatus : int32_t {
  kIdle = 0,    //!< Task Not used
  kSched = -1,  //!< Task Scheduled
  kDone = -2    //!< Task Completed
};

inline std::string TaskStatusToString(TaskStatus s) {
  switch (s) {
    case TaskStatus::kIdle:
      return "kIdle";
      break;
    case TaskStatus::kSched:
      return "kSched";
      break;
    case TaskStatus::kDone:
      return "kDone";
      break;
    default:
      return std::to_string(static_cast<int32_t>(s));
  }
}

enum class Policy : uint8_t { kByFrame, kBySymbol, kByTask, kByMega };

inline std::string PolicyToString(Policy p) {
  switch (p) {
    case Policy::kByFrame:
      return "byFrame";
      break;
    case Policy::kBySymbol:
      return "bySymbol";
      break;
    case Policy::kByTask:
      return "byTask";
      break;
    case Policy::kByMega:
      return "byMega";
      break;
    default:
      return "unKnown";
  }
}

enum class TaskType : uint16_t {
  kNullT,     //!< Null Task
  kLoad,      //!< Load Data
  kStore,     //!< Store Data
  kPFFT,      //!< Pilot FFT
  kBeam,      //!< Beamforming
  kBeamNM,    //!< Normalized Beamforming
  kUFFT,      //!< Uplink FFT
  kEqual,     //!< Equalization
  kDecode,    //!< Decoding
  kEncode,    //!< Encoding
  kModulate,  //!< Modulation
  kPrecode,   //!< Demodulation
  kDiFFT,     //!< Downlink iFFT
};

inline std::string TaskTypeToString(TaskType t) {
  switch (t) {
    case TaskType::kNullT:
      return "kNullT";
      break;
    case TaskType::kLoad:
      return "kLoad";
      break;
    case TaskType::kStore:
      return "kStore";
      break;
    case TaskType::kPFFT:
      return "kPFFT";
      break;
    case TaskType::kBeam:
      return "kBeam";
      break;
    case TaskType::kBeamNM:
      return "kBeamNM";
      break;
    case TaskType::kUFFT:
      return "kUFFT";
      break;
    case TaskType::kEqual:
      return "kEqual";
      break;
    case TaskType::kDecode:
      return "kDecode";
      break;
    case TaskType::kEncode:
      return "kEncode";
      break;
    case TaskType::kModulate:
      return "kModulate";
      break;
    case TaskType::kPrecode:
      return "kPrecode";
      break;
    case TaskType::kDiFFT:
      return "kDiFFT";
      break;
    default:
      return "Unknown";
  }
}

using DeviceId = int32_t;

struct FrameId {
  uint32_t frm_id;
  inline friend size_t hash_value(const FrameId &frame_id) {
    return phmap::HashState().combine(0, frame_id.frm_id);
  }
  inline bool operator==(const FrameId &frame_id) const {
    return frm_id == frame_id.frm_id;
  }
  inline bool operator<(const FrameId &frame_id) const {
    return frm_id < frame_id.frm_id;
  }
};

struct SymbolId : FrameId {
  uint16_t sym_id;
  inline friend size_t hash_value(const SymbolId &symbol_id) {
    return phmap::HashState().combine(0, symbol_id.frm_id, symbol_id.sym_id);
  }
  inline bool operator==(const SymbolId &symbol_id) const {
    return frm_id == symbol_id.frm_id && sym_id == symbol_id.sym_id;
  }
};

struct TaskId : SymbolId {
  TaskType job_id;
  inline friend size_t hash_value(const TaskId &task_id) {
    return phmap::HashState().combine(0, task_id.frm_id, task_id.sym_id,
                                      task_id.job_id);
  }
  inline bool operator==(const TaskId &task_id) const {
    return frm_id == task_id.frm_id && sym_id == task_id.sym_id &&
           job_id == task_id.job_id;
  }
};

struct ExecInfo {
  TaskId task_id;
  union {
    uint32_t circ_id;
    uint32_t rank_id;
  };
};

struct TaskTypeProp {
  static inline uint32_t local_job_offset(TaskType job_id) {
    switch (job_id) {
      case TaskType::kPFFT:
      case TaskType::kBeam:
      case TaskType::kBeamNM:
        return static_cast<uint32_t>(job_id) -
               static_cast<uint32_t>(TaskType::kPFFT) + 1;  // 1-load
      case TaskType::kUFFT:
      case TaskType::kEqual:
      case TaskType::kDecode:
        return static_cast<uint32_t>(job_id) -
               static_cast<uint32_t>(TaskType::kUFFT) + 2;  // 2-load & store
      case TaskType::kEncode:
      case TaskType::kModulate:
      case TaskType::kPrecode:
      case TaskType::kDiFFT:
        return static_cast<uint32_t>(job_id) -
               static_cast<uint32_t>(TaskType::kEncode) + 2;  // 2-load & store
      case TaskType::kLoad:
        return 0;
      case TaskType::kStore:
        return 1;
      default:
        return 0;
    }
  }

  static inline uint32_t jobs_per_symbol(Config::Symbol symbol_type) {
    switch (symbol_type) {
      case Config::NLPilot:
        return 1 + 1;  // 1-kLoad, 1-kPFFT
      case Config::Pilot:
        return (gconfig.frame.downlink_syms > 0
                    ? static_cast<uint32_t>(TaskType::kBeamNM)
                    : static_cast<uint32_t>(TaskType::kBeam)) -
               static_cast<uint32_t>(TaskType::kPFFT) + 1 + 1;  // 1-load
      case Config::Uplink:
        return static_cast<uint32_t>(TaskType::kDecode) -
               static_cast<uint32_t>(TaskType::kUFFT) + 1 +
               2;  // 2-load & store
      case Config::Downlink:
        return static_cast<uint32_t>(TaskType::kDiFFT) -
               static_cast<uint32_t>(TaskType::kEncode) + 1 +
               2;  // 2-load & store
      default:
        return 0;
    }
  }

  static inline uint32_t symbol_offset(uint32_t symbol_id) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < symbol_id; i++) {
      offset += jobs_per_symbol(gconfig.frame.gidx_sym[i]);
    }
    return offset;
  }

  static inline Config::Symbol symbol_type(TaskType job_id) {
    switch (job_id) {
      case TaskType::kPFFT:
      case TaskType::kBeam:
      case TaskType::kBeamNM:
        return Config::Pilot;
      case TaskType::kUFFT:
      case TaskType::kEqual:
      case TaskType::kDecode:
        return Config::Uplink;
      case TaskType::kEncode:
      case TaskType::kModulate:
      case TaskType::kPrecode:
      case TaskType::kDiFFT:
        return Config::Downlink;
      default:
        return Config::Pilot;
    }
  }

  static inline bool last_job(const TaskId task_id) {
    switch (task_id.job_id) {
      case TaskType::kPFFT:
        return gconfig.frame.gidx_sym[task_id.sym_id] == Config::NLPilot;
      case TaskType::kBeam:
        return gconfig.frame.downlink_syms > 0 ? false : true;
      case TaskType::kBeamNM:
      case TaskType::kStore:
        return true;
      case TaskType::kDecode:
      case TaskType::kDiFFT:
      default:
        return false;
    }
  }

  static inline uint32_t jobs_per_frame() {
    if (jobsPerFrame == 0) {
      jobsPerFrame =
          (gconfig.frame.pilot_syms - 1) * jobs_per_symbol(Config::NLPilot) +
          jobs_per_symbol(Config::Pilot) +
          gconfig.frame.uplink_syms * jobs_per_symbol(Config::Uplink) +
          gconfig.frame.downlink_syms * jobs_per_symbol(Config::Downlink);
    }
    return jobsPerFrame;
  }

  static inline const uint32_t offset(const TaskId &task_id) {
    return task_id.frm_id * jobs_per_frame() + symbol_offset(task_id.sym_id) +
           local_job_offset(task_id.job_id);
  }

  constexpr static uint32_t kNumTaskType =
      static_cast<uint32_t>(TaskType::kDiFFT) -
      static_cast<uint32_t>(TaskType::kNullT) + 1;

 private:
  static inline uint32_t jobsPerFrame;
};

enum class MemOpType : uint8_t {
  kNullMO,    //!< Null Memory Operation
  kPreFFT,    //!< In Memory for PreFFT
  kCSI,       //!< In/Out Memory for CSI
  kNmCSI,     //!< Out Memory for Normalized CSI
  kUFFT,      //!< Out Memory for Uplink FFT
  kEqual,     //!< Out Memory for Equalization
  kDecode,    //!< Out Memory for Decoding
  kUncode,    //!< In Memory for PreEncoding
  kEncode,    //!< Out Memory for Encoding
  kModulate,  //!< Out Memory for Modulation
  kPrecode,   //!< Out Memory for PreDecoding
  kDiFFT,     //!< Out Memory for Downlink iFFT
  kPFFTOp,    //!< Pilot FFT Operation
  kBeamOp,    //!< Beamforming Operation
  kUFFTOp,    //!< Uplink FFT Operation
  kDiFFTOp,   //!< Downlink iFFT Operation
};

inline std::string MemOpTypeToString(MemOpType m) {
  switch (m) {
    case MemOpType::kNullMO:
      return "kNullMO";
      break;
    case MemOpType::kPreFFT:
      return "kPreFFT";
      break;
    case MemOpType::kCSI:
      return "kCSI";
      break;
    case MemOpType::kNmCSI:
      return "kNmCSI";
      break;
    case MemOpType::kUFFT:
      return "kUFFT";
      break;
    case MemOpType::kEqual:
      return "kEqual";
      break;
    case MemOpType::kDecode:
      return "kDecode";
      break;
    case MemOpType::kUncode:
      return "kUncode";
      break;
    case MemOpType::kEncode:
      return "kEncode";
      break;
    case MemOpType::kModulate:
      return "kModulate";
      break;
    case MemOpType::kPrecode:
      return "kPrecode";
      break;
    case MemOpType::kDiFFT:
      return "kDiFFT";
      break;
    case MemOpType::kPFFTOp:
      return "kPFFTOp";
      break;
    case MemOpType::kBeamOp:
      return "kBeamOp";
      break;
    case MemOpType::kUFFTOp:
      return "kUFFTOp";
      break;
    case MemOpType::kDiFFTOp:
      return "kDiFFTOp";
      break;
    default:
      return "Unknown";
  }
}

struct MemOpId : SymbolId {
  MemOpType type_id;
};

struct MemOpTypeProp {
  static inline const uint32_t local_memop_offset(const MemOpType type_id) {
    switch (type_id) {
      case MemOpType::kPreFFT:
        return 0;
      case MemOpType::kCSI:
      case MemOpType::kNmCSI:
        return static_cast<uint32_t>(type_id) -
               static_cast<uint32_t>(MemOpType::kPreFFT) + 1;  // 1-kPFFTOp
      case MemOpType::kUFFT:
      case MemOpType::kEqual:
      case MemOpType::kDecode:
        return static_cast<uint32_t>(type_id) -
               static_cast<uint32_t>(MemOpType::kUFFT) + 1;  // 1-kPreFFT
      case MemOpType::kUncode:
      case MemOpType::kEncode:
      case MemOpType::kModulate:
      case MemOpType::kPrecode:
      case MemOpType::kDiFFT:
        return static_cast<uint32_t>(type_id) -
               static_cast<uint32_t>(MemOpType::kUncode);
      case MemOpType::kPFFTOp:
        return 1;  // 1-kPreFFT,kPFFTOp
      case MemOpType::kBeamOp:
        return static_cast<uint32_t>(type_id) -
               static_cast<uint32_t>(MemOpType::kPFFTOp) + 2 +
               (gconfig.frame.downlink_syms > 0);  // 2-kPreFFT~kCSI,1-kNmCSI
      case MemOpType::kUFFTOp:
        return 4;  // 4-kPreFFT,kUFFT~kEqual
      case MemOpType::kDiFFTOp:
        return 5;  // 5-kUncode~kDiFFT
      default:
        return 0;
    }
  }

  static inline const uint32_t memops_per_symbol(
      const Config::Symbol symbol_type) {
    switch (symbol_type) {
      case Config::NLPilot:
        return 1 + 1;  // 1-kPreFFT, 1-kPFFTOp
      case Config::Pilot:
        return (gconfig.frame.downlink_syms > 0
                    ? static_cast<uint32_t>(MemOpType::kNmCSI)
                    : static_cast<uint32_t>(MemOpType::kCSI)) -
               static_cast<uint32_t>(MemOpType::kPreFFT) + 1 +
               2;  // 2-kPFFTOp, kBeamOp
      case Config::Uplink:
        return static_cast<uint32_t>(MemOpType::kDecode) -
               static_cast<uint32_t>(MemOpType::kUFFT) + 1 +
               2;  // 2-kPreFFT,kUFFTOp
      case Config::Downlink:
        return static_cast<uint32_t>(MemOpType::kDiFFT) -
               static_cast<uint32_t>(MemOpType::kUncode) + 1 + 1;  // 1-kDiFFTOp
      default:
        return 0;
    }
  }

  static inline const uint32_t symbol_offset(const uint32_t symbol_id) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < symbol_id; i++) {
      offset += memops_per_symbol(gconfig.frame.gidx_sym[i]);
    }
    return offset;
  }

  static inline const uint32_t memops_per_frame() {
    if (memopsPerFrame == 0) {
      memopsPerFrame =
          (gconfig.frame.pilot_syms - 1) * memops_per_symbol(Config::NLPilot) +
          memops_per_symbol(Config::Pilot) +
          gconfig.frame.uplink_syms * memops_per_symbol(Config::Uplink) +
          gconfig.frame.downlink_syms * memops_per_symbol(Config::Downlink);
    }
    return memopsPerFrame;
  }

  static inline const uint8_t mem_refcount(const MemOpType type_id) {
    switch (type_id) {
      case MemOpType::kPreFFT:
      case MemOpType::kUFFT:
      case MemOpType::kEqual:
      case MemOpType::kDecode:
      case MemOpType::kUncode:
      case MemOpType::kEncode:
      case MemOpType::kModulate:
      case MemOpType::kPrecode:
      case MemOpType::kDiFFT:
        return 1;
      case MemOpType::kCSI:
        return gconfig.frame.uplink_syms + (gconfig.frame.downlink_syms > 0) +
               1;  // 1-last pilot
      case MemOpType::kNmCSI:
        return gconfig.frame.downlink_syms;
      default:
        return 0;
    }
  }

  static inline const uint32_t offset(const MemOpId &memop_id) {
    uint32_t sym_id = memop_id.sym_id;
    if (memop_id.type_id == MemOpType::kCSI)
      sym_id = gconfig.frame.pilots.back();

    return memop_id.frm_id * memops_per_frame() + symbol_offset(sym_id) +
           local_memop_offset(memop_id.type_id);
  }

  static constexpr uint32_t kNumMemOpType =
      static_cast<uint32_t>(MemOpType::kDiFFTOp) -
      static_cast<uint32_t>(MemOpType::kPreFFT) + 1;
  static constexpr uint32_t kNumMemType =
      static_cast<uint32_t>(MemOpType::kDiFFT) -
      static_cast<uint32_t>(MemOpType::kPreFFT) + 1;

 private:
  static inline uint32_t memopsPerFrame;
};

}  // namespace mega