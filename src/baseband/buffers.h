/**
 * @file buffers.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Baseband buffers header file
 * @version 0.1
 * @date 2023-12-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "utils/types.h"

namespace mega {

class BandBuffers {
 public:
  static ArrStream Streams;  //!< CUDA streams

  static Buffer HostBufferRecv;    //!< receive global buffer for CPU
  static Buffer DeviceBufferRecv;  //!< receive global buffer for CPU

  static Buffer
      DeviceBufferCsi;  //!< CSI global buffer for GPU (output of Pilot FFT)
  static Buffer DeviceBufferNmCsi;  //!< Normalized CSI global buffer for GPU
                                    //!< (for precode)

  static Buffer DeviceBufferUFFT;         //!< Uplink FFT global buffer for GPU
                                          //!< (workspace + output)
  static BufferPhy DeviceBufferEqualize;  //!< Equalize global buffer for GPU
  static BufferPhy DeviceBufferDecode;    //!< Decode global buffer for GPU (to
                                          //!< MAC layer)

  static BufferPhy HostBufferMacSend;  //!< Decode global buffer for CPU (to MAC
                                       //!< layer)
  static BufferPhy HostBufferMacRecv;  //!< Decode global buffer for CPU (from
                                       //!< MAC layer)

  static BufferPhy
      DeviceBufferUncode;  //!< Downlink encode input global buffer for GPU
  static BufferPhy
      DeviceBufferEncode;  //!< Downlink encode output global buffer for GPU

  static Buffer
      DeviceBufferModulate;  //!< Downlink modulate global buffer for GPU
  static Buffer
      DeviceBufferPrecode;         //!< Downlink precode global buffer for GPU
  static Buffer DeviceBufferIFFT;  //!< Downlink IFFT global buffer for GPU

  static Buffer HostBufferSend;  //!< send global buffer for CPU

  /**
   * @brief Construct a new Band Process object
   *
   */
  BandBuffers() {}
  /**
   * @brief Initialize static buffers
   *
   */
  static void init();

  /**
   * @brief Destroy static buffers
   *
   */
  static void destroy();
};

}  // namespace mega