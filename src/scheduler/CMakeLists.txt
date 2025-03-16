set(SCHEDULER_SOURCE_FILES gdr_utils.cc scheduler.cc cpu_buffer.cc
                           mem_manager.cc gpu_engine.cc exec_status.cc)

add_library(scheduler_objects OBJECT ${SCHEDULER_SOURCE_FILES})
target_link_libraries(
  scheduler_objects
  spdlog::spdlog
  ${CUDA_LIBS}
  ${CUPHY_LIBS}
  gdrapi
  phmap
  readerwriterqueue
  concurrentqueue)
# target_compile_definitions(scheduler_objects PRIVATE __DEBUG__)

add_dependencies(scheduler_objects gdrcopy)
