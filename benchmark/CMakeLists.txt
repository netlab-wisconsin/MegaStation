add_executable(mega_latency mega_latency.cc)
target_link_libraries(mega_latency ${CUDA_LIBS} ${CUPHY_LIBS} ${THIRD_LIBS}
                      gdrapi ${MEGA_OBJECTS})

add_executable(mega_throughput mega_throughput.cc)
target_link_libraries(mega_throughput ${CUDA_LIBS} ${CUPHY_LIBS} ${THIRD_LIBS}
                      gdrapi ${MEGA_OBJECTS})

add_executable(mega_reliable mega_reliable.cc)
target_link_libraries(mega_reliable ${CUDA_LIBS} ${CUPHY_LIBS} ${THIRD_LIBS}
                      gdrapi ${MEGA_OBJECTS})

add_executable(mega_fragment mega_fragment.cc arti_kernel.cu)
target_link_libraries(mega_fragment ${CUDA_LIBS} ${CUPHY_LIBS} ${THIRD_LIBS}
                      gdrapi ${MEGA_OBJECTS})
