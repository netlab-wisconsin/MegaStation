add_executable(latency_cdf latency_cdf.cc)

target_link_libraries(latency_cdf GTest::gtest_main ${CUDA_LIBS} ${CUPHY_LIBS}
                      ${THIRD_LIBS} ${MEGA_OBJECTS})

add_executable(latency_breakdown latency_breakdown.cc)

target_link_libraries(latency_breakdown GTest::gtest_main ${CUDA_LIBS}
                      ${CUPHY_LIBS} ${THIRD_LIBS} ${MEGA_OBJECTS})

add_executable(latency_partial latency_partial.cc arti_kernel.cu)

target_link_libraries(latency_partial GTest::gtest_main ${CUDA_LIBS}
                      ${CUPHY_LIBS} ${THIRD_LIBS} ${MEGA_OBJECTS})

add_executable(latency_delay latency_delay.cc arti_kernel.cu)

target_link_libraries(latency_delay GTest::gtest_main ${CUDA_LIBS}
                      ${CUPHY_LIBS} ${THIRD_LIBS} ${MEGA_OBJECTS})

add_executable(latency_fragment latency_fragment.cc arti_kernel.cu)

target_link_libraries(latency_fragment GTest::gtest_main ${CUDA_LIBS}
                      ${CUPHY_LIBS} ${THIRD_LIBS} ${MEGA_OBJECTS})

include(GoogleTest)
gtest_discover_tests(latency_cdf)
gtest_discover_tests(latency_breakdown)
gtest_discover_tests(latency_partial)
gtest_discover_tests(latency_delay)
gtest_discover_tests(latency_fragment)
