add_executable(
  test_ldpc
  test_ldpc.cc
)

target_link_libraries(
  test_ldpc
  GTest::gtest_main
  ${CUDA_LIBS}
  ${CUPHY_LIBS}
)

include(GoogleTest)
gtest_discover_tests(test_ldpc)