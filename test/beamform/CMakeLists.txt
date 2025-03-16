add_executable(
  test_beamform
  test_beamform.cc
)

target_link_libraries(
  test_beamform
  GTest::gtest_main
  ${CUDA_LIBS}
  cuda_objects
)

include(GoogleTest)
gtest_discover_tests(test_beamform)