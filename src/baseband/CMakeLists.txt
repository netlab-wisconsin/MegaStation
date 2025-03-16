set(BASEBAND_SOURCE_FILES process.cc buffers.cc)

add_library(baseband_objects OBJECT ${BASEBAND_SOURCE_FILES})
target_link_libraries(baseband_objects spdlog::spdlog ${CUPHY_LIBS}
                      ${CUDA_LIBS})
