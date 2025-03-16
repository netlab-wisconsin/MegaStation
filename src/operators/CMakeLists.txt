set(CUDA_SOURCE_FILES
    scrambler/scrambler.cu modulation/modulation.cu sgemv/equalize.cu
    sgemv/precode.cu beamform/beamform.cu fft/fft_op.cu)

add_library(cuda_objects OBJECT ${CUDA_SOURCE_FILES})
target_link_libraries(cuda_objects ${CUDA_LIBS})

set(LDPC_FAC_FILES ldpc/encoder.cc ldpc/decoder.cc)

add_library(ldpcfac_objects OBJECT ${LDPC_FAC_FILES})
target_link_libraries(ldpcfac_objects ${CUDA_LIBS} ${CUPHY_LIBS})
