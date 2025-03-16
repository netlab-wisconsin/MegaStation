/**
 * @file cgemm.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief CUTLASS CGEMM Strided Batched
 * @version 0.1
 * @date 2024-04-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm_complex.h>
#include <cutlass/gemm/kernel/gemm_batched.h>
#include <cutlass/numeric_types.h>

namespace mega {

using cutComplex = cutlass::complex<float>;

struct CgemmStridedBatched {
  // clang-format off
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<float>,
      1,
      cutlass::complex<float>,
      cutlass::complex<float>
    >;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;

  using DefaultCgemm = cutlass::gemm::kernel::DefaultGemmComplex<
    cutlass::complex<float>, cutlass::layout::ColumnMajor,
    cutlass::complex<float>, cutlass::layout::RowMajor,
    cutlass::complex<float>, cutlass::layout::RowMajor,
    cutlass::complex<float>,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    ThreadblockShape,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    5,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddComplex,
    false
  >::GemmKernel;

  using GemmKernel = cutlass::gemm::kernel::GemmBatched<typename DefaultCgemm::Mma, typename DefaultCgemm::Epilogue, ThreadblockSwizzle>;

  using Element = cutlass::complex<float>;
  
  struct Arguments {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<Element const, cutlass::layout::ColumnMajor> ref_A;
    int64_t stride_A;
    cutlass::TensorRef<Element const, cutlass::layout::RowMajor> ref_B;
    int64_t stride_B;
    cutlass::TensorRef<Element const, cutlass::layout::RowMajor> ref_C;
    int64_t stride_C;
    cutlass::TensorRef<Element, cutlass::layout::RowMajor> ref_D;
    int64_t stride_D;
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;

    CUTLASS_HOST_DEVICE
    Arguments() {}

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::TensorRef<Element const, cutlass::layout::ColumnMajor> ref_A_,
      int64_t stride_A_,
      cutlass::TensorRef<Element const, cutlass::layout::RowMajor> ref_B_,
      int64_t stride_B_,
      cutlass::TensorRef<Element const, cutlass::layout::RowMajor> ref_C_,
      int64_t stride_C_,
      cutlass::TensorRef<Element, cutlass::layout::RowMajor> ref_D_,
      int64_t stride_D_,
      int batch_count_ = 1,
      typename EpilogueOutputOp::Params epilogue_ = 
        typename EpilogueOutputOp::Params()
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      stride_A(stride_A_),
      ref_B(ref_B_),
      stride_B(stride_B_),
      ref_C(ref_C_),
      stride_C(stride_C_),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_) { }
  };
private:
  typename GemmKernel::Params params_;

public:
  CgemmStridedBatched() { }

  void operator()(
    Arguments const &args, 
    cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.batch_count);
    
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.stride_A,
      args.ref_B.non_const_ref(),
      args.stride_B,
      args.ref_C.non_const_ref(),
      args.stride_C,
      args.ref_D,
      args.stride_D,
      args.epilogue,
      args.batch_count
    };

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
  }
};

}  // namespace mega