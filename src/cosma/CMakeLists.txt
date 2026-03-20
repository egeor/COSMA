set(INSTALLED_TARGETS_LIST "")
set(cosma_src_files
  buffer.cpp
  communicator.cpp
  context.cpp
  interval.cpp
  layout.cpp
  local_multiply.cpp
  mapper.cpp
  math_utils.cpp
  matrix.cpp
  memory_pool.cpp
  multiply.cpp
  one_sided_communicator.cpp
  strategy.cpp
  two_sided_communicator.cpp
  random_generator.hpp
  cinterface.cpp
  environment_variables.cpp)

if (COSMA_GPU_BACKEND MATCHES "OFF")
	LIST(APPEND cosma_src_files blas.cpp)
endif ()

if (COSMA_WITH_SFC_GEMM)
  LIST(APPEND cosma_src_files sfc_gemm_wrapper.cpp)
  LIST(APPEND cosma_src_files bf16_instantiations.cpp)
endif ()

if (COSMA_GPU_BACKEND MATCHES "ROCM" OR COSMA_GPU_BACKEND MATCHES "CUDA")
  list(APPEND cosma_src_files "pinned_buffers.cpp")
  if (COSMA_WITH_NCCL OR COSMA_WITH_RCCL)
    list(APPEND cosma_src_files "gpu/nccl_utils.cpp")
  endif()
  if (COSMA_WITH_GPU_AWARE_MPI)
    list(APPEND cosma_src_files "gpu/gpu_aware_mpi_utils.cpp")
  endif()
endif()

add_library(cosma ${cosma_src_files})

target_include_directories(cosma PUBLIC
  $<BUILD_INTERFACE:${cosma_SOURCE_DIR}/src>
)

# sfc_ca_gemm / libxsmm includes and linkage
if (COSMA_WITH_SFC_GEMM)
  target_include_directories(cosma PRIVATE "${SFC_CA_GEMM_INCLUDE_DIR}")
  if (TARGET PkgConfig::LIBXSMM)
    target_link_libraries(cosma PRIVATE PkgConfig::LIBXSMM)
  elseif (TARGET libxsmm::libxsmm)
    # When libxsmm is fetched via FetchContent, link by file path
    # to avoid polluting the install export set.
    if (TARGET xsmm)
      target_include_directories(cosma PRIVATE
        $<TARGET_PROPERTY:xsmm,INTERFACE_INCLUDE_DIRECTORIES>)
      target_link_libraries(cosma PRIVATE $<TARGET_FILE:xsmm>)
      # Ensure xsmm is built before cosma
      add_dependencies(cosma xsmm)
    else()
      target_link_libraries(cosma PRIVATE libxsmm::libxsmm)
    endif()
  endif()
  find_package(OpenMP COMPONENTS CXX)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(cosma PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

target_compile_features(cosma PUBLIC cxx_std_14)
target_link_libraries(cosma PUBLIC
  MPI::MPI_CXX
  costa::costa
  $<TARGET_NAME_IF_EXISTS:roc::rccl>
  $<TARGET_NAME_IF_EXISTS:cosma::nccl>
  $<$<STREQUAL:${COSMA_GPU_VENDOR},"OFF">:cosma::BLAS::blas>
  $<TARGET_NAME_IF_EXISTS:Tiled-MM::Tiled-MM>
  $<$<STREQUAL:${COSMA_GPU_BACKEND},CUDA>:Tiled-MM::Tiled-MM>
  $<$<STREQUAL:${COSMA_GPU_BACKEND},ROCM>:Tiled-MM::Tiled-MM>
  $<$<BOOL:${COSMA_WITH_PROFILING}>:semiprof::semiprof>
  $<$<BOOL:${COSMA_SCALAPACK}>:cosma::scalapack::scalapack>)

target_compile_definitions(cosma PUBLIC
                              $<$<BOOL:${COSMA_WITH_NCCL}>:COSMA_WITH_NCCL>
                              $<$<STREQUAL:${COSMA_GPU_BACKEND},"ROCM">:__HIP_PLATFORM_HCC__>
                              $<$<BOOL:${COSMA_WITH_GPU_AWARE_MPI}>:COSMA_WITH_GPU_AWARE_MPI>
                              $<$<BOOL:${COSMA_WITH_NCCL}>:COSMA_WITH_NCCL>
                              $<$<BOOL:${COSMA_WITH_RCCL}>:COSMA_WITH_NCCL>
                              $<$<STREQUAL:${COSMA_BLAS_VENDOR},MKL>:COSMA_WITH_MKL_BLAS>
                              $<$<STREQUAL:${COSMA_BLAS_VENDOR},BLIS>:COSMA_WITH_BLIS_BLAS>
                              $<$<NOT:$<IN_LIST:${COSMA_BLAS_VENDOR},"MKL;BLISi;OFF">>:COSMA_WITH_BLAS>
                              $<$<STREQUAL:${COSMA_GPU_BACKEND},CUDA>:COSMA_HAVE_GPU>
                              $<$<STREQUAL:${COSMA_GPU_BACKEND},ROCM>:COSMA_HAVE_GPU>
                              $<$<BOOL:${COSMA_WITH_SFC_GEMM}>:COSMA_WITH_SFC_GEMM>
                              PRIVATE
                              $<$<BOOL:${COSMA_WITH_PROFILING}>:COSMA_WITH_PROFILING>)


list(APPEND INSTALLED_TARGETS_LIST "cosma")

# if SCALAPACK is found and cosma_pxgemm library is not already created
# then create it here and link it to the profiler if needed
# build as a shared library is necessary here because of the function interposing
if(COSMA_SCALAPACK)
  if (NOT TARGET cosma_pxgemm AND BUILD_SHARED_LIBS)
    add_library(cosma_pxgemm scalapack.cpp
      pxgemm_params.hpp
      cosma_pxgemm.cpp
      pxgemm.cpp
    )

    target_link_libraries(cosma_pxgemm PUBLIC cosma
    $<$<BOOL:${COSMA_WITH_PROFILING}>:semiprof::semiprof>)

    target_compile_definitions(cosma_pxgemm PRIVATE $<$<BOOL:${COSMA_WITH_PROFILING}>:COSMA_WITH_PROFILING>)
    list(APPEND INSTALLED_TARGETS_LIST "cosma_pxgemm")
  endif()

  # this is a library exposing the prefixed scalapack API (with cosma/COSMA prefix)
  # it is aimed for users who don't want to overwrite the available scalapack API with cosma.
  # if SCALAPACK is found and cosma_prefixed_pxgemm library is not already created
  # then create it here and link it to the profiler if needed
    add_library(cosma_prefixed_pxgemm scalapack.cpp
      pxgemm_params.hpp
      prefixed_pxgemm.cpp
      cosma_pxgemm.cpp
    )
    target_link_libraries(cosma_prefixed_pxgemm
    PUBLIC cosma
    PRIVATE $<$<BOOL:${COSMA_WITH_PROFILING}>:semiprof::semiprof>)
    target_compile_definitions(cosma_prefixed_pxgemm PRIVATE $<$<BOOL:${COSMA_WITH_PROFILING}>:COSMA_WITH_PROFILING>)
    list(APPEND INSTALLED_TARGETS_LIST "cosma_prefixed_pxgemm")

  # the following library is aimed only for testing purposes
  # it provides templated cosma::pxgemm call without
  # pxgemm.h, so that pxgemm calls of scalapack are not overwritten
  # and can still be compared to scalapack for correctness check
  if(NOT TARGET cosma_pxgemm_cpp)
    add_library(cosma_pxgemm_cpp scalapack.cpp
      pxgemm_params.hpp
      cosma_pxgemm.cpp
    )
    target_link_libraries(cosma_pxgemm_cpp
                            PUBLIC cosma
                            PRIVATE $<$<BOOL:${COSMA_WITH_PROFILING}>:semiprof::semiprof>)

    target_compile_definitions(cosma_pxgemm_cpp
                                  PRIVATE $<$<BOOL:${COSMA_WITH_PROFILING}>:COSMA_WITH_PROFILING>)
    list(APPEND INSTALLED_TARGETS_LIST "cosma_pxgemm_cpp")
  endif()
endif()

  install(TARGETS ${INSTALLED_TARGETS_LIST}
    EXPORT cosma_targets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  install(EXPORT cosma_targets
    FILE cosmaTargets.cmake
    NAMESPACE cosma::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cosma")
