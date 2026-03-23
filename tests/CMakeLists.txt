add_custom_target(tests COMMENT "Builds all tests.")

# CACHE GTest main into an OBJECT library to avoid recompiling.
#
add_library(main_gtest_mpi OBJECT main_gtest_mpi.cpp)
target_link_libraries(main_gtest_mpi PRIVATE MPI::MPI_CXX gtest_mpi)

add_library(main_gtest OBJECT main_gtest.cpp)
target_link_libraries(main_gtest PRIVATE gtest)

#  Unit tests
#
function(add_cosma_mpi_test test_name num_procs)
    set(tgt_ test.${test_name})
    add_executable(${tgt_} ${test_name}.cpp)
    target_link_libraries(${tgt_} PRIVATE main_gtest_mpi gtest_mpi ${ARGN})
    target_include_directories(${tgt_} PRIVATE .)
    add_test(NAME ${tgt_}
             WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${num_procs}
                     ${MPIEXEC_PREFLAGS} ./${tgt_} ${MPIEXEC_POSTFLAGS})
    add_dependencies(tests ${tgt_})
    install(TARGETS ${tgt_} DESTINATION "${CMAKE_INSTALL_BINDIR}")
endfunction()

add_executable(test.mapper mapper.cpp)
target_link_libraries(test.mapper PRIVATE main_gtest gtest cosma)
add_test(NAME test.mapper COMMAND test.mapper)
add_dependencies(tests test.mapper)
install(TARGETS test.mapper DESTINATION "${CMAKE_INSTALL_BINDIR}")

if(NOT COSMA_SCALAPACK MATCHES "OFF")
    add_cosma_mpi_test(pdgemm 16 cosma_pxgemm_cpp)
endif()

add_cosma_mpi_test(multiply_using_layout 4 cosma)
add_cosma_mpi_test(multiply 16 cosma)
add_cosma_mpi_test(scalar_matmul 8 cosma)

if(COSMA_WITH_SFC_GEMM)
    add_executable(test_bf16_mpi test_bf16_mpi.cpp)
    target_link_libraries(test_bf16_mpi PRIVATE cosma MPI::MPI_CXX)
    if(TARGET PkgConfig::LIBXSMM)
        target_link_libraries(test_bf16_mpi PRIVATE PkgConfig::LIBXSMM)
    elseif(TARGET xsmm)
        target_include_directories(test_bf16_mpi PRIVATE
            $<TARGET_PROPERTY:xsmm,INTERFACE_INCLUDE_DIRECTORIES>)
        target_link_libraries(test_bf16_mpi PRIVATE $<TARGET_FILE:xsmm>)
        add_dependencies(test_bf16_mpi xsmm)
    endif()
    install(TARGETS test_bf16_mpi DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
