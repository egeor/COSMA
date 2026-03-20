/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include "sfc_ca_gemm.hpp"

#define PRINT_THREAD_WORK_ASSIGNMENT

// Define struct to store work per thread
typedef struct
{
  long thread_id;
  long m_block;
  long n_block;
  long k_layer;
} thread_work_t;

template<typename DType, int skip_c_reduction = 0>
void run_gemm(gemm_config_t *config, DType *A, DType *B, typename output_type<DType>::type *C) {
  // Unpack configuration struct
  using CType = typename output_type<DType>::type;
  long M = config->M, N = config->N, K = config->K, Mb = config->Mb, Nb = config->Nb, Kb = config->Kb, bm = config->bm, bn = config->bn, bk = config->bk, K_layers = config->K_layers, brcount = config->brcount;
  long unblocked_bc = config->unblocked_bc;
  CType **scratch_C = (CType**)config->gemm_scratch;
  unsigned char *sfc_index_map = config->sfc_index_map;
  unsigned int index_tsize = config->index_tsize;
  libxsmm_gemmfunction brgemm_kernel = config->brgemm_kernel;
  libxsmm_meltwfunction_unary zero_kernel = config->zero_kernel;
  libxsmm_tilecfgfunction tileconfig_kernel = config->tileconfig_kernel;
  libxsmm_tilecfgfunction tilerelease_kernel = config->tilerelease_kernel;
  libxsmm_meltwfunction_binary l_add_kernel = config->l_add_kernel;
  libxsmm_meltwfunction_unary l_reduce_kernel = config->l_reduce_kernel;
  long Kb_per_layer = (Kb + K_layers - 1) / K_layers;
  long Kb_last_layer = Kb - (K_layers - 1) * Kb_per_layer;
  long K_rounds_per_layer = (Kb_per_layer + brcount - 1) / brcount;

  // Upfront packing of B from flat to blocked (unblocked_bc == 2)
  if (unblocked_bc == 2) {
    DType *scratch_B_buf = (DType *)config->scratch_B;
    libxsmm_meltwfunction_unary b_xform_kernel = config->b_xform_kernel;
    #pragma omp parallel for collapse(2)
    for (int i_n = 0; i_n < Nb; i_n++) {
      for (int i_k = 0; i_k < Kb; i_k++) {
        libxsmm_meltw_unary_param xform_param;
        xform_param.in.primary = (void *)((DType *)B + i_n * K * bn + i_k * bk);
        xform_param.out.primary = (void *)((DType *)scratch_B_buf + i_n * K * bn + i_k * bk * bn);
        b_xform_kernel(&xform_param);
      }
    }
  }
  
#pragma omp parallel
  {
    long brcount_use = brcount;
    if (tileconfig_kernel != NULL) tileconfig_kernel(NULL);
    for (int i_k = 0; i_k < Kb_per_layer; i_k += brcount){
#pragma omp for nowait
      for (int i_sfc = 0; i_sfc < Mb * Nb * K_layers; i_sfc++) {
        long brcount_use = brcount;
        libxsmm_gemm_param gemm_param;
        int i_m, i_n, i_k_layer = 0;
        sfc_ca_gemm_extract_indices_from_sfc(&i_m, &i_n, sfc_index_map, i_sfc % (Mb * Nb), index_tsize);
        i_k_layer = i_sfc / (Mb * Nb);
        if (i_k_layer == K_layers - 1) {
          if (i_k + brcount > Kb_last_layer) {
            brcount_use = Kb_last_layer - i_k;
          }
        } else {
          if (i_k + brcount > Kb_per_layer) {
            brcount_use = Kb_per_layer - i_k;
          }
        }
        gemm_param.op.tertiary = (void *)&brcount_use;
        gemm_param.a.primary = (void *)((DType *)A + i_m * K * bm + i_k * bk * bm + i_k_layer * Kb_per_layer * bk * bm);
        if (unblocked_bc == 2) {
          gemm_param.b.primary = (void *)((DType *)config->scratch_B + i_n * K * bn + i_k * bk * bn + i_k_layer * Kb_per_layer * bk * bn);
        } else if (unblocked_bc == 1) {
          gemm_param.b.primary = (void *)((DType *)B + i_n * K * bn + i_k * bk + i_k_layer * Kb_per_layer * bk);
        } else {
          gemm_param.b.primary = (void *)((DType *)B + i_n * K * bn + i_k * bk * bn + i_k_layer * Kb_per_layer * bk * bn);
        }
        if (unblocked_bc > 0) {
          gemm_param.c.primary = (i_k_layer > 0) ? (void *)((CType *)scratch_C[i_k_layer - 1] + i_n * M * bn + i_m * bm)
                                                 : (void *)((CType *)C + i_n * M * bn + i_m * bm);
        } else {
          gemm_param.c.primary = (i_k_layer > 0) ? (void *)((CType *)scratch_C[i_k_layer - 1] + i_n * M * bn + i_m * bn * bm)
                                                 : (void *)((CType *)C + i_n * M * bn + i_m * bn * bm);
        }
        if ((i_k == 0) && (K_rounds_per_layer != 1)){
          libxsmm_meltw_unary_param zero_param;
          zero_param.out.primary = (void*)gemm_param.c.primary;
          zero_kernel( &zero_param );
        }
        if (brcount_use > 0) {
          brgemm_kernel( &gemm_param );
        }
      }
    }
    if (tilerelease_kernel != NULL) tilerelease_kernel(NULL);
  }
  if constexpr (skip_c_reduction == 0) {
    if (K_layers > 1) {
      #pragma omp parallel for
      for (int i_red = 0; i_red < Mb*Nb; i_red++) {
        int i_m, i_n;
        sfc_ca_gemm_extract_indices_from_sfc(&i_m, &i_n, sfc_index_map, i_red, index_tsize);
        if (K_layers == 2) {
          libxsmm_meltw_binary_param add_param;
          if (unblocked_bc > 0) {
            add_param.in0.primary  = (void*)((CType*)scratch_C[0] + i_n * M * bn + i_m * bm );
            add_param.in1.primary  = (void*)((CType*)C + i_n * M * bn + i_m * bm );
            add_param.out.primary = (void*)((CType*)C + i_n * M * bn + i_m * bm );
          } else {
            add_param.in0.primary  = (void*)((CType*)scratch_C[0] + i_n * M * bn + i_m * bn * bm );
            add_param.in1.primary  = (void*)((CType*)C + i_n * M * bn + i_m * bn * bm );
            add_param.out.primary = (void*)((CType*)C + i_n * M * bn + i_m * bn * bm );
          }
          l_add_kernel(&add_param);
        } else {
          libxsmm_meltw_binary_param add_param;
          libxsmm_meltw_unary_param reduce_param;
          if (unblocked_bc > 0) {
            CType reduce_scratch[bm];
            for (long l_in = 0; l_in < bn; l_in++) {
              reduce_param.in.primary = (void*)((CType*)scratch_C[0] + i_n * M * bn + i_m * bm + l_in * M);
              reduce_param.out.primary = (void*)reduce_scratch;
              add_param.in0.primary = (void*)reduce_scratch;
              add_param.in1.primary = (void*)((CType*)C + i_n * M * bn + i_m * bm + l_in * M);
              add_param.out.primary = (void*)((CType*)C + i_n * M * bn + i_m * bm + l_in * M);
              l_reduce_kernel(&reduce_param);
              l_add_kernel(&add_param);
            }
          } else {
            CType reduce_scratch[bm*bn];
            reduce_param.in.primary = (void*)((CType*)scratch_C[0] + i_n * M * bn + i_m * bn * bm );
            reduce_param.out.primary  = (void*)reduce_scratch;
            add_param.in0.primary  = (void*)reduce_scratch;
            add_param.in1.primary  = (void*)((CType*)C + i_n * M * bn + i_m * bn * bm );
            add_param.out.primary = (void*)((CType*)C + i_n * M * bn + i_m * bn * bm );
            l_reduce_kernel(&reduce_param);
            l_add_kernel(&add_param);
          }
        }
      }
    }
  }
  return;
}

template<typename DType>
void run_gemm_n_layers(long n_layers, gemm_config_t *config, DType **A, DType **B, typename output_type<DType>::type **C) {
  for (int i = 0; i < n_layers; i++) {
    run_gemm<DType>(config, A[i], B[i], C[i]);
  }
  return;
}

template <typename DType>
void run_gemm_n_layers_no_reduction(long n_layers, gemm_config_t *config, DType **A, DType **B, typename output_type<DType>::type **C)
{
  for (int i = 0; i < n_layers; i++)
  {
    run_gemm<DType, 1>(config, A[i], B[i], C[i]);
  }
  return;
}

#ifndef SFC_CA_GEMM_NO_MAIN

template<typename DType>
int gemm_benchmark(int argc, char** argv) {
  using CType = typename output_type<DType>::type;
  // Setup default GEMM sizes
  long M = 1024*4, N = 1024*4, K = 1024*4;
  long bm = 32, bn = 32, bk = 32;
  long kbf = 1;
  long K_layers = 1;
  long n_layers = 1;
  long n_iters = 1;
  long i;
  long check_correctness = 0;
  long m_step = 1, n_step = 1;
  long unblocked_bc = 0;
  long use_nts = 0;

  ifreq = 1.0 / getFreq();
  // Read command line arguments
  if (argc > 1) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    bm = atoi(argv[4]);
    while (M % bm != 0) bm--;
    bn = atoi(argv[5]);
    while (N % bn != 0) bn--;
    bk = atoi(argv[6]);
    while (K % bk != 0 || bk % 2 != 0) {
      bk--;
      if (bk <= 0) {
        printf("Error: could not find suitable bk value\n");
        exit(0);
      }
    }
    if (argc > 7) {
      kbf = atoi(argv[7]);
    }
    if (argc > 8){
      K_layers = atoi(argv[8]);
    }
    if (argc > 9) {
      n_layers = atoi(argv[9]);
      if (n_layers == -1) {
        double size_total = (double)1.0 * ((double)sizeof(DType) * (double)M * (double)K + (double)sizeof(CType) * (double)M * (double)N + (double)sizeof(DType) * (double)K * (double)N) / (1024.0 * 1024.0 * 1024.0);
        double low_limit_in_gb = 5.0;
        n_layers = 1;
        while (size_total < low_limit_in_gb) {
          n_layers++;
          size_total = (double)n_layers * ((double)sizeof(DType) * (double)M * (double)K + (double)sizeof(CType) * (double)M * (double)N + (double)sizeof(DType) * (double)K * (double)N) / (1024.0 * 1024.0 * 1024.0);
        }
        printf("Autocalculated %ld layers with total size %.2g GB\n", n_layers, size_total);
      }
    }
    if (argc > 10) {
      n_iters = atoi(argv[10]);
    }
    if (argc > 11) {
      check_correctness = atoi(argv[11]);
    }
    // argv[12] is dtype, parsed in main()
    if (argc > 13) {
      m_step = atoi(argv[13]);
    }
    if (argc > 14) {
      n_step = atoi(argv[14]);
    }
    if (argc > 15) {
      unblocked_bc = atoi(argv[15]);
    }
    if (argc > 16) {
      use_nts = atoi(argv[16]);
    }
  }
  
  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  // Allocate buffers
  DType **B = (DType**) malloc(n_layers*sizeof(DType*));
  check_null_ptr(B, "B array");
  CType **C = (CType**) malloc(n_layers*sizeof(CType*));
  check_null_ptr(C, "C array");
  DType **A = (DType**) malloc(n_layers    *sizeof(DType*));
  check_null_ptr(A, "A array");
  for (i = 0; i < n_layers; i++) {
    A[i] = (DType*) libxsmm_aligned_malloc(M*K*sizeof(DType), ALIGNMENT_SIZE);
    B[i] = (DType*) libxsmm_aligned_malloc(K*N*sizeof(DType), ALIGNMENT_SIZE);
    C[i] = (CType*) libxsmm_aligned_malloc(M*N*sizeof(CType), ALIGNMENT_SIZE);
    check_null_ptr(A[i], "A[i] array");
    check_null_ptr(B[i], "B[i] array");
    check_null_ptr(C[i], "C[i] array");
  }
  
  // Allocate reference buffers
  float *naive_b = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_b, "naive_b array");
  float *naive_c = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_c, "naive_c array");
  float *naive_c_opt = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_c_opt, "naive_c_opt array");
  float *naive_a = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_a, "naive_a array");
  DType *naive_b_lp  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), ALIGNMENT_SIZE);
  check_null_ptr(naive_b_lp, "naive_b_lp array");
  CType *naive_c_lp = (CType*)libxsmm_aligned_malloc( M*N*sizeof(CType), ALIGNMENT_SIZE);
  check_null_ptr(naive_c_lp, "naive_c_lp array");
  DType *naive_a_lp = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), ALIGNMENT_SIZE);
  check_null_ptr(naive_a_lp, "naive_a_lp array");
  
  // Init buffers - for I8, initialize directly; for others, use float conversion
  if (std::is_same<DType, char>::value) {
    // Direct I8 initialization
    init_buf_int((char*)naive_a_lp, M*K, -128, 127);
    init_buf_int((char*)naive_b_lp, K*N, -128, 127);
    init_buf_int((int*)naive_c_lp, M*N, -1000000, 1000000);
    // Convert to float for reference computation
    sfc_ca_gemm_convert_lp_f32<DType>((void*)naive_a_lp, naive_a, M*K);
    sfc_ca_gemm_convert_lp_f32<DType>((void*)naive_b_lp, naive_b, N*K);
    sfc_ca_gemm_convert_lp_f32<CType>((void*)naive_c_lp, naive_c, N*M);
  } else {
    // Original float-based initialization for BF16/FP32
    init_buf( naive_b,    K*N, 0, 0 );
    init_buf( naive_c,    M*N, 0, 0 );
    init_buf( naive_a,    M*K, 0, 0 );
    sfc_ca_gemm_rne_convert_fp32_lp<DType>( naive_b,     (void*)naive_b_lp,     N*K);
    sfc_ca_gemm_convert_lp_f32<DType>( (void*)naive_b_lp, naive_b, N*K);
    sfc_ca_gemm_rne_convert_fp32_lp<CType>( naive_c,    (void*)naive_c_lp,    N*M);
    sfc_ca_gemm_convert_lp_f32<CType>( (void*)naive_c_lp, naive_c, N*M);
    sfc_ca_gemm_rne_convert_fp32_lp<DType>( naive_a,    (void*)naive_a_lp,    M*K);
    sfc_ca_gemm_convert_lp_f32<DType>( (void*)naive_a_lp, naive_a, M*K);
  }
  for (i = 0; i < n_layers; i++) {
    sfc_ca_gemm_matrix_copy_KC_to_KCCK<DType>( (void*)naive_a_lp, (void*)A[i], K, M, bk, bm);
    if (unblocked_bc > 0) {
      memcpy((void *)B[i], (void *)naive_b_lp, N*K*sizeof(DType));
      memcpy((void *)C[i], (void *)naive_c_lp, N*M*sizeof(CType));
    } else {
      sfc_ca_gemm_matrix_copy_NC_to_NCNC<DType>( (void*)naive_b_lp, (void*)B[i] , N, K, bn, bk);
      sfc_ca_gemm_matrix_copy_NC_to_NCNC<CType>( (void*)naive_c_lp, (void*)C[i], N, M, bn, bm);
    }
  }
  
  // Compute reference if requested
  if (check_correctness) {
    naive_gemm_t naive_param;
    naive_param.N = N;
    naive_param.K = K;
    naive_param.M = M;
    naive_param.fuse_type = 0;
    naive_gemm_fp(&naive_param, naive_b, naive_c, naive_a);
    sfc_ca_gemm_rne_convert_fp32_lp<CType>( naive_c,     (void*)naive_c_lp, (N*M) );
    sfc_ca_gemm_convert_lp_f32<CType>( (void*)naive_c_lp, naive_c, N*M);
  } 
  
  // Setup GEMM configuration
  gemm_config_t *gemm_cfg = setup_gemm_config<DType>(M, N, K, bm, bn, bk, kbf, K_layers, m_step, n_step, unblocked_bc, use_nts);

  // Warmup iteration
  run_gemm_n_layers<DType>(n_layers, gemm_cfg, A, B, C);

  // Check correctness if requested
  printf("##############################################################\n");
  printf("    %ld Sets of GEMMS with sizes  %ld x %ld x %ld  (M x N x K)  \n", n_layers, M, N, K);
  printf("##############################################################\n");

  if (check_correctness) {
    libxsmm_matdiff_info norms, diff;
    libxsmm_matdiff_clear(&norms);
    libxsmm_matdiff_clear(&diff);
    if (unblocked_bc > 0) {
      memcpy((void *)naive_c_lp, (void *)C[n_layers-1], N*M*sizeof(CType));
    } else {
      sfc_ca_gemm_matrix_copy_NCNC_to_NC<CType>( (void*)C[n_layers-1], (void*)naive_c_lp, N, M, bn, bm );
    }
    sfc_ca_gemm_convert_lp_f32<CType>( (void*)naive_c_lp, naive_c_opt, N*M );
    printf("##########################################\n");
    printf("#           Correctness                  #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*M, 1, naive_c, naive_c_opt, 0, 0);
    printf("L1 reference  : %.25g\n", norms.l1_ref);
    printf("L1 test       : %.25g\n", norms.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms.l2_rel);
    printf("Linf abs.error: %.24f\n", norms.linf_abs);
    printf("Linf rel.error: %.24f\n", norms.linf_rel);
    printf("Check-norm    : %.24f\n", norms.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms);
  }

  // benchmark the GEMM
  auto t_start = getTime();
  for (long it = 0; it < n_iters; it++) {
    run_gemm_n_layers<DType>(n_layers, gemm_cfg, A, B, C);
  }
  auto t_end = getTime();

  // Print performance/model numbers
  double gflop = (2.0*(double)n_layers*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.7g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("Effective A sizes: %.5g GB\n", ((double)sizeof(DType)*(double)n_layers*(double)M*(double)K)/(1024.0*1024.0*1024.0));
  printf("Effective total GEMM sizes: %.5g GB\n", ((double)n_layers * ((double)sizeof(DType) * (double)M * (double)K + (double)sizeof(CType) * (double)M * (double)N + (double)sizeof(DType) * (double)K * (double)N))/(1024.0*1024.0*1024.0));
  printf("Effective A BW is %.5g GB/s\n", (((double)sizeof(DType)*(double)n_layers*(double)M*(double)K) / (1024.0*1024.0*1024.0))/((t_end-t_start)/(1.0*n_iters)));
  printf("MEASURE %.7g SFC_CA_GEMM_%ld_%ld_%ld_%ld_%ld_%ld_bf%ld_replication_%ld_stepM_%ld_stepN_%ld_act_%ld_nts_%ld_threads%d\n", gflop / ((t_end - t_start) / (1.0 * n_iters)), M, N, K, bm, bn, bk, kbf, K_layers, m_step, n_step, unblocked_bc, use_nts, omp_get_max_threads());

#ifdef PRINT_THREAD_WORK_ASSIGNMENT
  // We run foo loop to capture work assignment thread_work_t
  // Open a file to write the thread work assignment
  FILE *f = fopen("thread_work_assignment.txt", "w");
  thread_work_t *thread_work = (thread_work_t *)malloc(Mb * Nb * K_layers * sizeof(thread_work_t));
  long index_to_store = 0;
#pragma omp parallel for
  for (int i_sfc = 0; i_sfc < Mb * Nb * K_layers; i_sfc++)
  {
    int tid = omp_get_thread_num();
    int i_m, i_n, i_k_layer = 0;
    sfc_ca_gemm_extract_indices_from_sfc(&i_m, &i_n, gemm_cfg->sfc_index_map, i_sfc % (Mb * Nb), gemm_cfg->index_tsize);
    i_k_layer = i_sfc / (Mb * Nb);
#pragma omp critical
    {
      thread_work[index_to_store].thread_id = tid;
      thread_work[index_to_store].m_block = i_m;
      thread_work[index_to_store].n_block = i_n;
      thread_work[index_to_store].k_layer = i_k_layer;
      index_to_store++;
    }
  }
  // Print the info stored in thread_work
  for (long i = 0; i < index_to_store; i++)
  {
    fprintf(f, "Thread %ld, M-block %ld, N-block %ld, K-layer %ld\n",
            thread_work[i].thread_id,
            thread_work[i].m_block,
            thread_work[i].n_block,
            thread_work[i].k_layer);
  }
  fclose(f);
  free(thread_work);
#endif

#if 0
  if (K_layers > 1) {
    // Now run without reduction inside the kernel and time that
    double time_full_gemm = (t_end - t_start) / (1.0 * n_iters);
    double time_reduction = 0.0;
    // Estimate time spent in reduction
    t_start = getTime();
    for (long it = 0; it < n_iters; it++) {
      run_gemm_n_layers_no_reduction<DType>(n_layers, gemm_cfg, A, B, C);
    }
    t_end = getTime();
    printf("Time without reduction is %.5g ms (%.7g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
    printf("MEASURE %.7g SFC_CA_GEMM_NO_REDUCTION_%ld_%ld_%ld_%ld_%ld_%ld_bf%ld_replication_%ld_threads%d\n", gflop / ((t_end - t_start) / (1.0 * n_iters)), M, N, K, bm, bn, bk, kbf, K_layers, omp_get_max_threads());
    time_reduction = time_full_gemm - (t_end - t_start) / (1.0 * n_iters);
    double total_c_reduction_volume = (double)sizeof(CType) * (double)M * (double)N * (double)(n_layers) * (double)(K_layers);
    double reduction_bw = (total_c_reduction_volume / (1024.0 * 1024.0 * 1024.0)) / (time_reduction);
    printf("Estimated reduction time is %.5g ms (%.5g GB/s)\n", 1000.0 * time_reduction, reduction_bw);
  }
#endif

  // Free buffers
  libxsmm_free(naive_b);
  libxsmm_free(naive_c);
  libxsmm_free(naive_c_opt);
  libxsmm_free(naive_a);
  libxsmm_free(naive_b_lp);
  libxsmm_free(naive_c_lp);
  libxsmm_free(naive_a_lp);
  for (i = 0; i < n_layers; i++) {
    libxsmm_free(A[i]);
    libxsmm_free(B[i]);
    libxsmm_free(C[i]);
  }
  // Free config buffers
  if (gemm_cfg->sfc_index_map != NULL) {
    libxsmm_free(gemm_cfg->sfc_index_map);
  }

  if (gemm_cfg->scratch_B != NULL) {
    libxsmm_free(gemm_cfg->scratch_B);
  }

  CType **output_partial_array = (CType**)gemm_cfg->gemm_scratch;
  if (output_partial_array[0] != NULL)
  {
    libxsmm_free(output_partial_array[0]);
  }
  free(B);
  free(C);
  free(A);
  libxsmm_free(gemm_cfg->gemm_scratch);
  delete gemm_cfg;
  return 0;
}

int main(int argc, char** argv) {
  int use_dtype = 1;
  if (argc > 12) {
    if (strcmp(argv[12],"BF16") == 0) {
      use_dtype = 1;
    }
    if (strcmp(argv[12],"BF8") == 0) {
      use_dtype = 2;
    }
    if (strcmp(argv[12],"F32") == 0) {
      use_dtype = 3;
    }
    if (strcmp(argv[12],"I8") == 0) {
      use_dtype = 4;
    }
    if (strcmp(argv[12],"F64") == 0) {
      use_dtype = 5;
    }
  }
  if (use_dtype == 1) {
    return gemm_benchmark<libxsmm_bfloat16>(argc, argv);  
  } else if (use_dtype == 2) {
    return gemm_benchmark<libxsmm_bfloat8>(argc, argv);
  } else if (use_dtype == 3) {
    return gemm_benchmark<float>(argc, argv);
  } else if (use_dtype == 4) {
    return gemm_benchmark<char>(argc, argv);
  } else if (use_dtype == 5) {
    return gemm_benchmark<double>(argc, argv);
  } else {
    return 0;
  }
}

#endif // SFC_CA_GEMM_NO_MAIN