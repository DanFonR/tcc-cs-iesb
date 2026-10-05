/**
 * @file tests.h
 * @brief Declares all functions to be used in main.c
 * @author Dannyel Fontenele Ribeiro
 * 
 * @details Exposes OLS and QR versions of linear regression after compilation
 * 
 */

#ifndef TESTS_H
#define TESTS_H

#include "lrcommons.h"
#include <time.h>

#define TIMEIT(func_call, start, end) \
do { \
    clock_gettime(CLOCK_MONOTONIC_RAW, &start); \
    func_call; \
    clock_gettime(CLOCK_MONOTONIC_RAW, &end); \
} while (0)

typedef struct {
    double formation_time; /** How long it took to prepare before decomp */
    double decomposition_time; /** Decomposition time of X into A and B */
    double solve_time; /** Time to determine coefficients */
    double total_time; /** Sum of formation, decomposition, and solve times */
} TimeResults;

typedef struct {
    char lr_method[3]; /** QR or NE (normal equations) */
    char data_type[5]; /** WELL or ILL dataset */
    char compiler[6]; /** Compiler used (GCC or Clang) */
    char optim[3]; /** Optimization level (O0, O1, O3 with -march=native) */
    size_t n_samples; /** Number of samples used */
    size_t n_features; /** Number of features used, including intercept */
    size_t run; /** The run which resulted in these stats */
} RunInfo;

typedef struct {
    double coef_err_l2; /** Overall magnitude of coefficient error */
    double relative_coef_err_l2; /** Relative coefficient error */
    double max_abs_coef_err; /** Worst absolute coefficient error */
    double min_abs_coef_err; /** Best absolute coefficient error */
    double coef_mae; /** Coefficient Mean Absolute Error */
    double coef_rmse; /** Coefficient Root Mean Square Error */
} CoefData;

typedef struct {
    double resid_err_l2; /** Overall magnitude of residual error */
    double relative_resid_err_l2; /** Relative residual error */
    double max_abs_resid_err; /** Worst absolute residual error */
    double min_abs_resid_err; /** Best absolute residual error */
    double resid_mae; /** Residual Mean Absolute Error */
    double resid_rmse; /** Residual Root Mean Square Error */
} ResidualsData;

/** Row to be written to each CSV */
typedef struct {
    CoefData coef_stats;
    ResidualsData resid_stats;
    RunInfo run_info;
    TimeResults times;
    double condition_number_2; /** Sensitivity to errors in input (κ2(X)) */
} CSVRow;

/* Clang tests */

Results ne_o0_clang(matrix, matrix, size_t, size_t);
Results ne_o1_clang(matrix, matrix, size_t, size_t);
Results ne_native_clang(matrix, matrix, size_t, size_t);

/* GCC Tests */

Results ne_o0_gcc(matrix, matrix, size_t, size_t);
Results ne_o1_gcc(matrix, matrix, size_t, size_t);
Results ne_native_gcc(matrix, matrix, size_t, size_t);

/* Base Householder */

Results hqr_base_gcc(matrix, matrix, size_t, size_t);

double timedelta(struct timespec, struct timespec);
CSVRow run_statistics(
    const RunInfo,
    const Results,
    matrix_ref,
    matrix_ref,
    matrix_ref,
    const size_t,
    const size_t
);


#endif
