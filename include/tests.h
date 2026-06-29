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

/* Clang tests */

Results ols_o0_clang(matrix, matrix, size_t, size_t);
Results ols_o2_clang(matrix, matrix, size_t, size_t);
Results ols_native_clang(matrix, matrix, size_t, size_t);
Results ols_parallel_clang(matrix, matrix, size_t, size_t);

/* GCC Tests */

Results ols_o0_gcc(matrix, matrix, size_t, size_t);
Results ols_o2_gcc(matrix, matrix, size_t, size_t);
Results ols_native_gcc(matrix, matrix, size_t, size_t);
Results ols_parallel_gcc(matrix, matrix, size_t, size_t);

/* Base Householder */

Results hqr_base_gcc(matrix, matrix, size_t, size_t);
Results hqr_native_gcc(matrix, matrix, size_t, size_t);

#endif
