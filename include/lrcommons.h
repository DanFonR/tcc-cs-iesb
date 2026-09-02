/**
 * @file lrcommons.h
 * @brief Typedefs used by other headers
 * @author Dannyel Fontenele Ribeiro
 */

#ifndef LRCOMMONS_H
#define LRCOMMONS_H

#include <stddef.h>

/**
 * @struct LRCoefs
 * @brief Coefficients and residuals from a linear regression operation
 */
typedef struct {
    double *coefs; /**< Coefficients β0 through βN */
    double *errors; /**< Residuals ε0 through εN */
    size_t n_coefs; /**< Number of coefficients (|ε| == |β| == n_coefs) */
} LRCoefs;

/**
 * @struct Results
 * @brief Results from linear regression and time to perform operations
 */
typedef struct {
    LRCoefs numerical_results;
    double compute_time;
    double matrix_build_time;
    double total_time;
} Results;

/**
 * @brief A type for the matrices used as parameters in the test functions
 */
typedef const double * matrix;

/**
 * @brief A type specifically to force some optimizations
 */
typedef const double * restrict matrix_ref;

#endif
