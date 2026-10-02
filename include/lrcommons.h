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
    double *coefs; /** Coefficients β0 through βN */
    double *errors; /** Residuals ε0 through εN */
    size_t n_coefs; /** Number of coefficients (|β|) */
    size_t n_resid; /** Number of samples (|ε|) */
} LRCoefs;

/**
 * @struct Results
 * @brief Results from linear regression and time to perform operations
 */
typedef struct {
    LRCoefs numerical_results;
    double formation_time;
    double solve_time;
    double decomposition_time;
    double total_time; /** Sum of formation, decomposition, and solve */
} Results;

/**
 * @brief A type for the matrices used as parameters in the test functions
 */
typedef const double * matrix;

/**
 * @brief A type specifically to allow some optimizations
 */
typedef const double * restrict matrix_ref;

#endif
