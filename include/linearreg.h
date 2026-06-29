/**
 * @file linearreg.h
 * @brief Provides interface for main function
 */

#ifndef LINEARREG_H
#define LINEARREG_H

#include <stddef.h>
#include "lrcommons.h"

/**
 * @brief Performs Multiple Linear Regression using OLS
 * 
 * This function uses Ordinary Least Squares to perform Multiple Linear
 * Regression.
 * 
 * @param X Matrix of independent variables (Translated into 1D array)
 * @param Y Matrix of dependent variables
 * @param n_samples Number of samples
 * @param n_features Number of features (in matrix X)
 * 
 * @return Coefficients, residuals, and time results
 */
Results ols_linear_regression(
    matrix_ref X, matrix_ref Y,
    size_t n_samples,
    size_t n_features
);

/**
 * @brief Performs Multiple Linear Regression using Householder QR Decomposition
 * 
 * @param X Matrix of independent variables (Translated into 1D array)
 * @param Y Matrix of dependent variables
 * @param n_samples Number of samples
 * @param n_features Number of features (in matrix X)
 * 
 * @return Coefficients, residuals, and time results
 */
Results hqr_linear_regression(
    matrix_ref X, matrix_ref Y,
    size_t n_samples,
    size_t n_features
);

#endif
