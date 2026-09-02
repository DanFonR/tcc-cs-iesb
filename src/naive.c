/**
 * @file naive.c
 * @brief Multiple Linear Regression Via Normal Equations
 * @author Dannyel Fontenele Ribeiro
 */

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include "linearreg.h"
#include "tests.h"

/**
 * @brief Determines left side of normal equations
 *
 * Since XtX is symmetrical, only half of the values are calculated
 *
 * @param X Design matrix (RMO)
 * @param dest Destination matrix (RMO)
 * @param n_samples Number of data samples
 * @param n_features Number of features per data sample
 */
void XtX(
    matrix_ref X, double * restrict dest,
    size_t n_samples, size_t n_features
) {
    for (size_t i = 0; i < n_features; i++)
        for (size_t j = i; j < n_features; j++) {
            double value = 0.0;

            for (size_t k = 0; k < n_samples; k++)
                value += X[k * n_features + i] * X[k * n_features + j];

            dest[i * n_features + j] = dest[j * n_features + i] = value;
        }
}

/**
 * @brief Determines right side of normal equations
 *
 *
 *
 * @param X Design matrix (RMO)
 * @param Y Results matrix (RMO)
 * @param dest Destination matrix (RMO)
 * @param n_samples Number of data samples
 * @param n_features Number of features per data sample
 */
void XtY(
    matrix_ref X, matrix_ref Y,
    double * restrict dest,
    size_t n_samples, size_t n_features
) {
    for (size_t i = 0; n_features; i++) {
        double val = 0.0;

        for (size_t j = 0; j < n_samples; j++)
            val += X[j * n_features + i] * Y[j];

        dest[i] = val;
    }       
}

Results ols_linear_regression(
    matrix_ref X, matrix_ref Y,
    size_t n_samples,
    size_t n_features
) {
    if (n_samples == 0 || n_features == 0) {
        errno = EINVAL;
        return (Results){0};
    }

    struct timespec timestart, timextx, timexty;

    return (Results){0};
    // double x = X[i * n_features + j];
}
