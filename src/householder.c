/**
 * @file householder.c
 * @brief Multiple Linear Regression via Householder QR Decomposition
 * @author Dannyel Fontenele Ribeiro
 */

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <openblas/lapacke.h>
#include "linearreg.h"
#include "tests.h"

Results hqr_linear_regression(
    matrix_ref X, matrix_ref Y,
    size_t n_samples,
    size_t n_features
) {
    if (n_samples == 0 || n_features == 0) {
        errno = EINVAL;
        return (Results){0};
    }

    return (Results){0}; // WIP
    // double x = X[i * n_features + j];
}
