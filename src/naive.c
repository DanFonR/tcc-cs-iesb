/**
 * @file naive.c
 * @brief Multiple Linear Regression Via Normal Equations
 * @author Dannyel Fontenele Ribeiro
 */

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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
 * @param n_features Number of features per data sample (including intercept)
 */
static void XtX(
    matrix_ref X, double * restrict dest,
    const size_t n_samples, const size_t n_features
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
 * @param n_features Number of features per data sample (including intercept)
 */
static void XtY(
    matrix_ref X, matrix_ref Y,
    double * restrict dest,
    const size_t n_samples, const size_t n_features
) {
    for (size_t i = 0; i < n_features; i++) {
        double val = 0.0;

        for (size_t j = 0; j < n_samples; j++)
            val += X[j * n_features + i] * Y[j];

        dest[i] = val;
    }       
}

/**
 * @brief Computes Cholesky decomposition of XTX
 * 
 * Replaces lower triangle of XTX with L in Cholesky's A = L * L^T
 * 
 * @param XTX Symmetric positive-definite matrix (RMO)
 * @param n_features Number of features per data sample (including intercept)
 */
static void Cholesky(double * restrict XTX, const size_t n_features) {
    for (size_t i = 0; i < n_features; i++) {
        for (size_t j = 0; j <= i; j++) {
            double sum = XTX[i*n_features + j];

            for (size_t k = 0; k < j; k++)
                sum -= XTX[i*n_features + k] * XTX[j*n_features + k];

            if (i == j)
                XTX[i*n_features + j] = sqrt(sum);
            else
                XTX[i*n_features + j] = sum / XTX[j*n_features + j];
        }
    }
}

/**
 * @brief Forward and backward substition of XTY to get betas
 * 
 * Replaces XTY with the predictors using forward (L * z = X^T * y) and
 * backward (L^T * beta = z) substition
 * 
 * @param XTX X^T * X post Cholesky sub, with lower triangle = L (RMO)
 * @param XTY X^T * y (RMO)
 * @param n_features Number of features per data sample (including intercept)
 */
static void predictor_solve(
    matrix_ref XTX,
    double * restrict XTY,
    const size_t n_features
) {
    // Forward: L z = XTY
    for (size_t i = 0; i < n_features; i++) {
        double sum = XTY[i];

        for (size_t k = 0; k < i; k++)
            sum -= XTX[i*n_features + k] * XTY[k];

        XTY[i] = sum / XTX[i*n_features + i];
    }

    // Backward: L^T beta = z
    for (size_t i = n_features; i-- > 0;) {
        double sum = XTY[i];

        for (size_t k = i + 1; k < n_features; k++)
            sum -= XTX[k*n_features + i] * XTY[k];

        XTY[i] = sum / XTX[i*n_features + i];
    }
}

Results ne_linear_regression(
    matrix_ref X, matrix_ref Y,
    size_t n_samples,
    size_t n_features
) {
    if (n_samples == 0 || n_features == 0 || X == NULL || Y == NULL) {
        errno = EINVAL;
        return (Results){0};
    }

    double *XTX = calloc(n_features * n_features, sizeof(double));

    if (XTX == NULL)
        return (Results){0};

    double *XTY = calloc(n_features, sizeof(double));

    if (XTY == NULL) {
        free(XTX);

        return (Results){0};
    }

    double *resid = calloc(n_samples, sizeof(double));

    if (resid == NULL) {
        free(XTX);
        free(XTY);

        return (Results){0};
    }

    struct timespec start, end;
    Results ret = {0};

    TIMEIT(XtX(X, XTX, n_samples, n_features), start, end);

    ret.formation_time = timedelta(start, end);

    TIMEIT(XtY(X, Y, XTY, n_samples, n_features), start, end);

    ret.formation_time += timedelta(start, end);

    TIMEIT(Cholesky(XTX, n_features), start, end);

    ret.decomposition_time = timedelta(start, end);

    TIMEIT(predictor_solve(XTX, XTY, n_features), start, end);

    ret.solve_time = timedelta(start, end);
    
    ret.total_time = (
        ret.formation_time +
        ret.decomposition_time +
        ret.solve_time
    );

    for (size_t i = 0; i < n_samples; i++) {
        double pred = 0.0;

        for (size_t j = 0; j < n_features; j++) 
            pred += X[i * n_features + j] * XTY[j];

        resid[i] = Y[i] - pred;
    }

    ret.numerical_results.coefs = XTY;
    ret.numerical_results.errors = resid;
    ret.numerical_results.n_coefs = n_features;
    ret.numerical_results.n_resid = n_samples;

    free(XTX);

    return ret;
}
