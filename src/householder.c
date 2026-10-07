/**
 * @file householder.c
 * @brief Multiple Linear Regression via Householder QR Decomposition
 * @author Dannyel Fontenele Ribeiro
 */

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <lapacke.h>
#include "linearreg.h"
#include "tests.h"

#define FAIL(in, out, tau, resid) \
do { \
    free(in); \
    free(out); \
    free(tau); \
    free(resid); \
    return (Results){0}; \
} while(0)

/**
 * @brief Performs Linear Regression using Householder QR decomposition
 * and other OpenBLAS LAPACK operations
 * 
 * @param X Design Matrix (RMO)
 * @param Y Results Matrix (RMO)
 * @param n_samples Number of data samples
 * @param n_features Number of features including intercept
 */
Results lapack_linear_regression(
    matrix_ref X, matrix_ref Y,
    size_t n_samples,
    size_t n_features
) {
    if (n_samples == 0 || n_features == 0 || X == NULL || Y == NULL ||
        n_samples < n_features) {
        errno = EINVAL;
        return (Results){0};
    }

    double *in = malloc(n_features * n_samples * sizeof(double));
    double *out = malloc(n_samples * sizeof(double));
    double *tau = malloc(n_features * sizeof(double));
    double *resid = calloc(n_samples, sizeof(double));

    if (in == NULL || out == NULL || tau == NULL || resid == NULL)
        FAIL(in, out, tau, resid);

    memcpy(in, X, n_samples * n_features * sizeof(double));
    memcpy(out, Y, n_samples * sizeof(double));

    struct timespec start, end;
    lapack_int info;
    Results ret = {0}; /* Formation time = 0, XTX and XTY are not formed */

    TIMEIT(
        /* X = Q * R */
        info = LAPACKE_dgeqrf(
            LAPACK_ROW_MAJOR, /* Orientation */
            n_samples, /* Number of rows */
            n_features, /* Number of columns */
            in, /* Matrix to factor (a copy of X) */
            n_features, /* leading dimension (stride) */
            tau /* scalar factor of Householder reflectors */
        ),
        start,
        end
    );

    if (info != 0) FAIL(in, out, tau, resid);

    ret.decomposition_time = timedelta(start, end);

    TIMEIT(
        /* Y <- Q^T * Y */
        info = LAPACKE_dormqr(
            LAPACK_ROW_MAJOR,
            'L', /* Apply Q^T from the left */
            'T', /* Transpose Q */
            n_samples, /* Number of rows in Y */
            1, /* Number of columns in Y */
            n_features, /* Number of Householder reflectors */
            in, /* Representation from dgeqrf */
            n_features, /* Leading dimension */
            tau, /* Scalar factors from dgeqrf */
            out, /* Out vector */
            1 /* Leading dimension of out */
        ),
        start,
        end
    );

    if (info != 0) FAIL(in, out, tau, resid);

    ret.solve_time = timedelta(start, end);

    TIMEIT(
        /* R * beta = Q^T * Y */
        info = LAPACKE_dtrtrs(
            LAPACK_ROW_MAJOR,
            'U', /* Matrix has upper-triangular system */
            'N', /* Don't transpose R */
            'N', /* Diagonal of R must be used (non-unit diagonal) */
            n_features, /* Order of R (n_features x n_features) */
            1, /* Number of right-hand sides (just one out vector) */
            in, /* Contains R in upper triangle */
            n_features, /* Leading dimension of R */
            out, /* Q^T * Y, overwritten with beta */
            1 /* Leading dimension of Q^T * Y */
        ),
        start,
        end
    );

    if (info != 0) FAIL(in, out, tau, resid);

    ret.solve_time += timedelta(start, end);

    ret.total_time = (
        ret.formation_time +
        ret.decomposition_time +
        ret.solve_time
    );

    for (size_t i = 0; i < n_samples; i++) {
        double pred = 0.0;

        for (size_t j = 0; j < n_features; j++) 
            pred += X[i * n_features + j] * out[j];

        resid[i] = Y[i] - pred;
    }

    double *coefs = malloc(n_features * sizeof(*coefs));

    if (coefs == NULL) FAIL(in, out, tau, resid);

    memcpy(coefs, out, n_features * sizeof(*coefs));

    ret.numerical_results.coefs = coefs;
    ret.numerical_results.errors = resid;
    ret.numerical_results.n_coefs = n_features;
    ret.numerical_results.n_resid = n_samples;

    free(in);
    free(tau);
    free(out);

    return ret;
}
