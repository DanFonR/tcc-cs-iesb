/**
 * @file householder.c
 * @brief Multiple Linear Regression via Householder QR Decomposition
 * @author Dannyel Fontenele Ribeiro
 */

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <openblas/lapacke.h>
#include "linearreg.h"
#include "tests.h"

#define FAIL(in, out, tau) \
do { \
    free(out); \
    free(in); \
    free(tau); \
    return (Results){0}; \
} while(0)

/**
 * @brief Performs Linear Regression using Householder QR decompositions
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
    if (n_samples == 0 || n_features == 0 || X == NULL || Y == NULL) {
        errno = EINVAL;
        return (Results){0};
    }

    double *in = malloc(n_features * n_samples * sizeof(double));

    if (in == NULL) return (Results){0};

    double *tau = malloc(n_features * sizeof(double));

    if (tau == NULL) {
        free(in);

        return (Results){0};
    }

    double *out = malloc(n_samples * sizeof(double));

    if (out == NULL) {
        free(tau);
        free(in);

        return (Results){0};
    }

    memcpy(in, X, n_samples * n_features * sizeof(double));
    memcpy(out, Y, n_samples * sizeof(double));

    struct timespec start, end;
    lapack_int info;
    Results ret = {0};

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

    if (info != 0) FAIL(in, out, tau);

    ret.decomposition_time = timedelta(start, end);

    TIMEIT(
        /* Y <- Q^T * Y */
        info = LAPACKE_dormqr(
            LAPACK_ROW_MAJOR,
            'L', /* Apply Q from the left (Q * Y) */
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

    if (info != 0) FAIL(in, out, tau);
    
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

    if (info != 0) FAIL(in, out, tau);
    
    ret.solve_time += timedelta(start, end);
    
    ret.total_time = (
        ret.formation_time +
        ret.decomposition_time +
        ret.solve_time
    );

    ret.numerical_results.coefs = out;
    ret.numerical_results.n_coefs = n_features;
    ret.numerical_results.n_resid = n_samples;

    free(tau);
    free(in);

    return ret;
}
