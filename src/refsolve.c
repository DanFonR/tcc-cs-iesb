/**
 * @file refsolve.c
 * @brief High-precision reference solver for linear least squares
 * @author Dannyel Fontenele Ribeiro
 */

#include <math.h>
#include <stdlib.h>
#include "refsolve.h"

int ref_lstsq_ld(
    const double *X, const double *y,
    size_t n_samples, size_t n_features,
    double *beta
) {
    const size_t n = n_samples, p = n_features;

    if (X == NULL || y == NULL || beta == NULL || p == 0 || n < p)
        return -1;

    /* Column-major long double copy: column ops become contiguous */
    long double *A = malloc(n * p * sizeof(long double));
    long double *b = malloc(n * sizeof(long double));
    long double *v = malloc(n * sizeof(long double));
    long double *x = malloc(p * sizeof(long double));

    if (!A || !b || !v || !x) {
        free(A); free(b); free(v); free(x);
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        b[i] = y[i];

        for (size_t c = 0; c < p; c++)
            A[c * n + i] = X[i * p + c];
    }

    int status = 0;

    for (size_t j = 0; j < p && status == 0; j++) {
        long double nrm2 = 0.0L;

        for (size_t i = j; i < n; i++)
            nrm2 += A[j * n + i] * A[j * n + i];

        const long double nrm = sqrtl(nrm2);

        if (nrm == 0.0L) { status = -1; break; }

        /* H = I - 2 v v^T / (v^T v), mapping column j to alpha * e_j */
        const long double ajj = A[j * n + j];
        const long double alpha = (ajj > 0.0L)? -nrm : nrm;

        v[0] = ajj - alpha;
        long double vtv = v[0] * v[0];

        for (size_t i = j + 1; i < n; i++) {
            v[i - j] = A[j * n + i];
            vtv += v[i - j] * v[i - j];
        }

        if (vtv == 0.0L) continue; /* column already in the right form */

        const long double tau = 2.0L / vtv;

        for (size_t c = j; c < p; c++) {
            long double dot = 0.0L;

            for (size_t i = j; i < n; i++)
                dot += v[i - j] * A[c * n + i];

            dot *= tau;

            for (size_t i = j; i < n; i++)
                A[c * n + i] -= dot * v[i - j];
        }

        long double dot = 0.0L;

        for (size_t i = j; i < n; i++)
            dot += v[i - j] * b[i];

        dot *= tau;

        for (size_t i = j; i < n; i++)
            b[i] -= dot * v[i - j];
    }

    if (status == 0) {
        /* R * x = (Q^T y)[0..p), back substitution */
        for (size_t i = p; i-- > 0;) {
            long double s = b[i];

            for (size_t k = i + 1; k < p; k++)
                s -= A[k * n + i] * x[k];

            if (A[i * n + i] == 0.0L) { status = -1; break; }

            x[i] = s / A[i * n + i];
        }
    }

    if (status == 0)
        for (size_t i = 0; i < p; i++)
            beta[i] = (double)x[i];

    free(A); free(b); free(v); free(x);

    return status;
}
