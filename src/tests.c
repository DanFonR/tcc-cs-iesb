/**
 * @file tests.c
 * @brief Time related functions
 * @author Dannyel Fontenele Ribeiro
 */

#include "tests.h"
#include "lrcommons.h"
#include <errno.h>
#include <stdlib.h>

/**
 * @brief Returns timedelta between 2 timespecs
 * 
 * @param start Start timespec
 * @param end End timespec
 * 
 */
double timedelta(struct timespec start, struct timespec end) {
    return (
        (double)(end.tv_sec - start.tv_sec) +
        (double)(end.tv_nsec - start.tv_nsec) / 1e9
    );
}

/**
 * @brief Calculates residuals
 * 
 * @param res Results from regression
 * @param X Design matrix (RMO)
 * @param Y Results matrix (RMO)
 * @param true_coefs The vector of true coefficients
 * @param n_samples Number of data samples
 * @param n_features Number of features including intercept
 */
void residuals(
    Results res,
    matrix_ref X,
    matrix_ref Y,
    matrix_ref true_coefs,
    size_t n_samples,
    size_t n_features
) {
    if (X == NULL || Y == NULL || n_samples == 0 || n_features == 0) {
        errno = EINVAL;

        return;
    }

    double *errs = calloc(n_samples, sizeof(double));

    if (errs == NULL) return;

    for (size_t i = 0; i < n_samples; i++) {
        
    }
}
