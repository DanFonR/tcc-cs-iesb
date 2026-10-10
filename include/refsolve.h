/**
 * @file refsolve.h
 * @brief High-precision reference solver for linear least squares
 * @author Dannyel Fontenele Ribeiro
 */

#ifndef REFSOLVE_H
#define REFSOLVE_H

#include <stddef.h>

/**
 * @brief Solves min ||X * beta - y||_2 with Householder QR in long double
 *
 * Used as the "exact" solution against which the double-precision methods
 * are compared (stability experiment). With x87 extended precision
 * (unit roundoff 5.4e-20, x86-64 Linux) the reference is ~2000 times more
 * precise than double. The relative forward error of Householder QR is about
 * u * (kappa + kappa^2 * ||r|| / (||X|| * ||beta||)), so the reference is
 * only meaningful while that bound stays well below the errors being
 * measured in double precision.
 *
 * Memory: 16 bytes per entry of X (long double copy), so it is meant for
 * moderate sizes (n_samples up to ~1e5 with 100 features).
 *
 * @param X Design matrix (RMO), n_samples x n_features
 * @param y Responses, n_samples entries
 * @param n_samples Number of data samples (>= n_features)
 * @param n_features Number of features including intercept
 * @param beta Output coefficients (n_features entries)
 * @return 0 on success, -1 on invalid input, rank deficiency or allocation failure
 */
int ref_lstsq_ld(
    const double *X, const double *y,
    size_t n_samples, size_t n_features,
    double *beta
);

#endif
