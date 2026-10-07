/**
 * @file tests.c
 * @brief Time related functions
 * @author Dannyel Fontenele Ribeiro
 */

#include "tests.h"
#include "lrcommons.h"
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <lapacke.h>

static double condition_number_2(
    const double *X,
    size_t n_samples,
    size_t n_features
) {
    double *a = malloc(n_samples * n_features * sizeof(double));
    double *s = malloc(n_features * sizeof(double));

    if (a == NULL || s == NULL) {
        free(a);
        free(s);
        return NAN;
    }

    memcpy(
        a,
        X,
        n_samples * n_features * sizeof(double)
    );

    /* Single Value Decomposition, to determine condition number */
    /* A = UΣV^T, only need Σ, with max = s[0] and min = s[n_features - 1] */
    lapack_int info = LAPACKE_dgesdd(
        LAPACK_ROW_MAJOR,
        'N', /* Don't calculate U or VT */
        (lapack_int)n_samples,
        (lapack_int)n_features,
        a, /* input (X) */
        (lapack_int)n_features,
        s, /* output SVD */
        NULL, /* Output left singular vectors */
        1, /* Leading dimension of U (not computed) */
        NULL, /* Output right singular vectors */
        1 /* Leading dimension of VT (not computed) */
    );

    double condition = NAN;

    if (info == 0 && s[n_features - 1] != 0.0)
        condition = s[0] / s[n_features - 1];

    free(a);
    free(s);

    return condition;
}

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
 * @brief Outputs data ready to be written as a CSV row
 * 
 * @param res Results from regression
 * @param X Design matrix (RMO)
 * @param Y Results matrix (RMO)
 * @param true_coefs The vector of true coefficients
 * @param n_samples Number of data samples
 * @param n_features Number of features including intercept
 */
CSVRow run_statistics(
    const RunInfo metadata,
    const Results res,
    matrix_ref X,
    matrix_ref Y,
    matrix_ref true_coefs,
    const size_t n_samples,
    const size_t n_features
) {
    if (X == NULL || Y == NULL || true_coefs == NULL ||
        n_samples == 0 || n_features == 0 || n_samples < n_features ||
        res.numerical_results.n_coefs != n_features ||
        res.numerical_results.n_resid != n_samples ||
        metadata.n_features != n_features || metadata.n_samples != n_samples
    ) {
        errno = EINVAL;

        return (CSVRow){0};
    }

    CSVRow ret = {0};
    ret.resid_stats.min_abs_resid_err = INFINITY;
    ret.coef_stats.min_abs_coef_err = INFINITY;

    double coef_l2 = 0.0, resid_l2 = 0.0;

    ret.run_info = metadata;
    ret.times.decomposition_time = res.decomposition_time;
    ret.times.formation_time = res.formation_time;
    ret.times.solve_time = res.solve_time;
    ret.times.total_time = res.total_time;

    ret.condition_number_2 = condition_number_2(X, n_samples, n_features);

    for (size_t i = 0; i < res.numerical_results.n_resid; i++) {
        double resid = fabs(res.numerical_results.errors[i]);

        if (ret.resid_stats.max_abs_resid_err < resid)
            ret.resid_stats.max_abs_resid_err = resid;
        if (ret.resid_stats.min_abs_resid_err > resid)
            ret.resid_stats.min_abs_resid_err = resid;

        ret.resid_stats.resid_err_l2 += resid * resid;
        ret.resid_stats.resid_mae += resid;

        resid_l2 += Y[i] * Y[i];
    }

    ret.resid_stats.resid_err_l2 = sqrt(ret.resid_stats.resid_err_l2);
    ret.resid_stats.relative_resid_err_l2 = (resid_l2 != 0.0)? (
        ret.resid_stats.resid_err_l2 / sqrt(resid_l2)
    ) : INFINITY;
    ret.resid_stats.resid_mae /= n_samples;
    ret.resid_stats.resid_rmse = (
        ret.resid_stats.resid_err_l2 / sqrt(n_samples)
    );

    for (size_t i = 0; i < res.numerical_results.n_coefs; i++) {
        double coef_diff = fabs(
            true_coefs[i] - res.numerical_results.coefs[i]
        );

        if (ret.coef_stats.max_abs_coef_err < coef_diff)
            ret.coef_stats.max_abs_coef_err = coef_diff;
        if (ret.coef_stats.min_abs_coef_err > coef_diff)
            ret.coef_stats.min_abs_coef_err = coef_diff;

        ret.coef_stats.coef_err_l2 += coef_diff * coef_diff;
        ret.coef_stats.coef_mae += coef_diff;

        coef_l2 += true_coefs[i] * true_coefs[i];
    }

    ret.coef_stats.coef_err_l2 = sqrt(ret.coef_stats.coef_err_l2);
    ret.coef_stats.relative_coef_err_l2 = (coef_l2 != 0.0)? (
        ret.coef_stats.coef_err_l2 / sqrt(coef_l2)
    ) : INFINITY;
    ret.coef_stats.coef_mae /= n_features;
    ret.coef_stats.coef_rmse = (
        ret.coef_stats.coef_err_l2 / sqrt(n_features)
    );

    if (isnan(ret.coef_stats.coef_err_l2)) {
        ret.coef_stats.max_abs_coef_err = NAN;
        ret.coef_stats.min_abs_coef_err = NAN;
    }
    if (isnan(ret.resid_stats.resid_err_l2)) {
        ret.resid_stats.max_abs_resid_err = NAN;
        ret.resid_stats.min_abs_resid_err = NAN;
    }

    return ret;
}
