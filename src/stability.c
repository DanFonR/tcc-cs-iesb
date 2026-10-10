/**
 * @file stability.c
 * @brief Numerical stability experiment: normal equations vs. Householder QR
 *
 * For each stability dataset (see datawriter.c) the least squares solution is
 * computed in long double (ref_lstsq_ld) and every method is run once; the
 * errors written to the CSV are relative to that reference solution, not to
 * the true betas, so they measure arithmetic error only (the statistical
 * error of the sample cancels out). The CSV has the same columns as the
 * timing experiment; condition_num_2 holds kappa_2(X).
 *
 * A failed Cholesky factorization of X^T X (normal equations breakdown for
 * kappa^2 * u >~ 1) shows up as NaN in the error columns.
 *
 * @author Dannyel Fontenele Ribeiro
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <lapacke.h>
#include "tests.h"
#include "refsolve.h"

typedef Results (*regressor)(matrix_ref, matrix_ref, size_t, size_t);

/** Exponents e of COL_EPS = 10^-e (0 = well-conditioned), as in datawriter.c */
static const int EXPS[] = {0, 3, 5, 6, 7};
/** Number of features including intercept, as in datawriter.c */
static const size_t FEATURES[] = {11, 51, 101};

/**
 * @brief Loads a dataset file (layout: n_samples, n_features, beta, y, X)
 *
 * @param path File path
 * @param ns Output number of samples read from the header
 * @param nf Output number of features read from the header
 * @param X Output design matrix (RMO), caller frees
 * @param y Output responses, caller frees
 * @param betas Output true coefficients, caller frees
 * @return 0 on success, -1 on failure (outputs are NULL)
 */
static int load_dataset(
    const char *path, size_t *ns, size_t *nf,
    double **X, double **y, double **betas
) {
    FILE *fp = fopen(path, "rb");

    *X = *y = *betas = NULL;

    if (fp == NULL) { perror(path); return -1; }

    if (fread(ns, sizeof *ns, 1, fp) != 1 ||
        fread(nf, sizeof *nf, 1, fp) != 1 || *ns < *nf || *nf == 0) {
        fprintf(stderr, "%s: Invalid header\n", path);
        goto fail;
    }

    *X = malloc(*ns * *nf * sizeof(double));
    *y = malloc(*ns * sizeof(double));
    *betas = malloc(*nf * sizeof(double));

    if (!*X || !*y || !*betas) {
        fprintf(stderr, "%s: Out of memory\n", path);
        goto fail;
    }

    if (fread(*betas, sizeof(double), *nf, fp) != *nf ||
        fread(*y, sizeof(double), *ns, fp) != *ns ||
        fread(*X, sizeof(double), *ns * *nf, fp) != *ns * *nf) {
        fprintf(stderr, "%s: Incomplete read\n", path);
        goto fail;
    }

    fclose(fp);
    return 0;

fail:
    free(*X); free(*y); free(*betas);
    *X = *y = *betas = NULL;
    fclose(fp);
    return -1;
}

/**
 * @brief Writes one CSV row (same columns as the timing experiment)
 */
static void write_row(FILE *csv, const CSVRow *m) {
    fprintf(csv,
        "%zu,%s,%s,%s,%s,%zu,%zu,"
        "%.17g,%.9f,%.9f,%.9f,%.9f,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
        m->run_info.run, m->run_info.lr_method, m->run_info.compiler,
        m->run_info.optim, m->run_info.data_type,
        m->run_info.n_samples, m->run_info.n_features,
        m->condition_number_2, m->times.formation_time,
        m->times.decomposition_time, m->times.solve_time,
        m->times.total_time,
        m->coef_stats.coef_err_l2, m->coef_stats.relative_coef_err_l2,
        m->coef_stats.max_abs_coef_err, m->coef_stats.min_abs_coef_err,
        m->coef_stats.coef_mae, m->coef_stats.coef_rmse,
        m->resid_stats.resid_err_l2, m->resid_stats.relative_resid_err_l2,
        m->resid_stats.max_abs_resid_err, m->resid_stats.min_abs_resid_err,
        m->resid_stats.resid_mae, m->resid_stats.resid_rmse
    );
}

/**
 * @brief Entry point: runs the stability experiment, writes
 * data/results/stability.csv
 */
int main(void) {
    openblas_set_num_threads(1);
    LAPACKE_set_nancheck(0);

    time_t now = time(NULL);
    struct tm local_now = *localtime(&now);

    char filename[64];

    strftime(filename, sizeof(filename), "data/results/%Y%m%d-%H%M%S-stability.csv", &local_now);

    FILE *csv = fopen(filename, "w");

    if (csv == NULL) { perror("data/results/stability.csv"); return 1; }

    fputs(
        "run,lr_method,compiler,optim,data_type,"
        "n_samples,n_features,condition_num_2,"
        "formation_time,decomposition_time,solve_time,total_time,"
        "coef_err_l2,relative_coef_err_l2,max_abs_coef_err,"
        "min_abs_coef_err,coef_mae,coef_rmse,"
        "resid_err_l2,relative_resid_err_l2,max_abs_resid_err,"
        "min_abs_resid_err,resid_mae,resid_rmse\n",
        csv
    );

    const regressor funcs[] = {
        ne_o0_clang, ne_o0_gcc, ne_o1_clang, ne_o1_gcc,
        ne_native_clang, ne_native_gcc, hqr_base_gcc
    };
    const size_t funcs_len = sizeof funcs / sizeof *funcs;
    const char *compilers[] = {"clang", "gcc"};
    const char *options[] = {"O0", "O1", "native"};

    for (size_t e = 0; e < sizeof EXPS / sizeof *EXPS; e++) {
        for (size_t f = 0; f < sizeof FEATURES / sizeof *FEATURES; f++) {
            char path[96], label[24];
            size_t ns, nf;
            double *X, *y, *betas;

            snprintf(
                path, sizeof path, "data/stability/stab_e%d_p%zu.bin",
                EXPS[e], FEATURES[f]
            );
            snprintf(label, sizeof label, "stab_e%d", EXPS[e]);

            if (load_dataset(path, &ns, &nf, &X, &y, &betas) != 0) {
                fclose(csv);
                return 1;
            }

            double *ref = malloc(nf * sizeof(double));

            if (ref == NULL || ref_lstsq_ld(X, y, ns, nf, ref) != 0) {
                fprintf(stderr, "%s: reference solution failed\n", path);
                free(ref); free(X); free(y); free(betas);
                fclose(csv);
                return 1;
            }

            for (size_t m = 0; m < funcs_len; m++) {
                const int is_qr = (m == funcs_len - 1);
                Results res = funcs[m](X, y, ns, nf);

                if (res.numerical_results.coefs == NULL) {
                    fprintf(stderr, "%s: method %zu failed\n", path, m);
                    free(res.numerical_results.errors);
                    continue;
                }

                const RunInfo info = {
                    .compiler = is_qr? "gcc" : compilers[m % 2],
                    .data_type = label,
                    .lr_method = is_qr? "QR" : "NE",
                    .n_features = nf,
                    .n_samples = ns,
                    .optim = is_qr? "O0" : options[m / 2],
                    .run = 0
                };

                /* ref plays the role of the "true" coefficients */
                CSVRow row = run_statistics(info, res, X, y, ref, ns, nf);

                write_row(csv, &row);

                free(res.numerical_results.coefs);
                free(res.numerical_results.errors);
            }

            fflush(csv);
            free(ref); free(X); free(y); free(betas);
        }
    }

    fclose(csv);

    return 0;
}
