/**
 * @file main.c
 * @brief Performs tests of various compilations options
 * @author Dannyel Fontenele Ribeiro
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include "tests.h"

typedef Results (*regressor)(matrix_ref, matrix_ref, size_t, size_t);

static volatile double warmup_sink;

static inline void warmup_consume(const Results *r) {
    if (r == NULL || r->numerical_results.coefs == NULL) return;

    warmup_sink += r->numerical_results.coefs[0];
}

static int load_dataset(
    const char *path, size_t n_features,
    double **X, double **y, double **betas
) {
    const size_t n_samples = 1000000;

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) { perror(path); return -1; }

    size_t ns = 0, nf = 0;
    *X = NULL; *y = NULL; *betas = NULL;

    if (fread(&ns, sizeof ns, 1, fp) != 1 ||
        fread(&nf, sizeof nf, 1, fp) != 1 ||
        ns != n_samples || nf != n_features
    ) {
        fprintf(stderr, "%s: Invalid header\n", path);
        goto fail;
    }

    *X = malloc(n_samples * n_features * sizeof(double));
    *y = malloc(n_samples * sizeof(double));
    *betas = malloc(n_features * sizeof(double));

    if (!*X || !*y || !*betas) {
        fprintf(stderr, "%s: Out of memory\n", path);
        goto fail;
    }

    if (fread(*betas, sizeof(double), n_features, fp) != n_features ||
        fread(*y, sizeof(double), n_samples, fp) != n_samples ||
        fread(*X, sizeof(double), n_samples * n_features, fp)
            != n_samples * n_features
    ) {
        fprintf(stderr, "%s: Incomplete read\n", path);
        goto fail;
    }

    fclose(fp);
    return 0;

    fail:
    free(*X);
    free(*y);
    free(*betas);

    *X = *y = *betas = NULL;

    fclose(fp);

    return -1;
}

void benchmark(
    matrix_ref X, matrix_ref y, matrix_ref betas,
    const char *type, size_t n_samples, size_t n_features,
    FILE *csv
) {
    const size_t warmups = 3, runs = 20;

    regressor funcs[] = {
        ne_o0_clang, ne_o0_gcc,
        ne_o1_clang, ne_o1_gcc,
        ne_native_clang, ne_native_gcc,
        NULL, hqr_base_gcc
    };
    size_t funcs_len = sizeof(funcs)/sizeof(*funcs);

    const char *compilers[] = {
        "clang", "gcc"
    };

    const char *options[] = {"O0", "O1", "native"};

    for (size_t f = 0; f < funcs_len; f++) {
        if (funcs[f] == NULL) continue;

        for (size_t i = 0; i < warmups; i++) {
            Results tmp = (funcs[f])(X, y, n_samples, n_features);

            warmup_consume(&tmp);

            free(tmp.numerical_results.coefs);
            free(tmp.numerical_results.errors);
        }

        for (size_t i = 0; i < runs; i++) {
            Results act = (funcs[f])(X, y, n_samples, n_features);

            if (act.numerical_results.coefs == NULL) {
                fprintf(stderr, "error on run %zu\n", i);
                free(act.numerical_results.errors);

                continue;
            }

            RunInfo metadata = {
                .compiler = compilers[f % 2],
                .data_type = type,
                .lr_method = (f == funcs_len - 1)? "QR" : "NE",
                .n_features = n_features,
                .n_samples = n_samples,
                .optim = (f == funcs_len - 1)? "O0" : options[f / 2],
                .run = i
            };

            CSVRow metrics = run_statistics(
                metadata, act, X, y, betas, n_samples, n_features
            );

            free(act.numerical_results.coefs);
            free(act.numerical_results.errors);

            fprintf(csv,
                "%zu,%s,%s,%s,"
                "%s,%zu,%zu,"
                "%.17g,%.9f,"
                "%.9f,%.9f,"
                "%.9f,%.17g,"
                "%.17g,"
                "%.17g,"
                "%.17g,"
                "%.17g,%.17g,"
                "%.17g,"
                "%.17g,"
                "%.17g,"
                "%.17g,"
                "%.17g,%.17g\n",
                i, metadata.lr_method, metadata.compiler, metadata.optim,
                metadata.data_type, n_samples, n_features,
                metrics.condition_number_2, metrics.times.formation_time,
                metrics.times.decomposition_time, metrics.times.solve_time,
                metrics.times.total_time, metrics.coef_stats.coef_err_l2,
                metrics.coef_stats.relative_coef_err_l2,
                metrics.coef_stats.max_abs_coef_err,
                metrics.coef_stats.min_abs_coef_err,
                metrics.coef_stats.coef_mae, metrics.coef_stats.coef_rmse,
                metrics.resid_stats.resid_err_l2,
                metrics.resid_stats.relative_resid_err_l2,
                metrics.resid_stats.max_abs_resid_err,
                metrics.resid_stats.min_abs_resid_err,
                metrics.resid_stats.resid_mae, metrics.resid_stats.resid_rmse
            );
        }
    }
}


int main() {
    openblas_set_num_threads(1);
    LAPACKE_set_nancheck(0);

    FILE *csv = fopen("data/results/results.csv", "w");

    if (csv == NULL) return 1;

    fputs(
        "run,lr_method,compiler,optim,data_type," /* Strings */
        "n_samples,n_features,condition_num_2," /* Numerical data */
        "formation_time,decomposition_time,solve_time,total_time," /* Times */
        "coef_err_l2,relative_coef_err_l2,max_abs_coef_err," /* Coefficients */
        "min_abs_coef_err,coef_mae,coef_rmse,"
        "resid_err_l2,relative_resid_err_l2,max_abs_resid_err," /* Residuals */
        "min_abs_resid_err,resid_mae,resid_rmse\n",
        csv
    );

    for (int kind = 0; kind < 2; kind++) {
        const char *type = (kind == 0)? "well" : "ill";

        for (size_t features = 11; features <= 101; features += 10) {
            char filename[64];
            snprintf(
                filename, sizeof(filename), "data/%scond/data%s%zu.bin",
                type, type, features
            );

            double *X, *y, *betas;
            if (load_dataset(filename, features, &X, &y, &betas) != 0) {
                fclose(csv);
                return 1;
            }

            for (size_t samples = 1000; samples <= 1000000; samples *= 10) {
                benchmark(X, y, betas, type, samples, features, csv);
                fflush(csv);
            }

            free(X);
            free(y);
            free(betas);
        }
    }

    fclose(csv);

    return 0;
}
