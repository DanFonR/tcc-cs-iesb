/**
 * @file main.c
 * @brief Performs tests of various compilations options
 * @author Dannyel Fontenele Ribeiro
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openblas/cblas.h>
#include "tests.h"

typedef Results (*regressor)(matrix_ref, matrix_ref, size_t, size_t);

static volatile double warmup_sink;

static inline void warmup_consume(const Results *r)
{
    warmup_sink += r->numerical_results.coefs[0];
}

void benchmark(
    const char *type, size_t n_samples, size_t n_features,
    FILE *csv
) {
    const size_t warmups = 3, runs = 20;

    char filename[64];
    sprintf(filename, "data/%scond/data%s%zu.bin", type, type, n_features);

    FILE *dataset = fopen(filename, "rb");
    if (dataset == NULL) return;

    size_t read_n_features = 0, read_n_samples = 0;

    if (fread(&read_n_samples, sizeof(size_t), 1, dataset) != 1 ||
        fread(&read_n_features, sizeof(size_t), 1, dataset) != 1
    ) {
        fclose(dataset);

        return;
    }

    if (read_n_features != n_features || read_n_samples != 1000000) {
        fclose(dataset);

        return;
    }

    double *X = malloc(1000000ULL * n_features * sizeof(double));
    double *y = malloc(1000000ULL * sizeof(double));
    double *betas = malloc(n_features * sizeof(double));

    if (X == NULL || y == NULL || betas == NULL) {
        free(X);
        free(y);
        free(betas);

        fclose(dataset);

        return;
    }

    if (
        fread(betas, sizeof(double), n_features, dataset) != n_features ||
        fread(y, sizeof(double), 1000000, dataset) != 1000000 ||
        fread(X, sizeof(double), 1000000 * n_features, dataset)
            != 1000000 * n_features
    ) {
        free(X);
        free(y);
        free(betas);

        fclose(dataset);

        return;
    }

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
            RunInfo metadata = {
                .compiler = compilers[f % 2],
                .data_type = type,
                .lr_method = (f == funcs_len - 1)? "QR" : "NE",
                .n_features = n_features,
                .n_samples = n_samples,
                .optim = (f == funcs_len - 1)? "native" : options[f / 2],
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
                "%.15f,%.15f,"
                "%.15f,%.15f,"
                "%.15f,%.15f,"
                "%.15f,"
                "%.15f,"
                "%.15f,"
                "%.15f,%.15f,"
                "%.15f,"
                "%.15f,"
                "%.15f,"
                "%.15f,"
                "%.15f,%.15f\n",
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

        for (size_t features = 11; features <= 101; features += 10) 
            for (size_t samples = 1000; samples <= 1000000; samples *= 10) 
                benchmark(type, samples, features, csv);
    }

    return 0;
}
