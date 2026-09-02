#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>

#define SEED 42
#define RAND() ((double)rand() / (double)RAND_MAX)

/* layout: beta, y, x */

int main() {
    const size_t n_samples = 1000000, n_features = 10;
    const double epsilon = 0.001;
    int status_code = EXIT_SUCCESS;

    FILE *fp;
    double *X, *y, *beta;

    X = malloc(n_samples * n_features * sizeof(size_t));
    if (!X) {
        fputs("failed to allocate design matrix\n", stderr);
        exit(EXIT_FAILURE);
    }

    y = malloc(n_samples * sizeof(size_t));
    if (!y) {
        fputs("failed to allocate result matrix\n", stderr);
        status_code = EXIT_FAILURE;
        goto done_x;
    }

    beta = malloc(n_features * sizeof(size_t));
    if (!beta) {
        fputs("failed to allocate coefficients matrix\n", stderr);
        status_code = EXIT_FAILURE;
        goto done_y;
    }

    fp = fopen("data/datarand.bin", "wb");
    if (!fp) {
        fputs("failed to open well-conditioned file\n", stderr);
        status_code = EXIT_FAILURE;
        goto done_beta;
    }

    srand(SEED);

    for (size_t i = 0; i < n_features; i++)
        beta[i] = RAND();

    for (size_t i = 0; i < n_samples; i++)
        for (size_t j = 0; j < n_features; j++)
            X[i * n_features + j] = RAND();

    for (size_t i = 0; i < n_samples; i++) {
        double val = 0.0;

        for (size_t j = 0; j < n_features; j++) 
            val += X[i * n_features + j] * beta[j];

        y[i] = val + epsilon * (2.0 * RAND() - 1.0);
    }

    if (
        fwrite(beta, sizeof(double), n_features, fp) != n_features ||
        fwrite(y, sizeof(double), n_samples, fp) != n_samples ||
        fwrite(X, sizeof(double), n_features * n_samples, fp) != n_features * n_samples
    ) {
        fputs("error writing to datarand.bin\n", stderr);
        status_code = EXIT_FAILURE;
        goto done_file;
    }

    fp = freopen("data/datacond.bin", "wb", fp);

    if (!fp) {
        fputs("failure to open ill-conditioned file\n", stderr);
        status_code = EXIT_FAILURE;
        goto done_file;
    }

    for (size_t i = 0; i < n_samples; i++)
        for (size_t j = 0; j < n_features; j++)
            X[i * n_features + j] = (j % 2 == 0)?
                RAND() : X[i * n_features + j - 1] + epsilon * (2 * RAND() - 1);

    for (size_t i = 0; i < n_samples; i++) {
        double val = 0.0;

        for (size_t j = 0; j < n_features; j++) 
            val += X[i * n_features + j] * beta[j];

        y[i] = val + epsilon * (2.0 * RAND() - 1.0);
    }

    if (
        fwrite(beta, sizeof(double), n_features, fp) != n_features ||
        fwrite(y, sizeof(double), n_samples, fp) != n_samples ||
        fwrite(X, sizeof(double), n_features * n_samples, fp) != n_features * n_samples
    ) {
        fputs("error writing to datacond.bin\n", stderr);
        status_code = EXIT_FAILURE;
        goto done_file;
    }

    done_file:
    fclose(fp);
    done_beta:
    free(beta);
    done_y:
    free(y);
    done_x:
    free(X);

    end:
    if (status_code == EXIT_SUCCESS)
        puts("data created successfully");
    exit(status_code);
}
