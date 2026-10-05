#include <stdio.h>
#include <stdlib.h>

#define SEED 42
#define RAND() ((double)rand() / (double)RAND_MAX)

#define EPSILON 0.001

static inline double noise() {
    return (EPSILON * (2.0 * RAND() - 1.0));
}

typedef enum {
    NO_ERROR = -1,
    DESIGN_MATRIX,
    RESULT_MATRIX,
    COEF_MATRIX,
    WELL_COND,
    ILL_COND
} err_codes;

typedef enum { WELL, ILL } cond_type;

void prog_exit(
    FILE *fp, double **allocs[], size_t n_allocs,
    err_codes err,
    int exit_status
) {
    const char *reasons[] = {
        "failed to allocate design matrix\n",
        "failed to allocate result matrix\n",
        "failed to allocate coefficients matrix\n",
        "failed to open well-conditioned file\n",
        "failed to open ill-conditioned file\n",
    };

    for (size_t i = 0; i < n_allocs; i++)
        free(*(allocs[i]));

    if (fp != NULL) fclose(fp);

    if (err != NO_ERROR) fputs(reasons[err], stderr);

    exit(exit_status);
}

void gen_dataset(
    cond_type cond,
    double *X, double *y, double *beta,
    size_t n_samples, size_t n_features
) {
    /* Populating betas */
    for (size_t i = 0; i < n_features; i++)
    beta[i] = RAND();

    /* Populating X */
    for (size_t i = 0; i < n_samples; i++) {
        X[i * n_features] = 1; /* Intercept */

        for (size_t j = 1; j < n_features; j++)
            X[i * n_features + j] = (cond == WELL || j % 2 == 1)?
                RAND() : X[i * n_features + j - 1] + noise();
    }

    /* Populating Y */
    for (size_t i = 0; i < n_samples; i++) {
        double val = 0.0;

        for (size_t j = 0; j < n_features; j++) 
            val += X[i * n_features + j] * beta[j];

        y[i] = val + noise();
    }
}

void write_dataset(
    cond_type cond,
    double *X, double *y, double *beta,
    size_t n_samples, size_t n_features
) {
    char filename[32];
    double **matrices[] = {&X, &y, &beta};
    size_t matrices_len = sizeof(matrices) / sizeof(*matrices);
    err_codes code;

    if (cond == WELL) {
        snprintf(
            filename, sizeof(filename),
            "data/wellcond/datawell%zu.bin", n_features
        );
        code = WELL_COND;
    }
    else {
        snprintf(
            filename, sizeof(filename),
            "data/illcond/dataill%zu.bin", n_features
        );
        code = ILL_COND;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) prog_exit(fp, matrices, matrices_len, code, EXIT_FAILURE);

    if (
        fwrite(&n_samples, sizeof(size_t), 1, fp) != 1 ||
        fwrite(&n_features, sizeof(size_t), 1, fp) != 1 ||
        fwrite(beta, sizeof(double), n_features, fp) != n_features ||
        fwrite(y, sizeof(double), n_samples, fp) != n_samples ||
        fwrite(X, sizeof(double), n_features * n_samples, fp)
            != n_features * n_samples
    )
        prog_exit(fp, matrices, matrices_len, code, EXIT_FAILURE);
}

/* layout: n_samples, n_features, beta, y, x */

int main() {
    const size_t n_samples = 1000000, init_n_features = 100 + 1;
    double *X = NULL, *y = NULL, *beta = NULL;
    double **matrices[] = {&X, &y, &beta};
    size_t matrices_len = sizeof(matrices) / sizeof(*matrices);

    X = malloc(n_samples * init_n_features * sizeof(double));
    if (!X)
        prog_exit(NULL, matrices, matrices_len, DESIGN_MATRIX, EXIT_FAILURE);

    y = malloc(n_samples * sizeof(double));
    if (!y)
        prog_exit(NULL, matrices, matrices_len, RESULT_MATRIX, EXIT_FAILURE);

    beta = malloc(init_n_features * sizeof(double));
    if (!beta)
        prog_exit(NULL, matrices, matrices_len, COEF_MATRIX, EXIT_FAILURE);

    srand(SEED);

    puts("well-conditioned dataset");
    for (int n = 1; n < 11; n++) {
        const size_t n_features = 10 * n;

        /* +1 for intercept */
        gen_dataset(WELL, X, y, beta, n_samples, n_features + 1);
        write_dataset(WELL, X, y, beta, n_samples, n_features + 1);
        printf("%d features + 1 intercept\n", n_features);
    }

    puts("\nill-conditioned dataset");
    for (int n = 1; n < 11; n++) {
        const size_t n_features = 10 * n;

        gen_dataset(ILL, X, y, beta, n_samples, n_features + 1);
        write_dataset(ILL, X, y, beta, n_samples, n_features + 1);
        printf("%d features + 1 intercept\n", n_features);
    }

    puts("\ndata created successfully");
    prog_exit(NULL, matrices, matrices_len, NO_ERROR, EXIT_SUCCESS);
}
