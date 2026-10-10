/**
 * @file datawriter.c
 * @brief Synthetic dataset generator for the OLS benchmarks
 *
 * Generates two families of binary datasets (layout: n_samples, n_features,
 * beta, y, X; X in row-major order, first column = intercept):
 *
 *  - timing sets: N_SAMPLES samples, 10 to 100 features (+1 intercept), one
 *    well-conditioned and one ill-conditioned file per size;
 *  - stability sets: STAB_N samples, with the collinearity noise swept
 *    over STAB_EXPS to reach condition numbers from ~1e2 up to ~1e8.
 *
 * Two independent noise levels are used:
 *
 *  - COL_EPS: amplitude of the noise that makes an "even" column almost a
 *    copy of the previous one. It controls the condition number of X
 *    (kappa_2(X) grows roughly as 1 / COL_EPS);
 *  - Y_NOISE: amplitude of the noise added to y (regression error term).
 *
 * @author Dannyel Fontenele Ribeiro
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SEED 42ULL

#ifndef N_SAMPLES
#define N_SAMPLES 1000000UL /**< Samples of the timing datasets */
#endif

#define STAB_N 10000UL /**< Samples of the stability datasets */

#if STAB_N > N_SAMPLES
#error "STAB_N must not exceed N_SAMPLES (shared buffers)"
#endif

#define TIME_COL_EPS 1e-3 /**< COL_EPS of the "ill" timing sets (kappa ~1e4) */
#define Y_NOISE 1e-3      /**< Noise amplitude of y */

/** Exponents e of COL_EPS = 10^-e for the stability sets (0 = well) */
static const int STAB_EXPS[] = {0, 3, 5, 6, 7};
/** Feature counts (without intercept) of the stability sets */
static const size_t STAB_FEATURES[] = {10, 50, 100};

/* --- xoshiro256** PRNG (public domain, Blackman & Vigna), seeded with
 * splitmix64: same stream on every platform, unlike rand() --- */

static uint64_t rng_state[4];

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/**
 * @brief Seeds the PRNG
 * @param seed Any 64-bit value
 */
static void rng_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++)
        rng_state[i] = splitmix64(&seed);
}

/**
 * @brief Next uniform double in [0, 1)
 */
static double rand01(void) {
    const uint64_t result = rotl64(rng_state[1] * 5, 7) * 9;
    const uint64_t t = rng_state[1] << 17;

    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl64(rng_state[3], 45);

    return (double)(result >> 11) * 0x1.0p-53;
}

/**
 * @brief Uniform noise in [-amp, amp)
 * @param amp Amplitude
 */
static inline double noise(double amp) {
    return amp * (2.0 * rand01() - 1.0);
}

/**
 * @brief Fills a dataset
 *
 * Column 0 is the intercept. Odd columns are U[0,1). Even columns (>= 2) are
 * the previous column plus noise(col_eps), unless col_eps <= 0, in which
 * case they are also U[0,1) (well-conditioned set).
 *
 * @param X Design matrix (RMO), n_samples * n_features entries
 * @param y Responses, n_samples entries
 * @param beta True coefficients, n_features entries
 * @param n_samples Number of data samples
 * @param n_features Number of features including intercept
 * @param col_eps Collinearity noise amplitude (<= 0 for independent columns)
 */
static void gen_dataset(
    double *X, double *y, double *beta,
    size_t n_samples, size_t n_features, double col_eps
) {
    for (size_t i = 0; i < n_features; i++)
        beta[i] = rand01();

    for (size_t i = 0; i < n_samples; i++) {
        X[i * n_features] = 1.0; /* Intercept */

        for (size_t j = 1; j < n_features; j++)
            X[i * n_features + j] = (col_eps <= 0.0 || j % 2 == 1)?
                rand01() : X[i * n_features + j - 1] + noise(col_eps);
    }

    for (size_t i = 0; i < n_samples; i++) {
        double val = 0.0;

        for (size_t j = 0; j < n_features; j++)
            val += X[i * n_features + j] * beta[j];

        y[i] = val + noise(Y_NOISE);
    }
}

/**
 * @brief Writes a dataset to a binary file
 *
 * Layout: n_samples, n_features, beta, y, X. Exits on any I/O error.
 *
 * @param path Destination file (its directory must already exist)
 * @param X Design matrix (RMO)
 * @param y Responses
 * @param beta True coefficients
 * @param n_samples Number of data samples
 * @param n_features Number of features including intercept
 */
static void write_dataset(
    const char *path,
    const double *X, const double *y, const double *beta,
    size_t n_samples, size_t n_features
) {
    FILE *fp = fopen(path, "wb");

    if (fp == NULL) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    if (
        fwrite(&n_samples, sizeof(size_t), 1, fp) != 1 ||
        fwrite(&n_features, sizeof(size_t), 1, fp) != 1 ||
        fwrite(beta, sizeof(double), n_features, fp) != n_features ||
        fwrite(y, sizeof(double), n_samples, fp) != n_samples ||
        fwrite(X, sizeof(double), n_features * n_samples, fp)
            != n_features * n_samples
    ) {
        fprintf(stderr, "%s: write failed\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    fclose(fp);
}

/**
 * @brief Entry point: generates every dataset in data/
 */
int main(void) {
    const size_t max_features = 100 + 1;
    char path[96];

    double *X = malloc(N_SAMPLES * max_features * sizeof(double));
    double *y = malloc(N_SAMPLES * sizeof(double));
    double *beta = malloc(max_features * sizeof(double));

    if (!X || !y || !beta) {
        fputs("failed to allocate buffers\n", stderr);
        free(X); free(y); free(beta);
        return EXIT_FAILURE;
    }

    rng_seed(SEED);

    for (int kind = 0; kind < 2; kind++) {
        const char *type = (kind == 0)? "well" : "ill";
        const double eps = (kind == 0)? 0.0 : TIME_COL_EPS;

        printf("%s-conditioned timing datasets\n", type);

        for (size_t n = 1; n < 11; n++) {
            const size_t n_features = 10 * n + 1; /* +1 for intercept */

            gen_dataset(X, y, beta, N_SAMPLES, n_features, eps);

            snprintf(
                path, sizeof path, "data/%scond/data%s%zu.bin",
                type, type, n_features
            );
            write_dataset(path, X, y, beta, N_SAMPLES, n_features);
            printf("%zu features + 1 intercept\n", n_features - 1);
        }
    }

    puts("\nstability datasets");

    for (size_t e = 0; e < sizeof STAB_EXPS / sizeof *STAB_EXPS; e++) {
        const int ex = STAB_EXPS[e];
        double eps = 0.0;

        for (int k = 0; k < ex; k++) /* eps = 10^-ex, 0 if ex == 0 */
            eps = (k == 0)? 1.0 / 10.0 : eps / 10.0;

        for (size_t f = 0; f < sizeof STAB_FEATURES / sizeof *STAB_FEATURES; f++) {
            const size_t n_features = STAB_FEATURES[f] + 1;

            gen_dataset(X, y, beta, STAB_N, n_features, eps);

            snprintf(
                path, sizeof path, "data/stability/stab_e%d_p%zu.bin",
                ex, n_features
            );
            write_dataset(path, X, y, beta, STAB_N, n_features);
            printf("e=%d, %zu features + 1 intercept\n", ex, STAB_FEATURES[f]);
        }
    }

    puts("\ndata created successfully");

    free(X);
    free(y);
    free(beta);

    return EXIT_SUCCESS;
}
