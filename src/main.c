/**
 * @file main.c
 * @brief Performs tests of various compilations options
 * @author Dannyel Fontenele Ribeiro
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tests.h"

typedef Results (*lr_func)(matrix_ref, matrix_ref, size_t, size_t);

void test(
    lr_func linear_regressor,
    matrix_ref X, matrix_ref Y,
    size_t n_samples, size_t n_features
) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);


    /* generate data how? */

    /* linear_regressor(); */
    Results res = linear_regressor(X, Y, n_samples, n_features);

    data_gen_time = (double)(t1 - t0)/ CLOCKS_PER_SEC;
    training_time = (double)(t2 - t1)/ CLOCKS_PER_SEC;
}

int main() {

    return 0;
}
