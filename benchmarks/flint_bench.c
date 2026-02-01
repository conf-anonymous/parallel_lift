/*
 * FLINT Benchmark for Exact Linear Solve
 *
 * Compares FLINT's fmpz_mat_solve against Parallel Lift's GPU implementation.
 *
 * Compile: gcc -O3 -o flint_bench flint_bench.c -lflint -lgmp -lmpfr -lm
 * Run: ./flint_bench 64 16
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>

/* Simple timing utility */
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* Generate random matrix with diagonal dominance */
void generate_random_matrix(fmpz_mat_t A, int n, unsigned long seed) {
    flint_rand_t state;
    flint_randinit(state);
    flint_randseed(state, seed, seed + 1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fmpz* entry = fmpz_mat_entry(A, i, j);
            if (i == j) {
                /* Diagonal: 100 to 1000 */
                fmpz_set_ui(entry, 100 + (n_randint(state, 900)));
            } else {
                /* Off-diagonal: -50 to 50 */
                slong val = (slong)n_randint(state, 101) - 50;
                fmpz_set_si(entry, val);
            }
        }
    }

    flint_randclear(state);
}

/* Generate random RHS vector */
void generate_random_rhs(fmpz_mat_t b, int n, unsigned long seed) {
    flint_rand_t state;
    flint_randinit(state);
    flint_randseed(state, seed, seed + 1);

    for (int i = 0; i < n; i++) {
        fmpz* entry = fmpz_mat_entry(b, i, 0);
        slong val = (slong)n_randint(state, 201) - 100;
        fmpz_set_si(entry, val);
    }

    flint_randclear(state);
}

/* Benchmark FLINT's fmpz_mat_solve for multi-RHS */
double benchmark_flint_multi_rhs(int n, int k, int trials) {
    fmpz_mat_t A, B, X;
    fmpz_t den;

    /* Initialize matrices */
    fmpz_mat_init(A, n, n);
    fmpz_mat_init(B, n, k);
    fmpz_mat_init(X, n, k);
    fmpz_init(den);

    /* Generate test data */
    generate_random_matrix(A, n, 42);

    /* Generate k RHS vectors as columns of B */
    for (int j = 0; j < k; j++) {
        fmpz_mat_t b_col;
        fmpz_mat_init(b_col, n, 1);
        generate_random_rhs(b_col, n, 42 + j);

        for (int i = 0; i < n; i++) {
            fmpz_set(fmpz_mat_entry(B, i, j), fmpz_mat_entry(b_col, i, 0));
        }

        fmpz_mat_clear(b_col);
    }

    /* Warmup run */
    fmpz_mat_solve(X, den, A, B);

    /* Timed runs */
    double total_time = 0.0;
    for (int t = 0; t < trials; t++) {
        double start = get_time_ms();
        fmpz_mat_solve(X, den, A, B);
        double end = get_time_ms();
        total_time += (end - start);
    }

    /* Cleanup */
    fmpz_mat_clear(A);
    fmpz_mat_clear(B);
    fmpz_mat_clear(X);
    fmpz_clear(den);

    return total_time / trials;
}

/* Benchmark FLINT's determinant */
double benchmark_flint_determinant(int n, int trials) {
    fmpz_mat_t A;
    fmpz_t det;

    fmpz_mat_init(A, n, n);
    fmpz_init(det);

    generate_random_matrix(A, n, 42);

    /* Warmup */
    fmpz_mat_det(det, A);

    /* Timed runs */
    double total_time = 0.0;
    for (int t = 0; t < trials; t++) {
        double start = get_time_ms();
        fmpz_mat_det(det, A);
        double end = get_time_ms();
        total_time += (end - start);
    }

    fmpz_mat_clear(A);
    fmpz_clear(det);

    return total_time / trials;
}

void print_header() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              FLINT Baseline Benchmark                         ║\n");
    printf("║     Comparison with State-of-the-Art CPU Exact Arithmetic     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    printf("FLINT version: %s\n\n", FLINT_VERSION);
}

int main(int argc, char *argv[]) {
    int max_size = 256;
    int k = 16;
    int trials = 5;
    const char* export_file = NULL;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-size") == 0 && i + 1 < argc) {
            max_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rhs") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
            trials = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) {
            export_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --max-size N   Maximum matrix size (default: 256)\n");
            printf("  --rhs K        Number of RHS vectors (default: 16)\n");
            printf("  --trials N     Number of trials (default: 5)\n");
            printf("  --export FILE  Export results to CSV file\n");
            return 0;
        }
    }

    print_header();

    printf("Configuration:\n");
    printf("  Max size: %d\n", max_size);
    printf("  RHS (k):  %d\n", k);
    printf("  Trials:   %d\n\n", trials);

    FILE* csv = NULL;
    if (export_file) {
        csv = fopen(export_file, "w");
        if (csv) {
            fprintf(csv, "n,k,flint_solve_ms,flint_det_ms\n");
        }
    }

    /* Multi-RHS Solve benchmark */
    printf("Multi-RHS Solve (AX = B):\n");
    printf("┌─────────┬─────┬──────────────┐\n");
    printf("│    n    │  k  │  FLINT (ms)  │\n");
    printf("├─────────┼─────┼──────────────┤\n");

    int sizes[] = {32, 64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes && sizes[i] <= max_size; i++) {
        int n = sizes[i];
        double flint_ms = benchmark_flint_multi_rhs(n, k, trials);
        printf("│ %7d │ %3d │ %12.2f │\n", n, k, flint_ms);

        if (csv) {
            double det_ms = benchmark_flint_determinant(n, trials);
            fprintf(csv, "%d,%d,%.6f,%.6f\n", n, k, flint_ms, det_ms);
        }
    }

    printf("└─────────┴─────┴──────────────┘\n\n");

    /* Determinant benchmark */
    printf("Determinant Computation:\n");
    printf("┌─────────┬──────────────┐\n");
    printf("│    n    │  FLINT (ms)  │\n");
    printf("├─────────┼──────────────┤\n");

    for (int i = 0; i < num_sizes && sizes[i] <= max_size; i++) {
        int n = sizes[i];
        double flint_ms = benchmark_flint_determinant(n, trials);
        printf("│ %7d │ %12.2f │\n", n, flint_ms);
    }

    printf("└─────────┴──────────────┘\n");

    if (csv) {
        fclose(csv);
        printf("\nResults exported to: %s\n", export_file);
    }

    return 0;
}
