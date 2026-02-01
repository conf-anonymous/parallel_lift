/*
 * IML (Integer Matrix Library) Benchmark for Exact Linear Solve
 *
 * Compares IML's nonsingSolvLlhsMM against Parallel Lift's GPU implementation.
 *
 * Compile: gcc -O3 -o iml_bench iml_bench.c -liml -lgmp -lblas -llapack -lm
 * Run: ./iml_bench --max-size 256 --rhs 16
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <gmp.h>
#include <iml.h>

/* Simple timing utility */
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* Simple LCG random number generator for reproducibility */
static unsigned long lcg_state = 42;

unsigned long lcg_rand() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1;
    return lcg_state;
}

void lcg_seed(unsigned long seed) {
    lcg_state = seed;
}

/* Generate random matrix with diagonal dominance (mpz_t version) */
void generate_random_matrix_mpz(mpz_t *A, long n, unsigned long seed) {
    lcg_seed(seed);

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            mpz_t *entry = &A[i * n + j];
            if (i == j) {
                /* Diagonal: 100 to 1000 */
                mpz_set_ui(*entry, 100 + (lcg_rand() % 900));
            } else {
                /* Off-diagonal: -50 to 50 */
                long val = (long)(lcg_rand() % 101) - 50;
                mpz_set_si(*entry, val);
            }
        }
    }
}

/* Generate random RHS matrix (mpz_t version) */
void generate_random_rhs_mpz(mpz_t *B, long n, long m, unsigned long seed) {
    lcg_seed(seed);

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < m; j++) {
            mpz_t *entry = &B[i * m + j];
            long val = (long)(lcg_rand() % 201) - 100;
            mpz_set_si(*entry, val);
        }
    }
}

/* Benchmark IML's nonsingSolvLlhsMM for multi-RHS */
double benchmark_iml_multi_rhs(long n, long k, int trials) {
    /* Allocate matrices */
    mpz_t *A = (mpz_t *)malloc(n * n * sizeof(mpz_t));
    mpz_t *B = (mpz_t *)malloc(n * k * sizeof(mpz_t));
    mpz_t *X = (mpz_t *)malloc(n * k * sizeof(mpz_t));
    mpz_t D;

    /* Initialize all mpz_t elements */
    for (long i = 0; i < n * n; i++) mpz_init(A[i]);
    for (long i = 0; i < n * k; i++) mpz_init(B[i]);
    for (long i = 0; i < n * k; i++) mpz_init(X[i]);
    mpz_init(D);

    /* Generate test data */
    generate_random_matrix_mpz(A, n, 42);
    generate_random_rhs_mpz(B, n, k, 43);

    /* Warmup run */
    nonsingSolvLlhsMM(RightSolu, n, k, A, B, X, D);

    /* Re-generate data (IML may modify inputs) */
    generate_random_matrix_mpz(A, n, 42);
    generate_random_rhs_mpz(B, n, k, 43);

    /* Timed runs */
    double total_time = 0.0;
    for (int t = 0; t < trials; t++) {
        /* Regenerate for each trial since IML may modify A */
        generate_random_matrix_mpz(A, n, 42);
        generate_random_rhs_mpz(B, n, k, 43);

        double start = get_time_ms();
        nonsingSolvLlhsMM(RightSolu, n, k, A, B, X, D);
        double end = get_time_ms();
        total_time += (end - start);
    }

    /* Cleanup */
    for (long i = 0; i < n * n; i++) mpz_clear(A[i]);
    for (long i = 0; i < n * k; i++) mpz_clear(B[i]);
    for (long i = 0; i < n * k; i++) mpz_clear(X[i]);
    mpz_clear(D);
    free(A);
    free(B);
    free(X);

    return total_time / trials;
}

void print_header() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              IML Baseline Benchmark                           ║\n");
    printf("║     Integer Matrix Library - Exact Linear Algebra             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    printf("IML: Integer Matrix Library (Waterloo)\n");
    printf("Using nonsingSolvLlhsMM for exact rational system solving\n\n");
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
            fprintf(csv, "n,k,iml_solve_ms\n");
        }
    }

    /* Multi-RHS Solve benchmark */
    printf("Multi-RHS Solve (AX = B) using nonsingSolvLlhsMM:\n");
    printf("┌─────────┬─────┬──────────────┐\n");
    printf("│    n    │  k  │   IML (ms)   │\n");
    printf("├─────────┼─────┼──────────────┤\n");

    int sizes[] = {32, 64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes && sizes[i] <= max_size; i++) {
        int n = sizes[i];
        double iml_ms = benchmark_iml_multi_rhs(n, k, trials);
        printf("│ %7d │ %3d │ %12.2f │\n", n, k, iml_ms);

        if (csv) {
            fprintf(csv, "%d,%d,%.6f\n", n, k, iml_ms);
        }
    }

    printf("└─────────┴─────┴──────────────┘\n");

    if (csv) {
        fclose(csv);
        printf("\nResults exported to: %s\n", export_file);
    }

    return 0;
}
