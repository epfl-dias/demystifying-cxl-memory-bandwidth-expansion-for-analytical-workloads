/* Strided read of a 32 GiB array of 8-byte integers, reading every
 * selectivity-th item for selectivity 1, 2, ..., 512. Prints
 * "selectivity,GiB/s" of the items read, the mean over the timed runs. */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#define ITEMS (1UL << 32)

static void filter(const uint64_t *a, size_t step) {
  uint64_t sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (size_t i = 0; i < ITEMS; i += step) sum += a[i];
  sink += sum;
}

int main(int argc, char **argv) {
  int runs = parse_runs(argc, argv);
  uint64_t *a = alloc_huge(ITEMS * sizeof(uint64_t));
#pragma omp parallel
  {
    unsigned seed = omp_get_thread_num();
#pragma omp for schedule(static)
    for (size_t i = 0; i < ITEMS; i++) a[i] = rand_r(&seed) & 15;
  }
  for (size_t step = 1; step <= 512; step *= 2) {
    double gibps = 0;
    filter(a, step);
    for (int r = 0; r < runs; r++) {
      double start = now();
      filter(a, step);
      gibps += ITEMS / step * sizeof(uint64_t) / GIB / (now() - start);
    }
    printf("%zu,%.6f\n", step, gibps / runs);
  }
  return 0;
}
