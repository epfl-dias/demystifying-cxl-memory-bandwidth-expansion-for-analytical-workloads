/* Radix partitioning of a 16 GiB array of shuffled 8-byte keys on their low
 * bits, for fanout 16, 32, ..., 16384: per-thread histograms, prefix sums,
 * then a scatter into one output array. Prints "fanout,GiB/s" of the input,
 * the mean over the timed runs. */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#define ITEMS (1UL << 31)
#define MAX_FANOUT 16384

static void partition(const uint64_t *in, uint64_t *out, size_t fanout,
                      size_t *hist) {
  uint64_t mask = fanout - 1;
#pragma omp parallel
  {
    int threads = omp_get_num_threads();
    int t = omp_get_thread_num();
    size_t begin = ITEMS / threads * t;
    size_t end = t == threads - 1 ? ITEMS : ITEMS / threads * (t + 1);
    size_t *own = hist + (size_t)t * fanout;
    memset(own, 0, fanout * sizeof(size_t));
    for (size_t i = begin; i < end; i++) own[in[i] & mask]++;
#pragma omp barrier
    /* Thread t writes partition p after partitions < p of all threads and
     * after partition p of threads < t. */
    size_t *offset = malloc(fanout * sizeof(size_t));
    size_t base = 0;
    for (size_t p = 0; p < fanout; p++) {
      offset[p] = base;
      for (int u = 0; u < threads; u++) {
        if (u < t) offset[p] += hist[(size_t)u * fanout + p];
        base += hist[(size_t)u * fanout + p];
      }
    }
    for (size_t i = begin; i < end; i++) out[offset[in[i] & mask]++] = in[i];
    free(offset);
  }
}

int main(int argc, char **argv) {
  int runs = parse_runs(argc, argv);
  uint64_t *in = alloc_huge(ITEMS * sizeof(uint64_t));
  uint64_t *out = alloc_huge(ITEMS * sizeof(uint64_t));
  size_t *hist = malloc((size_t)omp_get_max_threads() * MAX_FANOUT *
                        sizeof(size_t));
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < ITEMS; i++) {
    in[i] = i + 1;
    out[i] = 0;
  }
  srand(32768);
  for (size_t i = ITEMS - 1; i > 0; i--) {
    size_t j = (size_t)((double)rand() / ((double)RAND_MAX + 1) * i);
    uint64_t x = in[i];
    in[i] = in[j];
    in[j] = x;
  }
  for (size_t fanout = 16; fanout <= MAX_FANOUT; fanout *= 2) {
    double gibps = 0;
    partition(in, out, fanout, hist);
    for (int r = 0; r < runs; r++) {
      double start = now();
      partition(in, out, fanout, hist);
      gibps += ITEMS * sizeof(uint64_t) / GIB / (now() - start);
    }
    uint64_t sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (size_t i = 0; i < ITEMS; i++) sum += out[i];
    if (sum != ITEMS * (ITEMS + 1) / 2) {
      fprintf(stderr, "partition output is not a permutation\n");
      return 1;
    }
    printf("%zu,%.6f\n", fanout, gibps / runs);
  }
  return 0;
}
