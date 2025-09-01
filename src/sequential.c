/* Sequential read, write or AVX-512 read over a 32 GiB array of 8-byte
 * integers. Usage: sequential {read,write,simd_read} RUNS. Prints
 * "kernel,GiB/s", the mean over the timed runs. */
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#define ITEMS (1UL << 32)

static void scalar_read(uint64_t *a) {
  double sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (size_t i = 0; i < ITEMS; i++) sum += a[i];
  sink += (uint64_t)sum;
}

static void scalar_write(uint64_t *a) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < ITEMS; i++) a[i] = i;
}

static void simd_read(uint64_t *a) {
  uint64_t sum = 0;
#pragma omp parallel reduction(+ : sum)
  {
    __m512i acc = _mm512_setzero_si512();
#pragma omp for schedule(static)
    for (size_t i = 0; i < ITEMS; i += 8)
      acc = _mm512_add_epi64(acc, _mm512_load_si512(&a[i]));
    sum += _mm512_reduce_add_epi64(acc);
  }
  sink += sum;
}

int main(int argc, char **argv) {
  static const struct {
    const char *name;
    void (*run)(uint64_t *);
  } kernels[] = {{"read", scalar_read},
                 {"write", scalar_write},
                 {"simd_read", simd_read}};
  void (*kernel)(uint64_t *) = NULL;
  for (size_t k = 0; argc == 3 && k < 3; k++)
    if (strcmp(argv[1], kernels[k].name) == 0) kernel = kernels[k].run;
  if (kernel == NULL) {
    fprintf(stderr, "usage: %s {read,write,simd_read} RUNS\n", argv[0]);
    return 2;
  }
  int runs = parse_runs(argc - 1, argv + 1);
  uint64_t *a = alloc_huge(ITEMS * sizeof(uint64_t));
#pragma omp parallel
  {
    unsigned seed = omp_get_thread_num();
#pragma omp for schedule(static)
    for (size_t i = 0; i < ITEMS; i++) a[i] = rand_r(&seed) & 15;
  }
  double gibps = 0;
  kernel(a);
  for (int r = 0; r < runs; r++) {
    double start = now();
    kernel(a);
    gibps += ITEMS * sizeof(uint64_t) / GIB / (now() - start);
  }
  printf("%s,%.6f\n", argv[1], gibps / runs);
  return 0;
}
