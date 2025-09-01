/* Hash join on 16-byte buckets with the identity hash. The build phase reads
 * 2^28 unique random keys (2 GiB) in order and writes their buckets in a
 * 16 GiB table of 2^30 buckets; the probe phase looks up 2^32 keys, wrapping
 * around the key array. Prints "phase,GiB/s" as tuples * 16 bytes per
 * second, the mean over the timed runs. */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#define BUCKETS (1UL << 30)
#define KEYS (1UL << 28)
#define PROBES (1UL << 32)

typedef struct {
  uint64_t key;
  uint64_t value;
} bucket_t;

/* Bijection on [0, 2^30): xorshifts and odd multiplications mod 2^30, so the
 * keys are unique and spread over the table. */
static uint64_t mix(uint64_t x) {
  uint64_t mask = BUCKETS - 1;
  x = (x ^ (x >> 15)) * 0x2c1b3c6dUL & mask;
  x = (x ^ (x >> 12)) * 0x297a2d39UL & mask;
  return x ^ (x >> 15);
}

static void build(bucket_t *table, const uint64_t *keys) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < KEYS; i++) {
    uint64_t k = keys[i];
    table[k].key = k;
    table[k].value = k * 13;
  }
}

static uint64_t probe(const bucket_t *table, const uint64_t *keys) {
  uint64_t matches = 0, sum = 0;
#pragma omp parallel for reduction(+ : matches, sum) schedule(static)
  for (size_t i = 0; i < PROBES; i++) {
    uint64_t k = keys[i % KEYS];
    if (table[k].key == k) {
      matches++;
      sum += table[k].value;
    }
  }
  sink += sum;
  return matches;
}

int main(int argc, char **argv) {
  int runs = parse_runs(argc, argv);
  bucket_t *table = alloc_huge(BUCKETS * sizeof(bucket_t));
  uint64_t *keys = alloc_huge(KEYS * sizeof(uint64_t));
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < KEYS; i++) keys[i] = mix(i);
  double build_gibps = 0, probe_gibps = 0;
  for (int r = 0; r <= runs; r++) {
    double start = now();
    build(table, keys);
    double built = now();
    uint64_t matches = probe(table, keys);
    double probed = now();
    if (matches != PROBES) {
      fprintf(stderr, "probe matched %lu of %lu keys\n", matches, PROBES);
      return 1;
    }
    if (r > 0) {
      build_gibps += KEYS * sizeof(bucket_t) / GIB / (built - start);
      probe_gibps += PROBES * sizeof(bucket_t) / GIB / (probed - built);
    }
  }
  printf("build,%.6f\nprobe,%.6f\n", build_gibps / runs, probe_gibps / runs);
  return 0;
}
