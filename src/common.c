#define _GNU_SOURCE
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

volatile uint64_t sink;

void *alloc_huge(size_t bytes) {
  size_t huge = 2UL << 20;
  bytes = (bytes + huge - 1) / huge * huge;
  void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  if (p == MAP_FAILED) {
    perror("mmap MAP_HUGETLB");
    exit(1);
  }
  return p;
}

double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int parse_runs(int argc, char **argv) {
  int runs = argc == 2 ? atoi(argv[1]) : 0;
  if (runs < 1) {
    fprintf(stderr, "usage: %s RUNS\n", argv[0]);
    exit(2);
  }
  return runs;
}
