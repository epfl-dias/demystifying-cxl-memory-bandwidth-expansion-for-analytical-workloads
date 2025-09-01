#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <stdint.h>

#define GIB (1024.0 * 1024.0 * 1024.0)

/* Maps zero-filled memory on 2 MiB huge pages; exits on failure. The NUMA
 * placement is the memory policy of the process (numactl). */
void *alloc_huge(size_t bytes);

/* Monotonic time in seconds. */
double now(void);

/* Number of timed runs, from the only command-line argument; each benchmark
 * runs one untimed warm-up first. */
int parse_runs(int argc, char **argv);

/* Keeps results alive so the compiler does not drop the measured loops. */
extern volatile uint64_t sink;

#endif
