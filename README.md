# Demystifying CXL Memory Bandwidth Expansion for Analytical Workloads

Code for the experiments of our ADMS 2025 paper ([PDF][paper]).

[paper]: https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/ADMS/ADMS25-02.pdf

## Requirements

- A two-socket Intel Xeon (Sapphire Rapids) server with a CXL type 3 memory
  expander that Linux exposes as a CPU-less NUMA node. Sub-NUMA clustering
  (full socket, SNC-2, SNC-4) is a BIOS setting.
- Linux 6.9 or later (weighted interleaving), numactl with
  `--weighted-interleave`, msr-tools, sudo.
- 48 GiB of 2 MiB huge pages on the DRAM node and on the CXL node.
- GCC with OpenMP and AVX-512, CMake, Python 3 with matplotlib.
- Intel MLC and sysbench for Tables 1 and 2 and Figure 3.

## Build

```sh
cmake -B build && cmake --build build
```

## Run

`run.py` runs one experiment on one hardware configuration and appends its
results as CSV files to `--output`. It sets the prefetchers through MSR
0x1a4 and the DRAM:CXL weights through
`/sys/kernel/mm/mempolicy/weighted_interleave`. `--cores` lists the
physical cores of one socket, those local to the DRAM node first.

```sh
python3 run.py fig06 --config snc2 --prefetch on \
    --dram-node 0 --cxl-node 4 --cores 0-47 --output results
```

| Experiment | Output | Description |
|---|---|---|
| `table1` | `hardware/peak_bandwidth.csv` | MLC and sysbench read bandwidth |
| `table2` | `hardware/idle_latency.csv` | MLC idle latency |
| `fig03` | `mlc/loaded_latency.csv` | MLC loaded latency |
| `fig06` | `micro/sequential.csv` | sequential read and write, 21 weights |
| `fig07` | `micro/join.csv` | hash join build and probe, 11 weights |
| `fig08` | `micro/filter.csv` | strided read, selectivity 1 to 512 |
| `fig09` | `micro/partition.csv` | radix partitioning, fanout 16 to 16384 |
| `fig10` | `micro/scalability.csv` | read throughput over 1 to 96 threads |

`fig10` adds an interleaved placement with `--weights D:C`.

## Plot

```sh
python3 plot.py results figures
```

writes the figures for which `results` holds the input files.

## Citation

```bibtex
@inproceedings{lebedev2025demystifying,
  author    = {Georgiy Lebedev and Hamish Nicholson and Musa {\"U}nal and
               Sanidhya Kashyap and Anastasia Ailamaki},
  title     = {Demystifying {CXL} Memory Bandwidth Expansion for Analytical
               Workloads},
  booktitle = {VLDB Workshops},
  publisher = {VLDB.org},
  year      = {2025},
}
```
