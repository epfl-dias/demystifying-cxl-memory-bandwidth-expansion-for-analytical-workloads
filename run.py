"""Run one experiment of the paper and append its rows to OUTPUT_DIR.

Usage: python3 run.py EXPERIMENT --config {full,snc2,snc4}
           --prefetch {on,off} --dram-node N --cxl-node N --cores LIST
           [--weights D:C] --output OUTPUT_DIR
"""

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

BUILD = Path(__file__).resolve().parent / "build"
RUNS = 20
# MSR 0x1a4 values for hardware prefetching on and off.
PREFETCH_MSR = {"on": "0x20", "off": "0x2f"}
WEIGHTS10 = [(1, 0)] + [(d, 10 - d) for d in range(9, 0, -1)] + [(0, 1)]
WEIGHTS20 = [(1, 0)] + [(d, 20 - d) for d in range(19, 0, -1)] + [(0, 1)]
HEADER = ["config", "prefetch", "dram_weight", "cxl_weight"]


def cpulist(text):
    cpus = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus += range(int(first), int(last or first) + 1)
    return cpus


def siblings(cores):
    out = []
    for core in cores:
        path = Path(f"/sys/devices/system/cpu/cpu{core}/topology")
        text = (path / "thread_siblings_list").read_text().strip()
        out += [c for c in cpulist(text) if c != core]
    return out


def run(*cmd, stdin=None):
    return subprocess.run(
        cmd, input=stdin, text=True, check=True, capture_output=True
    ).stdout


def placement(args, weights):
    """numactl arguments that place all memory as DRAM:CXL = weights."""
    dram, cxl = weights
    if cxl == 0:
        return [f"--membind={args.dram_node}"]
    if dram == 0:
        return [f"--membind={args.cxl_node}"]
    root = "/sys/kernel/mm/mempolicy/weighted_interleave"
    run("sudo", "tee", f"{root}/node{args.dram_node}", stdin=str(dram))
    run("sudo", "tee", f"{root}/node{args.cxl_node}", stdin=str(cxl))
    return [f"--weighted-interleave={args.dram_node},{args.cxl_node}"]


def benchmark(args, weights, cpus, *command):
    """Rows "key,GiB/s" of one benchmark run with one thread per cpu."""
    os.environ.update(
        OMP_NUM_THREADS=str(len(cpus)),
        OMP_DYNAMIC="false",
        OMP_PLACES=",".join(f"{{{c}}}" for c in cpus),
        OMP_PROC_BIND="close",
    )
    out = run(
        "numactl",
        *placement(args, weights),
        str(BUILD / command[0]),
        *command[1:],
        str(RUNS),
    )
    return [line.split(",") for line in out.split()]


def append(args, path, header, rows):
    path = args.output / path
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if new:
            writer.writerow(header)
        writer.writerows(rows)


def sweep(args, path, key, placements, *commands):
    rows = [
        (args.config, args.prefetch, *w, k, gibps)
        for w in placements
        for command in commands
        for k, gibps in benchmark(args, w, args.cores, *command)
    ]
    append(args, path, HEADER + [key, "gibps"], rows)


def fig06(args):
    sweep(
        args,
        "micro/sequential.csv",
        "kernel",
        WEIGHTS20,
        ("sequential", "read"),
        ("sequential", "write"),
    )


def fig07(args):
    sweep(args, "micro/join.csv", "phase", WEIGHTS10, ("join",))


def fig08(args):
    sweep(args, "micro/filter.csv", "selectivity", WEIGHTS10, ("filter",))


def fig09(args):
    sweep(args, "micro/partition.csv", "fanout", WEIGHTS10, ("partition",))


def fig10(args):
    """Threads on the given cores in order, then on their siblings."""
    cpus = args.cores + siblings(args.cores)
    placements = [(1, 0), (0, 1)] + ([args.weights] if args.weights else [])
    rows = [
        (args.config, args.prefetch, *w, n, k, gibps)
        for w in placements
        for n in range(1, len(cpus) + 1)
        for kernel in ("read", "simd_read")
        for k, gibps in benchmark(args, w, cpus[:n], "sequential", kernel)
    ]
    append(
        args,
        "micro/scalability.csv",
        HEADER + ["threads", "kernel", "gibps"],
        rows,
    )


def memories(args):
    return (("dram", args.dram_node), ("cxl", args.cxl_node))


def fig03(args):
    rows = [
        (args.config, args.prefetch, memory, int(delay), latency)
        for memory, node in memories(args)
        for delay, latency, _ in mlc_loaded(args, node)
        if int(delay) <= 2000
    ]
    append(
        args,
        "mlc/loaded_latency.csv",
        ["config", "prefetch", "memory", "inject_delay", "latency_ns"],
        rows,
    )


def mlc_loaded(args, node, *options):
    """Rows (inject delay, latency in ns, bandwidth in MB/s) of MLC's loaded
    latency test: latency on the first core, load on the others."""
    others = ",".join(map(str, args.cores[1:]))
    out = run(
        "numactl",
        f"--membind={node}",
        "mlc",
        "--loaded_latency",
        "-b1000000",
        f"-c{args.cores[0]}",
        f"-k{others}",
        "-r",
        "-e",
        "-L",
        *options,
    )
    return re.findall(r"^\s*(\d{5})\s+([0-9.]+)\s+([0-9.]+)", out, re.M)


def table1(args):
    """MLC read bandwidth at zero inject delay and sysbench sequential read,
    both on the given cores, in GB/s."""
    cores = ",".join(map(str, args.cores))
    rows = []
    for memory, node in memories(args):
        out = run(
            "numactl",
            f"--physcpubind={cores}",
            f"--membind={node}",
            "sysbench",
            "memory",
            "--memory-block-size=1G",
            "--memory-total-size=10000G",
            "--memory-scope=local",
            "--memory-oper=read",
            "--memory-access-mode=seq",
            "--memory-hugetlb=on",
            f"--threads={len(args.cores)}",
            "--time=60",
            "run",
        )
        mib = float(re.search(r"\(([0-9.]+) MiB/sec\)", out)[1])
        rows += [
            (args.config, args.prefetch, memory, "sysbench",
             round(mib * 2**20 / 1e9, 1)),
            (args.config, args.prefetch, memory, "mlc",
             round(float(mlc_loaded(args, node, "-d0")[0][2]) / 1e3, 1)),
        ]  # fmt: skip
    append(
        args,
        "hardware/peak_bandwidth.csv",
        ["config", "prefetch", "memory", "tool", "gbps"],
        rows,
    )


def table2(args):
    rows = []
    for memory, node in memories(args):
        out = run(
            "numactl",
            f"--membind={node}",
            "mlc",
            "--idle_latency",
            f"-c{args.cores[0]}",
            "-L",
        )
        latency = re.search(r"\(\s*([0-9.]+)\s*ns\)", out)[1]
        rows.append((args.config, memory, latency))
    append(
        args,
        "hardware/idle_latency.csv",
        ["config", "memory", "latency_ns"],
        rows,
    )


EXPERIMENTS = {
    f.__name__: f
    for f in (table1, table2, fig03, fig06, fig07, fig08, fig09, fig10)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument(
        "--config", required=True, choices=["full", "snc2", "snc4"]
    )
    parser.add_argument("--prefetch", required=True, choices=["on", "off"])
    parser.add_argument("--dram-node", required=True, type=int)
    parser.add_argument("--cxl-node", required=True, type=int)
    parser.add_argument(
        "--cores",
        required=True,
        type=cpulist,
        help="physical cores, those of the DRAM node first",
    )
    parser.add_argument(
        "--weights",
        type=lambda s: tuple(map(int, s.split(":"))),
        help="DRAM:CXL weights of the interleaved placement in fig10",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run("sudo", "wrmsr", "-a", "0x1a4", PREFETCH_MSR[args.prefetch])
    EXPERIMENTS[args.experiment](args)


if __name__ == "__main__":
    main()
