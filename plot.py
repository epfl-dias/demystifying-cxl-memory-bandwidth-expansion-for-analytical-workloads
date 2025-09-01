"""Plot the figures of the paper from a data directory.

Usage: python3 plot.py DATA_DIR OUTPUT_DIR
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CONFIGS = [("full", "Full Socket"), ("snc2", "SNC-2"), ("snc4", "SNC-4")]
PREFETCH = [
    ("on", "HW prefetching enabled"),
    ("off", "HW prefetching disabled"),
]
# DRAM, 9:1, ..., 1:9, CXL.
WEIGHTS10 = [(1, 0)] + [(d, 10 - d) for d in range(9, 0, -1)] + [(0, 1)]
WEIGHTS20 = [(1, 0)] + [(d, 20 - d) for d in range(19, 0, -1)] + [(0, 1)]
PALETTE = [
    "#E6194B",
    "#3CB44B",
    "#FFE119",
    "#0082C8",
    "#F58231",
    "#911EB4",
    "#46F0F0",
    "#F032E6",
    "#D2F53C",
    "#FABEBE",
    "#008080",
]
VIRIDIS = [plt.cm.viridis(x) for x in (0.2, 0.5, 0.8)]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def label(w):
    return {(1, 0): "DRAM", (0, 1): "CXL"}.get(w, f"{w[0]}:{w[1]}")


def weights(row):
    return int(row["dram_weight"]), int(row["cxl_weight"])


def grid(rows, cols, width, height):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    return fig, axes


def caption(ax, index, text):
    ax.set_title(
        f"({chr(ord('a') + index)}) {text}",
        y=-0.5,
        fontsize=7,
        fontweight="bold",
    )


def legend(fig, labels, colors, y=1.04):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, y),
        frameon=False,
    )


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig01(data, out):
    rows = read(data / "hardware/dram_bandwidth_per_core.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.bar(
        [r["cpu"].replace(" ", "\n") for r in rows],
        [float(r["gbps_per_core"]) for r in rows],
        color="blue",
    )
    ax.set_ylabel("DRAM Bandwidth Per Core, GB/s")
    save(fig, out, "fig01")


def fig03(data, out):
    rows = read(data / "mlc/loaded_latency.csv")
    idle = {
        (r["config"], r["memory"]): float(r["latency_ns"])
        for r in read(data / "hardware/idle_latency.csv")
    }
    lines = [(c, p, "dram") for p, _ in PREFETCH for c, _ in CONFIGS]
    lines += [("full", "on", "cxl"), ("full", "off", "cxl")]
    names = {c: n for c, n in CONFIGS}
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for (config, prefetch, memory), color in zip(lines, PALETTE):
        pts = [
            (1 - int(r["inject_delay"]) / 20000, float(r["latency_ns"]))
            for r in rows
            if (r["config"], r["prefetch"], r["memory"])
            == (config, prefetch, memory)
        ]
        if not pts:
            continue
        name = f"{names[config]} DRAM" if memory == "dram" else "CXL"
        state = "enabled" if prefetch == "on" else "disabled"
        ax.plot(
            [100 * x for x, _ in pts],
            [y for _, y in pts],
            marker="o",
            markersize=2,
            color=color,
            label=f"{name}, HW prefetching {state}",
        )
    cxl = [v for (_, memory), v in idle.items() if memory == "cxl"]
    if cxl:
        ax.axhline(
            cxl[0],
            linestyle="--",
            color=PALETTE[8],
            label="CXL idle latency",
        )
    ax.set_yscale("log")
    ax.set_yticks([100, 220, 350, 1200], ["100", "220", "350", "1200"])
    ax.minorticks_off()
    ax.set_xlabel("Memory Bandwidth Utilization, %")
    ax.set_ylabel("Memory Latency, ns")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False)
    fig.savefig(out / "fig03.pdf", bbox_inches="tight")
    plt.close(fig)


def geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def ssb(data):
    runtimes = defaultdict(dict)
    for r in read(data / "ssb/runtimes.csv"):
        runtimes[r["config"], r["prefetch"], weights(r)][r["query"]] = float(
            r["runtime_ms"]
        )
    return runtimes


def speedup(runtimes, key, w):
    base = runtimes[(*key, (1, 0))]
    return geomean([base[q] / t for q, t in runtimes[(*key, w)].items()])


def fig04(data, out):
    runtimes = ssb(data)
    fig, axes = grid(2, 3, 8, 4.2)
    for i, ((prefetch, ptext), (config, ctext)) in enumerate(
        (p, c) for p in PREFETCH for c in CONFIGS
    ):
        ax = axes[i // 3][i % 3]
        ax.bar(
            range(11),
            [speedup(runtimes, (config, prefetch), w) for w in WEIGHTS10],
            width=0.6,
            color=PALETTE,
        )
        ax.set_xticks(range(11), [label(w) for w in WEIGHTS10], rotation=65)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Geometric Mean Speedup")
        caption(ax, i, f"{ctext}, {ptext}")
    legend(fig, [label(w) for w in WEIGHTS10], PALETTE)
    save(fig, out, "fig04")


def fig05(data, out):
    runtimes = ssb(data)
    fig, axes = grid(2, 3, 8, 3.8)
    shown = set()
    for i, ((prefetch, ptext), (config, ctext)) in enumerate(
        (p, c) for p in PREFETCH for c in CONFIGS
    ):
        ax = axes[i // 3][i % 3]
        best = max(
            WEIGHTS10[1:-1],
            key=lambda w: speedup(runtimes, (config, prefetch), w),
        )
        base = runtimes[config, prefetch, (1, 0)]
        queries = list(base)
        for j, w in enumerate([(1, 0), best, (0, 1)]):
            shown.add(w)
            ax.bar(
                [q + (j - 1) * 0.27 for q in range(len(queries))],
                [runtimes[config, prefetch, w][q] / base[q] for q in queries],
                width=0.27,
                color=PALETTE[WEIGHTS10.index(w)],
            )
        ax.set_xticks(range(len(queries)), queries, fontsize=6)
        ax.set_ylabel("Normalized Runtime")
        caption(ax, i, f"{ctext}, {ptext}")
    order = [w for w in WEIGHTS10 if w in shown]
    legend(
        fig,
        [label(w) for w in order],
        [PALETTE[WEIGHTS10.index(w)] for w in order],
    )
    save(fig, out, "fig05")


def weight_curves(ax, rows, placements, value):
    for (config, _), color in zip(CONFIGS, VIRIDIS):
        y = [value(rows, config, w) for w in placements]
        if None in y:
            continue
        y = [v / y[0] for v in y]
        ax.plot(range(len(y)), y, marker="o", markersize=2, color=color)
        ax.axvline(y.index(max(y)), linestyle="--", linewidth=0.8, color=color)
    ax.set_xticks(
        range(len(placements)),
        [label(w) for w in placements],
        rotation=45,
        fontsize=6,
    )
    ax.set_ylabel("Normalized Throughput")


def weights_figure(data, out, name, path, panels, placements, key):
    rows = {}
    for r in read(data / path):
        rows[r["config"], r["prefetch"], weights(r), r[key]] = float(r["gibps"])
    fig, axes = grid(2, 2, 7, 3.6)
    for i, (prefetch, kind, text) in enumerate(panels):
        ax = axes[i // 2][i % 2]
        weight_curves(
            ax,
            rows,
            placements,
            lambda rs, c, w: rs.get((c, prefetch, w, kind)),
        )
        caption(ax, i, text)
    legend(fig, [n for _, n in CONFIGS], VIRIDIS)
    save(fig, out, name)


def fig06(data, out):
    weights_figure(
        data,
        out,
        "fig06",
        "micro/sequential.csv",
        [
            ("on", "read", "Sequential Read, HW prefetching enabled"),
            ("on", "write", "Sequential Write, HW prefetching enabled"),
            ("off", "read", "Sequential Read, HW prefetching disabled"),
            ("off", "write", "Sequential Write, HW prefetching disabled"),
        ],
        WEIGHTS20,
        "kernel",
    )


def fig07(data, out):
    weights_figure(
        data,
        out,
        "fig07",
        "micro/join.csv",
        [
            ("on", "build", "Build Phase, HW prefetching enabled"),
            ("on", "probe", "Probe Phase, HW prefetching enabled"),
            ("off", "build", "Build Phase, HW prefetching disabled"),
            ("off", "probe", "Probe Phase, HW prefetching disabled"),
        ],
        WEIGHTS10,
        "phase",
    )


def sweep_figure(data, out, name, path, key):
    rows = defaultdict(dict)
    for r in read(data / path):
        rows[r["config"], r["prefetch"], weights(r)][int(r[key])] = float(
            r["gibps"]
        )
    fig, axes = grid(2, 3, 8, 4)
    for i, ((prefetch, ptext), (config, ctext)) in enumerate(
        (p, c) for p in PREFETCH for c in CONFIGS
    ):
        ax = axes[i // 3][i % 3]
        base = rows.get((config, prefetch, (1, 0)), {})
        for w, color in zip(WEIGHTS10, PALETTE):
            series = rows.get((config, prefetch, w), {})
            xs = [x for x in sorted(series) if x in base]
            ax.plot(
                xs,
                [series[x] / base[x] for x in xs],
                marker="o",
                markersize=2,
                color=color,
            )
        ax.set_xscale("log", base=2)
        ax.set_ylabel("Normalized Throughput")
        caption(ax, i, f"{ctext}, {ptext}")
    legend(fig, [label(w) for w in WEIGHTS10], PALETTE)
    save(fig, out, name)


def fig08(data, out):
    sweep_figure(data, out, "fig08", "micro/filter.csv", "selectivity")


def fig09(data, out):
    sweep_figure(data, out, "fig09", "micro/partition.csv", "fanout")


def fig10(data, out):
    rows = defaultdict(dict)
    for r in read(data / "micro/scalability.csv"):
        w = weights(r)
        memory = {(1, 0): "DRAM", (0, 1): "CXL"}.get(w, "DRAM:CXL")
        rows[r["config"], r["prefetch"], memory, r["kernel"]][
            int(r["threads"])
        ] = float(r["gibps"])
    lines = [
        ("DRAM", "read"),
        ("DRAM", "simd_read"),
        ("CXL", "read"),
        ("CXL", "simd_read"),
        ("DRAM:CXL", "read"),
        ("DRAM:CXL", "simd_read"),
    ]
    colors = [
        PALETTE[0],
        PALETTE[2],
        PALETTE[1],
        PALETTE[3],
        PALETTE[4],
        PALETTE[5],
    ]
    names = [f"{m} {'Scalar' if k == 'read' else 'SIMD'}" for m, k in lines]
    fig, axes = grid(2, 3, 8, 4.2)
    for i, ((prefetch, ptext), (config, ctext)) in enumerate(
        (p, c) for p in PREFETCH for c in CONFIGS
    ):
        ax = axes[i // 3][i % 3]
        for (memory, kernel), color in zip(lines, colors):
            series = rows.get((config, prefetch, memory, kernel))
            if series:
                xs = sorted(series)
                ax.plot(
                    xs,
                    [series[x] for x in xs],
                    marker="o",
                    markersize=1.5,
                    linewidth=0.8,
                    color=color,
                )
        ax.axvspan(
            48.5,
            96.5,
            facecolor="none",
            edgecolor="grey",
            hatch="//",
            linewidth=0,
        )
        ax.text(50, ax.get_ylim()[1] * 0.05, "Hyper-Threads", color="grey")
        ax.set_xticks([1] + list(range(8, 97, 8)))
        ax.tick_params(axis="x", labelsize=6)
        ax.set_ylabel("Throughput, GiB/s")
        caption(ax, i, f"{ctext}, {ptext}")
    legend(fig, names, colors)
    save(fig, out, "fig10")


# Input files of each figure; a figure is skipped when one is missing.
FIGURES = {
    fig01: ["hardware/dram_bandwidth_per_core.csv"],
    fig03: ["mlc/loaded_latency.csv", "hardware/idle_latency.csv"],
    fig04: ["ssb/runtimes.csv"],
    fig05: ["ssb/runtimes.csv"],
    fig06: ["micro/sequential.csv"],
    fig07: ["micro/join.csv"],
    fig08: ["micro/filter.csv"],
    fig09: ["micro/partition.csv"],
    fig10: ["micro/scalability.csv"],
}


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for figure, inputs in FIGURES.items():
        if all((data / f).exists() for f in inputs):
            figure(data, out)


if __name__ == "__main__":
    main()
