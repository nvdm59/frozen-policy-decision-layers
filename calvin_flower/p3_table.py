"""Build the P3 FINAL TABLE block from arm summaries.

Usage: python3 p3_table.py <split_tag> ...   e.g. p3_table.py p2_it1 p3_val
For each tag, expects <tag>_{baseline,control,layer}_summary.json in
~/cleanroom_calvin/results/. Layer for p2_it1 comes from p2_it2 (frozen
v1.1 on the same seeds).
"""
import json
import pathlib
import sys
from math import comb, sqrt

ROOM = pathlib.Path.home() / "cleanroom_calvin" / "results"

# tuning/observation-tainted chains (P1 re-rolled 42 P0-breaking chains;
# P2 iterated on the 21-chain core)
P1_OBSERVED = [3, 10, 16, 19, 25, 27, 33, 34, 37, 38, 44, 45, 47, 48, 50,
               52, 54, 57, 76, 81, 96, 98, 103, 112, 115, 120, 130, 131,
               136, 138, 141, 143, 146, 148, 159, 171, 175, 180, 181, 183,
               193, 196]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(tag, arm):
    # frozen layer for the dev tag lives under p2_it2
    if tag == "p2_it1" and arm == "layer":
        f = ROOM / "p2_it2_layer_summary.json"
    else:
        f = ROOM / f"{tag}_{arm}_summary.json"
    d = json.load(open(f))
    return {r["chain_idx"]: r["n_success"] for r in d["records"]}, \
        d["summary"].get("gpu_container_hours", 0)


def stats(ns):
    n = len(ns)
    avg = sum(ns) / n
    var = sum((x - avg) ** 2 for x in ns) / max(n - 1, 1)
    se = sqrt(var / n)
    full = sum(1 for x in ns if x == 5) / n
    return avg, se, full


def row(tag, label):
    arms = {}
    hours = 0
    for arm in ("baseline", "control", "layer"):
        arms[arm], h = load(tag, arm)
        hours += h
    cis = sorted(arms["layer"])
    out = [f"**{label}** (n={len(cis)}, paired seeds)"]
    for arm in ("baseline", "control", "layer"):
        ns = [arms[arm][ci] for ci in cis]
        avg, se, full = stats(ns)
        out.append(f"| {arm} | {avg:.3f} +- {se:.3f} | {full * 100:.1f}% |")
    b = sum(1 for ci in cis if arms["layer"][ci] > arms["control"][ci])
    c = sum(1 for ci in cis if arms["control"][ci] > arms["layer"][ci])
    p = mcnemar_exact(b, c)
    out.append(f"layer vs control: discordant {b}-{c}, exact McNemar "
               f"p={p:.2e}; gpu_hours={hours:.2f}")
    # untainted subset (exclude P1/P2-observed chains)
    clean = [ci for ci in cis if ci not in P1_OBSERVED]
    if len(clean) != len(cis) and clean:
        for arm in ("baseline", "control", "layer"):
            ns = [arms[arm][ci] for ci in clean]
            avg, se, full = stats(ns)
            out.append(f"|   {arm} (untainted n={len(clean)}) | "
                       f"{avg:.3f} +- {se:.3f} | {full * 100:.1f}% |")
        b2 = sum(1 for ci in clean if arms["layer"][ci] > arms["control"][ci])
        c2 = sum(1 for ci in clean if arms["control"][ci] > arms["layer"][ci])
        out.append(f"untainted layer vs control: {b2}-{c2}, "
                   f"p={mcnemar_exact(b2, c2):.2e}")
    return "\n".join(out)


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        label = {"p2_it1": "dev core-21", "p3_val": "val-200",
                 "p3_heldout": "heldout-600"}.get(tag, tag)
        try:
            print(row(tag, label))
        except FileNotFoundError as e:
            print(f"{label}: missing {e.filename}")
        print()
