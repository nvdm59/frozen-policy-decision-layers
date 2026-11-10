"""Generate the P3 FINAL TABLE journal block from all arm summaries.

Usage: python3 finalize_p3.py [--append]   (--append writes to JOURNAL.md)
"""
import json
import pathlib
import sys
import time
from math import comb, sqrt

ROOM = pathlib.Path.home() / "cleanroom_calvin"
R = ROOM / "results"

P1_OBSERVED = [3, 10, 16, 19, 25, 27, 33, 34, 37, 38, 44, 45, 47, 48, 50,
               52, 54, 57, 76, 81, 96, 98, 103, 112, 115, 120, 130, 131,
               136, 138, 141, 143, 146, 148, 159, 171, 175, 180, 181, 183,
               193, 196]
CORE = [19, 25, 33, 37, 38, 45, 47, 52, 96, 98, 112, 115, 120, 130, 131,
        138, 141, 148, 180, 181, 183]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    p = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(tag, arm):
    f = (R / "p2_it2_layer_summary.json"
         if (tag == "p2_it1" and arm == "layer")
         else R / f"{tag}_{arm}_summary.json")
    d = json.load(open(f))
    return {r["chain_idx"]: r["n_success"] for r in d["records"]}


def stats(ns):
    n = len(ns)
    avg = sum(ns) / n
    var = sum((x - avg) ** 2 for x in ns) / max(n - 1, 1)
    return avg, sqrt(var / n), sum(1 for x in ns if x == 5) / n


def block():
    L = []
    L.append("## P3 FINAL TABLE (frozen layer v1.1 vs matched controls; "
             + time.strftime("%Y-%m-%d %H:%M") + ")")
    L.append("")
    L.append("Policy: FLOWER VLA (mbreuss/flower_calvin_abc), frozen. "
             "Env: CALVIN ABC->D, vendor eval semantics, llvmpipe forced "
             "across all arms. Budget: 2 extra rollout-attempt tokens per "
             "chain (control: blind retries; layer: precondition recoveries "
             "+ retries from the same pool). Same per-chain seeds across "
             "arms of each split.")
    L.append("")
    for tag, label, seeds in (
            ("p2_it1", "DEV core-21 (P1 systematic core; tuning set)",
             "510000+ci"),
            ("p3_val", "VAL-200 (frozen before val)", "600000+ci"),
            ("p3_heldout", "HELDOUT-600 (one shot)", "700000+ci")):
        try:
            arms = {a: load(tag, a) for a in ("baseline", "control", "layer")}
        except FileNotFoundError:
            L.append(f"### {label}: NOT AVAILABLE")
            L.append("")
            continue
        cis = sorted(arms["layer"])
        L.append(f"### {label} — seeds {seeds}, n={len(cis)}")
        L.append("")
        L.append("| arm | avg seq len ± SE | full-chain 5/5 |")
        L.append("|---|---|---|")
        for a in ("baseline", "control", "layer"):
            avg, se, full = stats([arms[a][ci] for ci in cis])
            L.append(f"| {a} | {avg:.3f} ± {se:.3f} | {full * 100:.1f}% |")
        b = sum(1 for ci in cis if arms["layer"][ci] > arms["control"][ci])
        c = sum(1 for ci in cis if arms["control"][ci] > arms["layer"][ci])
        L.append("")
        L.append(f"Layer vs control: paired wins {b} / losses {c}, "
                 f"exact McNemar p = {mcnemar_exact(b, c):.2e}.")
        clean = [ci for ci in cis if ci not in P1_OBSERVED]
        if 0 < len(clean) < len(cis):
            L.append(f"Untainted subset (excl. all 42 P1/P2-observed "
                     f"chains; n={len(clean)}):")
            for a in ("baseline", "control", "layer"):
                avg, se, full = stats([arms[a][ci] for ci in clean])
                L.append(f"- {a}: {avg:.3f} ± {se:.3f}, 5/5 "
                         f"{full * 100:.1f}%")
            b2 = sum(1 for ci in clean
                     if arms["layer"][ci] > arms["control"][ci])
            c2 = sum(1 for ci in clean
                     if arms["control"][ci] > arms["layer"][ci])
            L.append(f"- layer vs control {b2}-{c2}, "
                     f"p = {mcnemar_exact(b2, c2):.2e}")
        L.append("")

    # mechanism profile (val + heldout if present)
    for tag, lab in (("p3_val", "val"), ("p3_heldout", "heldout")):
        try:
            recs = json.load(open(R / f"{tag}_layer_summary.json"))["records"]
            crecs = json.load(open(R / f"{tag}_control_summary.json"))["records"]
        except FileNotFoundError:
            continue
        from collections import Counter
        rt, ft, ab = Counter(), Counter(), Counter()
        chains_i = 0
        for r in recs:
            evs = [e for e in r["events"] if e["type"] == "recovery"]
            chains_i += bool(evs)
            for e in evs:
                rt[e["recovery"]] += 1
                ft[e["recovery"]] += bool(e["fixed"])
            for e in r["events"]:
                if e["type"] == "attempt" and e.get("status") == "aborted":
                    ab[e["subtask"]] += 1
        es_l = sum(r["env_steps"] for r in recs)
        es_c = sum(r["env_steps"] for r in crecs)
        L.append(f"Mechanism ({lab}): interventions on {chains_i}/"
                 f"{len(recs)} chains; recoveries fixed/total: "
                 + ", ".join(f"{k} {ft[k]}/{v}" for k, v in rt.items())
                 + f"; doom-aborts {dict(ab)}; env-steps layer {es_l} vs "
                   f"control {es_c} ({(es_l / es_c - 1) * 100:+.0f}%).")
        L.append("")

    spend = json.load(open(ROOM / "spend.json"))
    L.append(f"Spend: ${spend['total_usd']} of ${spend['cap_usd']:.0f} cap "
             f"(ledger: spend.json).")
    L.append("")
    L.append("Caveats: (1) same-seed arms are pseudo-paired — cuDNN "
             "nondeterminism gives a ~2-3% symmetric flip noise floor "
             "(journaled before heldout); exact McNemar is unaffected under "
             "H0. (2) The 42 P1-observed chains (incl. the 21-chain tuning "
             "core) are excluded in the 'untainted' rows; those rows are "
             "the generalization claim. (3) All arms llvmpipe, A10G, "
             "vendor oracle + eval loop untouched; layer interventions use "
             "only env.get_info()-derived state, no privileged resets. "
             "(4) Baseline reproduction of the published leaderboard "
             "number was P0 (4.525 vs 4.54, n=200).")
    return "\n".join(L)


if __name__ == "__main__":
    out = block()
    print(out)
    if "--append" in sys.argv:
        with open(ROOM / "JOURNAL.md", "a") as f:
            f.write("\n" + out + "\n")
        print("\n[appended to JOURNAL.md]")
