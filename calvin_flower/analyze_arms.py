"""Paired per-chain comparison of two arm summaries (layer vs control).

Usage: python3 analyze_arms.py <tag> [baseline_summary.json]
Reads ~/cleanroom_calvin/results/<tag>_{layer,control}_summary.json.
"""
import json
import pathlib
import sys
from math import comb

ROOM = pathlib.Path.home() / "cleanroom_calvin" / "results"


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "p2_it1"
    arms = {}
    for arm in ("layer", "control"):
        d = json.load(open(ROOM / f"{tag}_{arm}_summary.json"))
        arms[arm] = {r["chain_idx"]: r for r in d["records"]}
    base = None
    if len(sys.argv) > 2:
        d = json.load(open(ROOM / sys.argv[2]))
        base = {r["chain_idx"]: r["n_success"] for r in d["records"]}

    cis = sorted(arms["layer"])
    print(f"{'chain':>6} {'base':>4} {'ctrl':>4} {'layer':>5} "
          f"{'d(l-c)':>6}  events")
    tot = {"layer": 0, "control": 0, "base": 0}
    b_disc = c_disc = 0  # chains where layer>control / control>layer
    win5 = {"layer": 0, "control": 0}
    for ci in cis:
        L, C = arms["layer"][ci], arms["control"][ci]
        nl, nc = L["n_success"], C["n_success"]
        tot["layer"] += nl
        tot["control"] += nc
        win5["layer"] += nl == 5
        win5["control"] += nc == 5
        if nl > nc:
            b_disc += 1
        elif nc > nl:
            c_disc += 1
        evs = [e for e in L["events"] if e["type"] == "recovery"]
        ev_str = "; ".join(
            f"sub{e['sub_idx']}:{e['recovery']}"
            f"{'+' if e['fixed'] else '-'}" for e in evs)
        aborts = sum(1 for e in L["events"]
                     if e["type"] == "attempt" and e.get("status") == "aborted")
        if aborts:
            ev_str += f" [{aborts} doom-abort]"
        bnum = base.get(ci, "-") if base else "-"
        if base and ci in base:
            tot["base"] += base[ci]
        print(f"{ci:>6} {bnum:>4} {nc:>4} {nl:>5} {nl - nc:>+6}  {ev_str}")

    n = len(cis)
    print(f"\nn={n} chains")
    if base:
        print(f"baseline avg: {tot['base'] / n:.3f}")
    print(f"control avg:  {tot['control'] / n:.3f}  (5/5 on {win5['control']})")
    print(f"layer avg:    {tot['layer'] / n:.3f}  (5/5 on {win5['layer']})")
    print(f"discordant: layer-better {b_disc}, control-better {c_disc}")
    print(f"exact McNemar (chain n_success direction): "
          f"p={mcnemar_exact(b_disc, c_disc):.4f}")
    es_l = sum(arms['layer'][ci]['env_steps'] for ci in cis)
    es_c = sum(arms['control'][ci]['env_steps'] for ci in cis)
    print(f"env steps: layer {es_l}, control {es_c} "
          f"(ratio {es_l / max(es_c, 1):.2f})")


if __name__ == "__main__":
    main()
