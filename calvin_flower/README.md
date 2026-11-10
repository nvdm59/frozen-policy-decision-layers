# FLOWER on CALVIN: a training-free decision layer between instructions

**Policy.** FLOWER VLA (`mbreuss/flower_calvin_abc`, arXiv 2509.04996), frozen.

**Benchmark.** CALVIN ABC→D long-horizon evaluation: 1,000 chains of five language instructions in a
row, with no reset in between. The score is the average number of instructions completed per chain
(0 to 5).

**Failure pattern found on the development split.** Many failures are not execution failures. The
next instruction is *unwinnable at the moment it is given*, because the previous step left the scene in
the wrong state. The typical case: the robot is still holding or touching the block it was meant to
let go of, so the next "push" or "lift" cannot count.

**The layer** (`modal_flower_p2.py`, arm `layer`): before each instruction it evaluates that
instruction's precondition on the simulator state (contacts, door and light states). If the
precondition is violated it runs a short recovery first (for example, release the block), then gives
the instruction. It has a budget of two extra attempts per chain.

**The control.** A blind-retry arm with the *same* budget of two extra attempts per chain, spent on
simply retrying the failed instruction. This separates "better decisions" from "more tries".

## Result: 600 held-out chains, one pass, identical seeds across arms

| Arm | Avg. instructions per chain | Chains completing 5/5 | Environment steps |
|---|---|---|---|
| FLOWER alone | 4.487 | 461 / 600 | 238,038 |
| + blind retries (matched budget) | 4.565 | 482 / 600 | 323,089 |
| **+ decision layer (matched budget)** | **4.732** | **523 / 600** | 265,764 |

- **Layer vs blind retries, full-chain success:** 55 wins / 14 losses, exact McNemar p = 6.9e-7.
- **Chains failing before step 5:** 139 for FLOWER alone, 77 with the layer (45% fewer).
- **Cost:** the layer uses 18% fewer environment steps than blind retries.
- **Reference:** FLOWER's published ABC→D score is 4.54 on the full 1,000 chains; our reproduction on
  the first 200 chains was 4.525.

These numbers are recomputed from the per-chain records in `results/heldout/`.

## Protocol

- **Splits fixed before any layer work** (`results/splits.json`): the 1,000 standard chains shuffled
  with seed 2026 into development 200 / validation 200 / held-out 600.
- **Development → validation → held-out.** The layer was iterated on development chains, checked on
  validation against a pre-set gate, frozen, and then run once on the held-out chains.
- **Paired design.** Every arm sees the same chain with the same seed (`700000 + chain index` on
  held-out).
- **Same renderer for all arms** (Mesa llvmpipe), so arms are comparable with each other.

## Limits

- **Simulation only, privileged state.** The layer reads contact and joint state from the simulator.
- **Slices of the standard protocol.** The held-out score is on 600 of the 1,000 standard chains, so
  it is not directly a leaderboard number.
- **Some held-out chains had been seen earlier.** The baseline reproduction ran chains 0–199 before
  the split existed; 114 of those fall in the held-out split (listed in `results/splits.json`). They
  were run with the policy alone, before the layer existed.
- **Scripted recoveries.** Recovery motions such as "open the gripper and back off" are scripted.
- **The working log is not included.** The protocol above and the final table are; the day-by-day
  log is kept privately.

## Files

| Path | What it is |
|---|---|
| `modal_flower_p2.py` | precondition checks, recoveries, and the three arms (`baseline`, `control`, `layer`) |
| `modal_flower_eval.py`, `modal_calvin_env.py`, `modal_fetch_flower.py` | Modal image, environment and checkpoint setup |
| `modal_flower_p1.py`, `modal_flower_p1debug.py` | failure taxonomy runs on the baseline |
| `modal_calvin_video.py` | replay with frame capture |
| `analyze_arms.py`, `finalize_p3.py`, `p3_table.py` | analysis and the final table |
| `results/heldout/{baseline,control,layer}.jsonl` | one line per held-out chain per arm |
| `results/p3_val_*`, `results/p2_*`, `results/p0_*`, `results/p1_*` | validation, development and baseline summaries |

Each held-out line: `chain_idx, arm, seed, budget, sequence, n_success, tokens_left, env_steps,
events`. `events` lists every attempt and every layer intervention.

Some scripts refer to local working paths (a `cleanroom_calvin` folder and a Modal volume named
`calvin-data`); adjust these to your setup.
