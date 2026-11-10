# π0.5 on LIBERO-PRO position swap: a training-free position-correcting layer

**Policy.** Physical Intelligence's `pi05_libero` checkpoint from openpi, frozen. Served unchanged.

**Benchmark.** LIBERO-PRO (arXiv 2510.03827) position-perturbation suites (`*_swap`): the instruction is
the same as in LIBERO, but objects and fixtures have exchanged starting places. π0.5 scores ~97% on the
original suites and collapses on these.

**Failure pattern found on the development suite.** π0.5 usually grasps the right object and then
carries it to where the destination *used to be*.

**The layer** (`p2_runner.py`, arm `layer`; parsing helpers in `layer_v1.py`):

1. Parse the instruction for the object to grasp and the destination ("put X on Y", "pick up X … and
   place it on/in Y"), resolving duplicates such as "left plate" from positions.
2. Let π0.5 run unchanged until the target object is lifted 5 cm.
3. Raise the object to a clearance height and carry it above the destination's current position.
4. If the destination is a movable object (plate, basket, bowl): lower until the object stops
   descending, open the gripper, retreat. If it is a fixture (stove, cabinet): hand back to π0.5 from
   the hover.
5. Return control to π0.5 for the rest of the episode.

Every scripted step counts against the same step budget as the baseline. No training, no model calls
beyond the policy itself.

## Results

### Pre-registered held-out test (10 trials per task, 300 paired episodes)

The design was tuned on `libero_goal_swap` only, frozen with file hashes, and pre-registered in
[`JOURNAL.md`](JOURNAL.md) before any held-out episode was run.

| Held-out suite | π0.5 alone | π0.5 + layer | Wins / losses | Exact McNemar p |
|---|---|---|---|---|
| `libero_spatial_swap` | 47/100 | 77/100 | 32 / 2 | 7e-8 |
| `libero_object_swap` | 18/100 | 27/100 | 11 / 2 | 0.022 |
| `libero_10_swap` | 5/100 | 6/100 | 3 / 2 | 1.0 |
| **Pooled (primary endpoint)** | **70/300** | **110/300** | **46 / 6** | **1e-8** |

### Spatial suite at the benchmark's 50 trials per task (500 paired episodes)

A follow-up declared in the journal before the layer arm was run. Same frozen runner.

| Bowl location in the instruction | π0.5 alone (of 50) | π0.5 + layer (of 50) |
|---|---|---|
| next to the cookie box | 0 | 49 |
| on the stove | 2 | 49 |
| on the wooden cabinet | 0 | 39 |
| next to the ramekin | 14 | 42 |
| on the ramekin | 42 | 49 |
| between the plate and the ramekin | 49 | 50 |
| next to the plate | 49 | 49 |
| from table center | 50 | 47 |
| in the top drawer of the wooden cabinet | 20 | 18 |
| on the cookie box | 0 | 0 |
| **Total** | **226/500 (45.2%)** | **392/500 (78.4%)** |

185 wins / 19 losses, exact McNemar p = 2.3e-35.

## Limits

- **Privileged state.** The layer reads object and fixture positions from the simulator.
- **Scripted motion.** The carry and (for movable destinations) the release are scripted.
- **It only acts after a lift.** Where π0.5 never lifts the object (for example the bowl on the cookie
  box, and five of the ten object-suite tasks), the layer never fires.
- **No effect on `libero_10_swap`.** On the two-object tasks the layer handles only the first object.
- **Baseline differs from the LIBERO-PRO leaderboard on spatial.** π0.5 alone scores 45% here against
  a published 20%. The other suites agree (object 0.18 vs 0.17, LIBERO-10 0.05 vs 0.08, goal 0.31 vs
  0.38). One candidate explanation: LIBERO-PRO's swap generator picks the swap partner with an unseeded
  `random.choice`, the published `*_swap` files are one draw, and the baseline depends strongly on the
  partner (plate↔cookie box 85%, plate↔second bowl 37%, plate↔ramekin 2%, plate↔cabinet 0%). This is
  unconfirmed. A 100-episode check with the task-file wording gave 52%, so wording is not the cause.
  Quote the baseline measured here, on the files named below.
- **Instruction parsing on two tasks.** A parse check on the held-out suites was run against a stale
  deployment; the design was already frozen, so the affected tasks count as they fell.
- **An earlier analysis was retracted.** A trace bug (live views of simulator arrays) produced a wrong
  failure breakdown during the development study; the retraction and fix are in the journal.

## Protocol

- openpi LIBERO evaluation settings: 10 settle steps, replan every 5 actions, step budgets 220
  (spatial) / 280 (object) / 300 (goal) / 520 (LIBERO-10), environment seed 7, trial `k` uses the
  suite's init state `k`.
- Instructions are the registry's task language (standard LIBERO wording).
- LIBERO-PRO files: Hugging Face dataset `zhouxueyang/LIBERO-Pro` (swap files unchanged since their
  2025-11-05 upload); code at the upstream repository's `eafdb80`.

## Files

| Path | What it is |
|---|---|
| `p2_runner.py` | arms `base` and `layer` (plus development arms `post`, `place`, `both`, `parse`) |
| `layer_v1.py` | instruction parsing and object resolution used by the layer |
| `pro_runner.py` | π0.5-alone runner and shared helpers (observation building, policy client) |
| `modal_pi05_server.py`, `modal_sim.py` | Modal apps: GPU policy server and CPU simulation shards |
| `analyze_p2.py`, `analyze_spatial50.py` | the paired statistics above |
| `results/p2_heldout/` | pre-registered held-out run, both arms |
| `results/p2_spatial_base50/`, `results/p2_spatial_layer50/` | 50-trial spatial runs |
| `results/p2_spatial_bddlwording/` | wording check |
| `results/p2_dev_*/` | development runs on `libero_goal_swap` (not part of any claim) |
| `JOURNAL.md` | working log: development study, retraction, pre-registration, results |
| `layer_v2.py`, `layer_v3.py`, `layer_runner.py`, `switch_probe.py`, `build_pool.py` | an earlier line of work on the task-perturbation suites that did not pass its development gate; kept for the record |

Each result line is one episode: `suite, arm, task_id, trial, prompt, target, dest, success, steps,
events, final_lift`. `events` lists the layer's actions as `[name, end_step, n_steps]`.

**Hashes.** The journal records SHA-256 hashes of the frozen files. In this public copy one line of
`pro_runner.py` (the policy server address) and one line of `setup_env.sh` (a scratch path) were
replaced, so those two files no longer match the recorded hashes. `p2_runner.py` and `layer_v1.py`
are byte-identical to the frozen versions.

## Reproduce

```bash
# 1. policy server (A10G) and simulation image
modal deploy modal_pi05_server.py
modal deploy modal_sim.py
export PI05_HOST=wss://<your-modal-workspace>--pi05-libero-pro-server-policy-server.us-east.modal.run

# 2. one shard of the layer arm, 50 trials per task, 10 shards
python -c "import modal; f = modal.Function.from_name('pi05-liberopro-sim', 'p2_shard'); \
print(f.remote('libero_spatial_swap', 'layer', 50, 0, 10, 'my_run', ''))"

# 3. statistics
python analyze_spatial50.py
```

`setup_env.sh` builds the simulation environment (LIBERO-PRO checkout, benchmark files, pinned
requirements). Note that the Modal image bakes in `PI05_HOST` only if you set it in `modal_sim.py`.
