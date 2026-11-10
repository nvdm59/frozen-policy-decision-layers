# JOURNAL — π0.5 × LIBERO-PRO

## 2026-11-09 Step 1 (P0) setup
- Env: ~/claude-code/python/pi05_liberopro/.venv (isolated; July openpi_poc untouched). LIBERO-PRO vendored,
  registered via .pth (top-level libero/ has no __init__.py → setuptools mapped nothing).
  Isolated LIBERO config: .libero_pro/config.yaml via LIBERO_CONFIG_PATH (shared ~/.libero → July setup).
- LIBERO-PRO HF dataset (zhouxueyang/LIBERO-Pro) bddl+init files → vendor/.../bddl_files, init_files
  (existing original-suite init files NOT overwritten).
- SAFETY: third-party .pruned_init pickles loaded via safe_init.py (allowlisted Unpickler: numpy
  _reconstruct/ndarray/dtype/_codecs.encode only). Static pickletools scan: only numpy globals.
  torch weights_only path fails on protocol-4 FRAME opcode, hence the custom allowlist loader.
- TRAP FOUND: LIBERO-PRO registry returns ORIGINAL instruction for *_task suites; runner reads
  (:language …) from the bddl file. Changed goal tasks: bottom drawer; plate→stove; wine bottle→bowl (×2 scenes);
  cream cheese→top drawer; plate→top of cabinet; push cream cheese→stove front; turn OFF stove;
  wine bottle→plate; cream cheese→rack.
- PRE-REGISTERED EXCLUSION (before P0 results): task 7 "Turn off the stove" is degenerate — stove starts
  off (no Turnon in :init), goal satisfied at t=0 (smoke: OK at 44 steps). Excluded from all claims.
- Smoke (1 trial/task): 2/10 OK (t7 degenerate, t8 wine bottle→plate 137 steps). ~30 s/episode.
- P0 launched: libero_goal 10×10 + libero_goal_task 10×20, 4 workers each, caffeinated, per-episode resume.
- 08:35 relaunch: July server had max_containers=1 (7/8 workers stuck in connect). Deployed copy pi05-libero-pro-server (identical image → cached build, same weights), max_containers=8. 16 eps kept (resume).

## P0 RESULT (2026-11-09 ~09:30)
- libero_goal original: 91/100 = 0.91 (published 0.97; t4 6/10 drives the gap). Wording check: LIBERO-PRO
  libero_goal bddl language IDENTICAL to July LIBERO files (earlier "rewording" hypothesis was wrong).
- libero_goal_task (changed): 38/200 raw; EXCLUDING degenerate t7 (18/20): 20/180 = 0.111 (published 0.0).
  Per task: bottom drawer 0/20 · plate→stove 3/20 · wine→bowl 5/20 & 2/20 · cream cheese→top drawer 1/20 ·
  plate→cabinet top 3/20 · push cream cheese 1/20 · wine→plate 5/20 · cream cheese→rack 0/20.
- FAILURE MODE (contact sheet p0_fail_sheet.png): π0.5 replays the ORIGINAL memorized task of the scene:
  "bottom drawer" → opens MIDDLE drawer; "cream cheese in top drawer" → puts BOWL in top drawer;
  "plate on stove" → puts BOWL on stove; "wine bottle in bowl" → grabs the right object (wine bottle) but carries
  it to the memorized target (cabinet top). → decision-shaped (instruction→trajectory memorization), not motor.
- STEP 1 GATE: PASSED (reproduces original ~0.9; collapse on changed tasks; failures = memorized-task replay).
- Step 2 launched: switch_probe.py, tasks 2,6,8,9 × {B_new,B_dest} × 10 trials (paired with P0 trials 0-9), 8 workers.
- 2026-11-09 Step 2 STOPPED by user before any episode completed (no local multi-worker runs until user says so).
- 2026-11-09: ALL sim moved to Modal (user instruction: no local workers). modal_sim.py (CPU, OSMesa) + π0.5 GPU
  server max_containers=16. Cloud smoke OK (t0 FAIL 300 steps, 109 s incl. cold boot).
- PRE-REGISTERED SPLIT (before any held-out data): dev = libero_goal_task (P0/step 2 used it);
  HELD-OUT = libero_spatial_task, libero_object_task, libero_10_task. Layer may only be designed on dev.
- Launched: step 2 switch probe (4 shards) + step-3 π0.5-alone baselines on the 3 held-out suites (20 trials/task,
  4 shards each, videos) — 16 shards total, deployed+spawned.

## 2026-11-09 12:20 — Step 2 result (dev, goal_task, paired trials 0-9), 80/80 episodes
Grasp fired in 80/80. B_new success: t2 10/10 (P0 3/10), t6 7/10 (0/10), t8 10/10 (3/10), t9 0/10 (0/10).
B_dest: t2 9/10, t6 8/10, t8 7/10, t9 0/10. Kill rule (>80% ignored) NOT triggered -> proceed.
t9 (cream cheese -> rack) fails under both variants: switch does not rescue it (likely placement skill absent, not instruction-following).
Step 3 baselines (held-out, π0.5 alone) partial: spatial 113/200, object 20/200, libero_10 34/168 (running).
Note: several *_task prompts appear identical/near-identical to originals (e.g. spatial t5,t7,t9 = 20/20) -> must audit bddl vs original before counting as "changed".

## 2026-11-09 14:10 — Audit, layer v1, dev run launched
Audit: held-out *_task bddl vs July-original LIBERO: every task's goal object differs -> all genuinely changed; none excluded (spatial 20/20 tasks = changed instruction that happens to describe bowl_2's real location; π0.5 follows it).
Final π0.5-alone held-out baselines (p3_base, 20 trials/task): spatial 113/200, object 20/200, libero_10 38/200.
layer_v1.py (generic, no per-task tables): parse grasp targets from language + scene objects (+ sim poses for duplicate
instances) -> phase A "pick up the <target>" -> verified target lift switches to the changed instruction; wrong-object lift
-> release macro + retry (<=2); 150-step fallback; multi-target sequential. Parser validated vs bddl goal objects: 39/40
(the miss = push task, excluded by design). DISCLOSURE: lexicon/resolver fixed after viewing held-out instruction text and
checking parse vs goal objects (= target identity effectively oracle-grade); no held-out ROLLOUTS of the layer seen.
Arms: base | blind (phase A fixed 52 steps then switch, no checks) | layer. Same step budget.
Dev run p4_dev: goal_task, layer + blind, trials 0-9, launched 14:08.
Pre-registered go rule for held-out: dev layer > P0 base (McNemar p<0.05) AND layer >= blind on dev.

## 2026-11-09 14:32 — v1 failing on dev; v2 designed (dev only) and launched
v1 interim (96 eps): layer 0 wins / 1 loss vs P0; blind 7/1. Diagnosis: wine t2 verified grasp 9/10 but 3/10 success
(step-2 hand-picked TRAINED phase A gave 10/10) -> the ingredient is grasping under a trained same-scene instruction;
template "pick up the X" grasps don't finish; plate: π0.5 lifts memorized bowl regardless; retry/timeout burn budget.
v2 = phase A from train_pool.json (40 original LIBERO-4 instructions; same scene signature + first goal object == target;
put/pick only; highest P0-original reliability). No retries. No candidate -> base behaviour.
COVERAGE (offline, text-only): dev goal: t2,t3,t6,t8,t9 have candidates. HELD-OUT: spatial 0/10, object 0/10
(one training task per scene / target is the other bowl instance), libero_10: t0(1st obj), t1(1st obj), t4(2nd obj), t8.
=> v2 can only act on ~4 of 30 held-out tasks. Held-out test as pre-registered cannot show a large effect by construction.
Dev run p5_dev_v2: tasks 2,3,6,8,9 x trials 0-9, arms layer+blind (v2 candidates), launched 14:32.

## 2026-11-09 14:45 — v2 bug + relaunch; v1 dev nearly complete
p5_dev_v2 INVALID: scene signature built from env dicts != bddl signature -> no_candidate in 10/10 layer eps (== base).
Cancelled. Fixed: signature from env.parsed_problem (LIBERO's own bddl parse). Verified offline: wine -> "Put the wine
bottle on the top of the drawer", cream cheese -> "Put the cream cheese on the bowl", plate -> None. layer_v2 sha 12f78e5c.
Relaunched as p6_dev_v2 (tasks 2,3,6,8,9 x trials 0-9, layer+blind) 14:44.
v1 dev (p4_dev, 172/200): layer 10/76 vs base 8 (4 wins/2 losses, p=0.69) -> v1 FAILS go rule.
UNEXPECTED: v1 blind arm ("pick up the X" 52 steps, then switch, no checks) 21/80 vs base 8 (15 wins/2 losses, p=0.0023).
The verification machinery (retry/timeout) hurt; a fixed schedule helped. Treat as a dev observation, not a claim.

## 2026-11-09 15:10 — DESIGN AUDIT (user request). All shards stopped; nothing running.
Facts gathered:
- PI's only sim-evaluable released fine-tune is pi05_libero (README: 98.8/98.2/98.0/92.4, avg 96.85). pi0_libero / pi0_fast_libero
  exist only as training configs (base ckpts public, fine-tuned not released). DROID/ALOHA ckpts = real robot. => within "PI model
  + sim benchmark", pi05 x LIBERO(-variants) is the only option; standard LIBERO has no headroom.
- LIBERO-PRO registry returns the ORIGINAL instruction for *_task suites; openpi's eval uses task.language. Their published
  pi0.5 P1 = 0.0 on all suites is therefore likely an eval artifact (original prompt scored against new goal). Our bddl-prompt
  baselines: goal 0.11 (excl. degenerate t7), spatial 0.57, object 0.10, libero_10 0.19. Unverified what they ran; replicable.
- LIBERO-PRO P2 (position perturbation) uses ORIGINAL instructions -> no prompt artifact; published pi0.5 = 0.4/0.2/0.1/0.2.
- v3 smoke (1 ep each): goal t1 template FAIL (other object lifted @56); object t0 cross-scene picked a GOAL-scene prompt
  ("Put the cream cheese on the bowl") -> wrong, cap@60, FAIL; spatial t1 template OK (base already 18/20).
Verdict on P1 instruction-switch program: mechanism real on dev (27/30 vs 6/30 with trained same-scene grasp prompts), but a
generic layer is bounded by trained-prompt coverage, ~0 on spatial/object. Not a benchmark-beating path.
Recommendation: pivot to P2 (position perturbation) with the CALVIN recipe (precondition check + retry, matched budget,
proprioception-only signals), after a P0 failure-mode study. Keep P1 switch result as a dev-level mechanism finding.

## 2026-11-09 — PIVOT to P2 (position perturbation), user-approved
P2 = *_swap suites: objects/fixtures exchange init regions; instructions identical to original LIBERO (0/40 language lines
differ) -> no prompt artifact. Published π0.5: goal 0.4? (README order spatial/object/goal/10 = 0.4/0.2/0.1/0.2 — verify column order).
SPLIT (pre-registered, same as P1): dev = libero_goal_swap; HELD-OUT = libero_spatial_swap, libero_object_swap, libero_10_swap.
No held-out swap episode has been run.
Failure-mode study: diag_runner.py (π0.5 alone, per-step trace of eef/gripper/all object poses + video).
Smoke 1 episode OK. Launched p2_dev_base: goal_swap x 20 trials x 10 tasks, 4 shards.

## 2026-11-09 16:10 — RETRACTION: p2_dev_base object traces invalid
diag_runner appended live VIEWS of sim.data.body_xpos -> every row equals the final state -> all lift/moved/nearest-object
measures were zero/garbage. The "empty grasp, nothing moved (72/72)" breakdown reported to the user is WRONG.
Success counts, videos, eef and gripper traces from p2_dev_base remain valid.
Video sheet (t2,t4,t0,t5 k0): π0.5 DOES grasp the right object (wine bottle, bowl) and carries it to the MEMORIZED
destination location (now occupied by the rack, since cabinet<->rack swapped). Plate task: plate lifted onto stove.
Fix: np.array copy + fixture poses. Verified on 1 real episode (wine bottle max lift 0.332). Relaunched as p2_dev_trace
(10 trials x 10 tasks). Two-arm retry/reposition test NOT launched: premise (empty grasp) was false.

## 2026-11-09 — P2 dev repositioning test, final (tag p2_dev_repos, 80 episodes, libero_goal_swap = DEV)
| task | π0.5 alone | post (dest correction) | both (+pre-position) |
|---|---|---|---|
| t1 bowl→stove (harm check) | 13/20 | 8/10 | 3/10 |
| t2 wine→top of drawer | 0/20 | 0/10 (carried 9) | 0/10 (carried 3) |
| t4 bowl→top of drawer | 0/20 | 10/10 | 10/10 |
| t6 cream cheese→bowl | 0/15 | 0/10 (never lifted) | 2/10 (carried 10; raise+carry hit the 60-step cap) |
- t4: destination correction fully rescues (20/20 vs 0/20).
- t2: video (post k1) shows the bottle knocked against the cabinet side during the carry. Cause = layer bug: carry height
  was rest+0.30 applied to the GRIPPER, so a tall object hanging below the fingers does not clear the cabinet.
- pre-positioning HURTS a working task (t1 3/10 vs 8/10): drop it from the candidate design.
- Fix (p2_runner.py): raise until the OBJECT centre is at rest+CARRY_H. Rerun post arm t1,t2,t4 ×10 as tag p2_dev_repos2 (dev).

## 2026-11-09 — P2 dev rerun with object-referenced carry height (tag p2_dev_repos2, post arm, 30 episodes, DEV)
t1 bowl→stove 9/10 (alone 13/20) | t2 wine→top of drawer 9/10 (alone 0/20) | t4 bowl→top of drawer 9/10 (alone 0/20). Total 27/30.
t2's one failure: bottle never lifted (no carry fired). t4's one failure: raise hit the 60-step cap.
Candidate design = post arm only (no pre-position). Not yet frozen; held-out swap suites still untouched.

## 2026-11-09 — P2 design candidate + dev gate (user said GO for held-out)
Published π0.5 Position-perturbation numbers (LIBERO-PRO GitHub leaderboard, 50 ep/task): Goal 0.38, Spatial 0.20, LIBERO-10 0.08, Object 0.17.
Our dev π0.5-alone on libero_goal_swap: 54/175 = 0.31 (consistent with 0.38).
Prompt source: task.language from the LIBERO-PRO registry (== the bddl wording for swap suites; verified by parse check p2_parse3).
Design changes made BEFORE any held-out policy episode (from instruction text + parse checks only, no held-out outcomes):
- dest = first on/in mention after the placing verb (was: last prepositioned mention); left/right plates resolved from sim poses.
- carry only for on-type destinations, or in-type destinations that are movable containers (basket). in-fixture (microwave, drawer) -> no carry.
- pre-positioning dropped. Arms: base (π0.5 alone, same runner) | post.
Parse check on held-out suites (no policy): spatial 10/10 -> plate; object 10/10 -> basket; libero_10: 5/10 tasks get a carry (t0,t1,t4,t6,t7), 5 untouched.
Dev gate running: tag p2_dev_gate, post arm, all 10 goal_swap tasks x 10 trials.

## 2026-11-09 — P2 dev iteration results, then FREEZE + PRE-REGISTRATION (written before any held-out swap policy episode)
Dev (libero_goal_swap), 10 trials/task:
- p2_dev_gate (post = hover hand-back, all 10 tasks): 47/100 vs π0.5 alone 54/175. t1 10/10, t2 5/10, t4 10/10, t8 4/10 (alone 9/15: hand-back above the plate makes π0.5 hesitate and drop badly).
- p2_dev_place (scripted lower-until-contact + release, tasks 1,2,4,8): t1 3/10 (stove body origin is not the cook region), t2 8/10, t4 9/10, t8 10/10.
- Rule chosen on dev: destination is a MOVABLE object -> lower+release; destination is a FIXTURE -> hover hand-back. Smoke (p2_smoke_layer) OK.
FROZEN files (sha256):
  p2_runner.py 5c1b59384c1a723e1bd096ced28c343911e6657024139f803d30dbe62c6e4610
  layer_v1.py 6c0dd4c61030f261c6633221fcda9fb0ff6a8d386c62c1f9488379500070a92e
  pro_runner.py e39caf538b688456f52bb4c34a19fef1881032911b878164e357e97d5e8b547a
  modal_sim.py 182be9dd69c92933a94eac306338c16d844d417c7eae0f76de26259bd946487a
PRE-REGISTRATION
- Held-out suites: libero_spatial_swap, libero_object_swap, libero_10_swap (no policy episode run on them before this entry; parse-only checks were run).
- Arms: base (frozen π0.5 pi05_libero alone) vs layer (same checkpoint + p2_runner arm=layer). Same runner, same prompt (task.language),
  same init states (trial k -> init state k), same step budget (220/280/520; all layer macro steps count), 10 settle steps, replan 5, env seed 7.
- 10 tasks x 10 trials (init states 0-9) per suite per arm = 600 episodes. Tag p2_heldout.
- PRIMARY endpoint: pooled success over the 3 suites, layer vs base, exact two-sided McNemar on pairs (suite, task, trial), alpha 0.05.
- Secondary: per-suite McNemar; per-task table; fraction of layer episodes where a carry fired; comparison with published π0.5 Pos (Spatial 0.20, LIBERO-10 0.08, Object 0.17; 50 ep/task).
- No changes to the frozen files after this entry. Infrastructure failures (crashed shard) are resumed, never re-rolled; every episode written is kept.
- Disclosures for any write-up: layer uses simulator object/fixture positions (privileged); it scripts carry and, for movable destinations, the release;
  design was tuned on libero_goal_swap (dev), which is excluded from the claim.

## 2026-11-10 — P2 HELD-OUT RESULT (tag p2_heldout, 600 episodes, 0 duplicates, frozen hashes unchanged)
PRIMARY (pooled, 300 pairs): π0.5 alone 70/300 (23.3%) vs +layer 110/300 (36.7%); wins 46 / losses 6; exact McNemar p = 1.0e-8. PASS.
Secondary: spatial_swap 47 -> 77 /100 (32/2, p=6.9e-8); object_swap 18 -> 27 /100 (11/2, p=0.022); libero_10_swap 5 -> 6 /100 (3/2, p=1.0).
Notes:
- Shard failures (19 of 44 on first launch, 3 on second) were startup failures under load; resumed, no episode re-rolled. Error text not captured.
- Prompt = task.language = standard LIBERO wording. The earlier parse check (p2_parse3) showing "Pick the akita black bowl" ran on a stale deployment.
- Layer never fired on spatial t3, t4 and on object t2,t4,t5,t7,t8 (target never lifted >= 0.05) -> those are π0.5 grasp failures the layer does not address.
- libero_10 t4/t6: carry fired 10/10, still 0/10 (two-step tasks; layer handles only the first object).
- Our π0.5-alone numbers vs published (50 ep/task): spatial 0.47 vs 0.20, object 0.18 vs 0.17, libero_10 0.05 vs 0.08. Spatial gap unexplained.
- Summary table: results/p2_heldout_summary.txt

## 2026-11-10 — Spatial at 50 trials/task (user-approved extension, declared before the layer run)
- p2_spatial_base50: π0.5 alone, 50 trials/task (475/500 in at time of writing: 217/475).
- p2_spatial_layer50: launched now with the frozen p2_runner (sha 5c1b5938…), arm=layer, 50 trials/task, same init states as base50.
- Analysis fixed in advance: exact McNemar on pairs (task, trial) between base50 and layer50. This is a protocol-matched follow-up;
  the pre-registered primary result (110/300 vs 70/300) stands as is and is not replaced.
- Gap vs published spatial 0.20: LIBERO-PRO SwapPerturbator uses unseeded random.choice of the swap partner; HF swap files (uploaded 2025-11-05,
  unchanged since) are one draw. Our per-partner baseline: cookies 161/190, bowl_2 52/140, ramekin 2/90, cabinet 0/48. Plausible, unconfirmed.
- p2_spatial_bddlwording: 100-episode baseline with the bddl wording, to size the wording effect (upstream uses filename wording = ours).
- RESULT p2_spatial_base50: 226/500 = 45.2% (per task 49,14,50,0,20,42,0,2,49,0).
- RESULT p2_spatial_bddlwording ("Pick the akita black bowl ..."): 52/100 (per task 10,2,10,0,9,10,0,1,10,0). Wording does NOT explain the gap with published 0.20.

## 2026-11-10 — SPATIAL 50-TRIAL RESULT (p2_spatial_base50 vs p2_spatial_layer50, 500 pairs, frozen runner, all shards rc 0)
π0.5 alone 226/500 (45.2%) vs +layer 392/500 (78.4%); wins 185 / losses 19; exact McNemar p = 2.3e-35.
Per task (base -> layer, of 50): t0 49->50, t1 14->42, t2 50->47, t3 0->0 (never lifted), t4 20->18 (layer never fires), t5 42->49,
t6 0->49, t7 2->49, t8 49->49, t9 0->39. Summary: results/p2_spatial50_summary.txt
