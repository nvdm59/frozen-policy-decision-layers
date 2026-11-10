"""Step 2 feasibility: does π0.5 follow an instruction SWITCH after a verified grasp?

Phase A: a TRAINED instruction that grasps the right object in this scene.
Switch trigger: object lifted >= LIFT_M above its settled height (sim state).
Phase B variants:
  B_new   = the changed (LIBERO-PRO) instruction itself
  B_dest  = a TRAINED instruction whose destination matches the goal
Same 300-step budget as baseline, same init trials as P0 (paired). Success = bddl goal (env done).
Kill rule (BRIEF): if switched instruction is ignored in >80% of dev trials, stop.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_runner as pr  # noqa: E402  (sets LIBERO_CONFIG_PATH + safe init loader)

LIFT_M = 0.05
# task_id in libero_goal_task -> (object, phase A trained instr, {variant: phase B instr})
PROBES = {
    2: ("wine_bottle_1", "put the wine bottle on top of the cabinet",
        {"B_new": "put the wine bottle in the bowl", "B_dest": "put the cream cheese in the bowl"}),
    6: ("wine_bottle_1", "put the wine bottle on top of the cabinet",
        {"B_new": "put the wine bottle in the bowl", "B_dest": "put the cream cheese in the bowl"}),
    8: ("wine_bottle_1", "put the wine bottle on top of the cabinet",
        {"B_new": "Put the wine bottle on the plate", "B_dest": "put the bowl on the plate"}),
    9: ("cream_cheese_1", "put the cream cheese in the bowl",
        {"B_new": "Put the cream cheese on the rack", "B_dest": "put the wine bottle on the rack"}),
}


def obj_z(env, name):
    e = env.env if hasattr(env, "env") else env
    return float(e.sim.data.body_xpos[e.obj_body_id[name]][2])


def run_switch(env, policy, obj, instr_a, instr_b, max_steps):
    obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()
    plan = collections.deque()
    t, z0, switched_at, prompt = 0, None, None, instr_a
    while t < max_steps + pr.SETTLE:
        if t < pr.SETTLE:
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            t += 1
            if t == pr.SETTLE:
                z0 = obj_z(env, obj)
            continue
        if switched_at is None and obj_z(env, obj) - z0 >= LIFT_M:
            switched_at = t - pr.SETTLE
            prompt = instr_b
            plan.clear()                      # replan immediately under the new instruction
        if not plan:
            plan.extend(policy.infer(pr.build_obs(obs, prompt))["actions"][:pr.REPLAN])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            return True, t - pr.SETTLE, switched_at
        t += 1
    return False, t - pr.SETTLE, switched_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="p2_switch")
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    out_dir = os.path.join(pr.HERE, "results", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"w{args.worker}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["task_id"], r["variant"], r["trial"]) for r in map(json.loads, open(out))}
    suite = benchmark.get_benchmark_dict()["libero_goal_task"]()
    jobs = [(t, v, k) for t in PROBES for v in PROBES[t][2] for k in range(args.trials)]
    jobs = [j for i, j in enumerate(jobs) if i % args.workers == args.worker and j not in done]
    print(f"switch probe: {len(jobs)} episodes (worker {args.worker})", flush=True)
    if not jobs:
        return
    policy = pr.Policy(pr.HOST)
    env, cur = None, None
    for task_id, variant, trial in jobs:
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        if cur != task_id:
            if env is not None:
                env.close()
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
            env.seed(7)
            inits = suite.get_task_init_states(task_id)
            cur = task_id
        env.reset()
        env.set_init_state(inits[trial % len(inits)])
        obj, instr_a, vb = PROBES[task_id]
        t0 = time.time()
        ok, steps, sw = run_switch(env, policy, obj, instr_a, vb[variant], 300)
        rec = {"task_id": task_id, "variant": variant, "trial": trial,
               "goal_prompt": pr.bddl_language(bddl), "instr_a": instr_a, "instr_b": vb[variant],
               "success": ok, "steps": steps, "switched_at": sw, "seconds": round(time.time() - t0, 1)}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[w{args.worker}] t{task_id} {variant} k{trial} {'OK ' if ok else 'FAIL'} "
              f"switch@{sw} steps={steps}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
