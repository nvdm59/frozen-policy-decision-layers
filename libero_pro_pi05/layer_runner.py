"""Arms runner: base | blind | layer (layer_v1) on LIBERO-PRO *_task suites. Same protocol as pro_runner
(bddl prompt, 10 settle steps, replan 5, suite step budget). Resumable per episode.
  --parse_only : build each task's env, print parsed targets (no policy calls)."""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_runner as pr  # noqa: E402
L = __import__(os.environ.get("LAYER", "layer_v1"))  # noqa: E402

NOOP = [0.0] * 6 + [-1.0]


def run(env, policy, instr, max_steps, arm, video=None):
    import imageio
    import numpy as np
    obs = None
    for _ in range(pr.SETTLE):
        obs, _, _, _ = env.step(NOOP)
    lay = L.Layer(env, instr, arm)
    plan, macro, frames = collections.deque(), collections.deque(), []
    t, last, ok = 0, None, False
    while t < max_steps:
        if video is not None:
            frames.append(obs["agentview_image"][::-1, ::-1])
        if not macro:
            changed, mac = lay.step(t, env, last)
            if changed:
                plan.clear()
            if mac:
                macro.extend(mac)
        if macro:
            a = macro.popleft()
            if not macro:
                lay.rebaseline(env)
        else:
            if not plan:
                plan.extend(policy.infer(pr.build_obs(obs, lay.prompt()))["actions"][:pr.REPLAN])
            a = plan.popleft().tolist()
        obs, _, done, _ = env.step(a)
        last = a
        t += 1
        if done:
            ok = True
            break
    if video is not None and frames:
        imageio.mimwrite(video, [np.asarray(f) for f in frames], fps=30)
    return ok, t, lay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--arm", required=True, choices=["base", "blind", "layer"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--trial0", type=int, default=0)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--parse_only", action="store_true")
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.parse_only:
        for tid in range(suite.n_tasks):
            task = suite.get_task(tid)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
            env.seed(7)
            env.reset()
            env.set_init_state(suite.get_task_init_states(tid)[0])
            for _ in range(pr.SETTLE):
                env.step(NOOP)
            instr = pr.bddl_language(bddl)
            goal = open(bddl).read().split("(:goal")[1].split("\n")[1].strip()
            print(json.dumps({"suite": args.suite, "task": tid, "instr": instr,
                              "targets": L.parse_targets(env, instr), "goal": goal,
                              "movable": list(env.env.objects_dict)}), flush=True)
            env.close()
        return

    out_dir = os.path.join(pr.HERE, "results", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.suite}_{args.arm}_w{args.worker}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["task_id"], r["trial"]) for r in map(json.loads, open(out))}
    jobs = [(t, k) for t in range(suite.n_tasks) for k in range(args.trial0, args.trial0 + args.trials)]
    if os.environ.get("TASKS"):
        keep = {int(x) for x in os.environ["TASKS"].split(",")}
        jobs = [j for j in jobs if j[0] in keep]
    jobs = [j for i, j in enumerate(jobs) if i % args.workers == args.worker and j not in done]
    print(f"{args.suite}/{args.arm}: {len(jobs)} episodes (worker {args.worker})", flush=True)
    if not jobs:
        return
    policy = pr.Policy(pr.HOST)
    max_steps = pr.MAX_STEPS[pr.base_suite(args.suite)]
    env, cur = None, None
    for tid, trial in jobs:
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        if cur != tid:
            if env is not None:
                env.close()
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
            env.seed(args.seed)
            inits = suite.get_task_init_states(tid)
            cur = tid
        env.reset()
        env.set_init_state(inits[trial % len(inits)])
        instr = pr.bddl_language(bddl)
        vid = os.path.join(out_dir, f"{args.suite}_{args.arm}_t{tid}_k{trial}.mp4") if args.videos else None
        t0 = time.time()
        ok, steps, lay = run(env, policy, instr, max_steps, args.arm, vid)
        rec = {"suite": args.suite, "arm": args.arm, "task_id": tid, "trial": trial, "prompt": instr,
               "targets": lay.targets, "layer": L.__name__, "cands": getattr(lay, "cands", None), "success": ok, "steps": steps, "events": lay.events,
               "seconds": round(time.time() - t0, 1)}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{args.arm} w{args.worker}] t{tid} k{trial} {'OK ' if ok else 'FAIL'} {steps} {lay.events}",
              flush=True)
    env.close()


if __name__ == "__main__":
    main()
