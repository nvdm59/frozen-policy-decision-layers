"""P2 failure-mode study: π0.5 alone on LIBERO-PRO position-perturbation (*_swap) suites, with a per-step
state trace (eef, gripper, every movable object's pose) so failure modes are computable offline.
Same protocol as pro_runner. Resumable per episode. Traces: results/<tag>/trace/<suite>_t<id>_k<trial>.npz"""
import argparse, collections, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_runner as pr  # noqa: E402

NOOP = [0.0] * 6 + [-1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()
    import imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    out_dir = os.path.join(pr.HERE, "results", args.tag)
    os.makedirs(os.path.join(out_dir, "trace"), exist_ok=True)
    out = os.path.join(out_dir, f"{args.suite}_w{args.worker}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["task_id"], r["trial"]) for r in map(json.loads, open(out))}
    suite = benchmark.get_benchmark_dict()[args.suite]()
    jobs = [(t, k) for t in range(suite.n_tasks) for k in range(args.trials)]
    if args.tasks:
        keep = {int(x) for x in args.tasks.split(",")}
        jobs = [j for j in jobs if j[0] in keep]
    jobs = [j for i, j in enumerate(jobs) if i % args.workers == args.worker and j not in done]
    print(f"{args.suite}: {len(jobs)} episodes (worker {args.worker})", flush=True)
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
            env.seed(7)
            inits = suite.get_task_init_states(tid)
            cur = tid
        env.reset()
        env.set_init_state(inits[trial % len(inits)])
        e = env.env
        names = list(e.objects_dict)
        fixn = list(e.fixtures_dict)
        prompt = pr.bddl_language(bddl)
        obs = None
        for _ in range(pr.SETTLE):
            obs, _, _, _ = env.step(NOOP)
        plan, frames, tr = collections.deque(), [], collections.defaultdict(list)
        ok, t, t0 = False, 0, time.time()
        while t < max_steps:
            frames.append(obs["agentview_image"][::-1, ::-1])
            if not plan:
                plan.extend(policy.infer(pr.build_obs(obs, prompt))["actions"][:pr.REPLAN])
            a = plan.popleft().tolist()
            tr["eef"].append(np.array(obs["robot0_eef_pos"])); tr["grip"].append(np.array(obs["robot0_gripper_qpos"]))
            tr["act"].append(a)
            tr["obj"].append(np.array([e.sim.data.body_xpos[e.obj_body_id[n]] for n in names]))  # COPY: body_xpos is a live view
            tr["fix"].append(np.array([e.sim.data.body_xpos[e.obj_body_id[n]] for n in fixn]))
            obs, _, d, _ = env.step(a)
            t += 1
            if d:
                ok = True
                break
        stem = f"{args.suite}_t{tid}_k{trial}"
        np.savez_compressed(os.path.join(out_dir, "trace", stem + ".npz"), names=np.array(names), fixnames=np.array(fixn),
                            **{k: np.asarray(v, dtype=np.float32) for k, v in tr.items()})
        imageio.mimwrite(os.path.join(out_dir, stem + ".mp4"), [np.asarray(f) for f in frames], fps=30)
        rec = {"suite": args.suite, "task_id": tid, "trial": trial, "prompt": prompt, "success": ok,
               "steps": t, "interest": list(e.obj_of_interest), "seconds": round(time.time() - t0, 1)}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{args.suite} w{args.worker}] t{tid} k{trial} {'OK ' if ok else 'FAIL'} {t} '{prompt}'", flush=True)
    env.close()


if __name__ == "__main__":
    main()
