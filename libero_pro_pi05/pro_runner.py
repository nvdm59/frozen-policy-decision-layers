"""Step 1 (P0) runner: PI's official pi05_libero on LIBERO / LIBERO-PRO suites.

- Policy: deployed openpi server (pi05_libero) over websocket; openpi LIBERO protocol
  (180° image flip, 8-D state, replan every 5, 10 settle steps) — copied from openpi_poc/audit_runner.py.
- PROMPT IS READ FROM THE BDDL FILE (:language ...). LIBERO-PRO's registry returns the ORIGINAL
  instruction for *_task suites; using task.language would silently evaluate the wrong prompt.
- Init states load through safe_init (allowlisted unpickler) — third-party pickles cannot run code.
- Resumable: one JSON line per episode in results/<tag>/<suite>_w<k>.jsonl; finished keys are skipped.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isolated LIBERO config -> LIBERO-PRO paths (shared ~/.libero still serves the July openpi_poc setup)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".libero_pro"))
import safe_init  # noqa: E402

safe_init.install()

HERE = os.path.dirname(os.path.abspath(__file__))
SETTLE = 10
REPLAN = 5
MAX_STEPS = {"libero_goal": 300, "libero_spatial": 220, "libero_object": 280, "libero_10": 520}
HOST = os.environ.get("PI05_HOST", "wss://<your-modal-workspace>--pi05-libero-pro-server-policy-server.us-east.modal.run")


def quat2axisangle(q):
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def build_obs(obs, prompt):
    return {
        "observation/image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "observation/wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "observation/state": np.concatenate((obs["robot0_eef_pos"],
                                             quat2axisangle(obs["robot0_eef_quat"]),
                                             obs["robot0_gripper_qpos"])),
        "prompt": prompt,
    }


class Policy:
    def __init__(self, host):
        import websockets.sync.client as wc
        from openpi_client import websocket_client_policy
        if not getattr(wc, "_p05_patched", False):
            _orig = wc.connect
            wc.connect = lambda *a, **k: _orig(*a, **{"open_timeout": 300, "ping_interval": 60,
                                                      "ping_timeout": 300, **k})
            wc._p05_patched = True
        self._mk = lambda: websocket_client_policy.WebsocketClientPolicy(host, None)
        self._connect()

    def _connect(self):
        last = None
        for i in range(8):
            try:
                self._p = self._mk()
                return
            except Exception as e:  # cold boot / transient
                last = e
                print(f"  connect {i + 1}/8 failed: {type(e).__name__}; retry 20s", flush=True)
                time.sleep(20)
        raise SystemExit(f"policy server unreachable: {last}")

    def infer(self, o):
        for i in range(3):
            try:
                return self._p.infer(o)
            except Exception as e:
                print(f"  infer failed ({type(e).__name__}); reconnect {i + 1}/3", flush=True)
                self._connect()
        return self._p.infer(o)


def bddl_language(path):
    m = re.search(r"\(:language\s+([^)]*)\)", open(path).read())
    return m.group(1).strip()


def base_suite(suite):
    for b in ("libero_goal", "libero_spatial", "libero_object", "libero_10"):
        if suite.startswith(b):
            return b
    raise ValueError(suite)


def run_episode(env, prompt, policy, max_steps, video=None):
    import imageio
    obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()
    plan = collections.deque()
    frames = []
    t, success = 0, False
    while t < max_steps + SETTLE:
        if t < SETTLE:
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            t += 1
            continue
        if video is not None:
            frames.append(obs["agentview_image"][::-1, ::-1])
        if not plan:
            plan.extend(policy.infer(build_obs(obs, prompt))["actions"][:REPLAN])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            success = True
            break
        t += 1
    if video is not None and frames:
        imageio.mimwrite(video, [np.asarray(f) for f in frames], fps=30)
    return success, t - SETTLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)          # e.g. libero_goal, libero_goal_task
    ap.add_argument("--trials", type=int, default=10)  # init states per task
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="p0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--host", default=HOST)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    out_dir = os.path.join(HERE, "results", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.suite}_w{args.worker}.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["task_id"], r["trial"]))

    suite = benchmark.get_benchmark_dict()[args.suite]()
    jobs = [(t, k) for t in range(suite.n_tasks) for k in range(args.trials)]
    jobs = [j for i, j in enumerate(jobs) if i % args.workers == args.worker and j not in done]
    print(f"{args.suite}: {len(jobs)} episodes to run (worker {args.worker}/{args.workers}, "
          f"{len(done)} already done)", flush=True)
    if not jobs:
        return
    policy = Policy(args.host)
    max_steps = MAX_STEPS[base_suite(args.suite)]
    env, env_task = None, None
    for task_id, trial in jobs:
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        prompt = bddl_language(bddl)
        if env_task != task_id:
            if env is not None:
                env.close()
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
            env.seed(args.seed)
            env_task = task_id
            inits = suite.get_task_init_states(task_id)
        env.reset()
        env.set_init_state(inits[trial % len(inits)])
        vid = (os.path.join(out_dir, f"{args.suite}_t{task_id}_k{trial}.mp4")
               if args.videos else None)
        t0 = time.time()
        ok, steps = run_episode(env, prompt, policy, max_steps, vid)
        rec = {"suite": args.suite, "task_id": task_id, "trial": trial, "name": task.name,
               "prompt": prompt, "success": ok, "steps": steps,
               "seconds": round(time.time() - t0, 1)}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{args.suite} w{args.worker}] t{task_id} k{trial} {'OK ' if ok else 'FAIL'} "
              f"{steps:3d} steps {rec['seconds']}s  '{prompt}'", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
