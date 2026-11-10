"""P2 dev test: position-correcting wrapper around frozen π0.5 on *_swap suites.
Arms: post = after a verified grasp, carry the object above the instruction's destination, hand back.
      both = additionally pre-position the open gripper above the grasp target at episode start.
Positions come from simulator state (privileged; a real robot would need a detector). Same step budget; macro steps count."""
import argparse, collections, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_runner as pr  # noqa: E402
import layer_v1 as L  # noqa: E402
L.SYN["cabinet"] = ["drawer"]
NOOP = [0.0] * 6 + [-1.0]
K, TOL, MAXM = 8.0, 0.015, 60
PRE_H, CARRY_H = 0.15, 0.30
PLACE_MAX, DESC_A = 80, 0.6


ON_PREPS = ("on", "on top of", "on the top of", "onto")
IN_PREPS = ("in", "into", "inside")


def _prep_before(text, start):
    words = L._strip_tail(text[:start].rstrip()).split()
    for k in (4, 3, 2, 1):
        if len(words) >= k and " ".join(words[-k:]) in ON_PREPS + IN_PREPS:
            return " ".join(words[-k:])
    return None


def dest_name(env, text, target):
    """Destination of the FIRST target = first 'on/in <thing>' mention after the placing verb that follows the
    target phrase ('put X on Y' -> after 'put'; 'pick up X ... and place it on Y' -> after 'place').
    Carry applies only to: on-type destinations, or in-type destinations that are movable containers.
    -> (name | None, reason)"""
    e = env.env
    movable = list(e.objects_dict); alln = movable + list(e.fixtures_dict)
    verbs = [m.end() for m in re.finditer(r"\b(put|place)\b", text)]
    if not verbs:
        return None, "no_place_verb"
    start = verbs[1] if (text.startswith("pick") and len(verbs) > 1) else verbs[0]
    if text.startswith("pick") and len(verbs) == 1:
        start = verbs[0]
    for h in L._mentions(text, alln):
        if h[0] < start:
            continue
        prep = _prep_before(text, h[0])
        if prep is None:
            continue
        adj = re.search(r"\b(left|right)\s+$", text[:h[0]])
        rel = (adj.group(1).upper() + " ") if adj else ""
        cands = [n for n in alln if L._base(n) == h[2] and n != target]
        name = L._resolve(env, h[2], rel, cands, alln) if cands else None
        if name is None:
            return None, "dest_unresolved"
        if prep in IN_PREPS and name not in movable:
            return None, "in_fixture"
        return name, prep
    return None, "no_dest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--arm", required=True, choices=["base", "post", "place", "layer", "both", "parse"])
    ap.add_argument("--trials", type=int, default=10); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default=""); ap.add_argument("--lang", default="task", choices=["task", "bddl"])
    args = ap.parse_args()
    import imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    out_dir = os.path.join(pr.HERE, "results", args.tag); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.suite}_{args.arm}_w{args.worker}.jsonl")
    done = {(r["task_id"], r["trial"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    jobs = [(t, k) for t in range(suite.n_tasks) for k in range(args.trials)]
    if args.tasks:
        keep = {int(x) for x in args.tasks.split(",")}; jobs = [j for j in jobs if j[0] in keep]
    jobs = [j for i, j in enumerate(jobs) if i % args.workers == args.worker and j not in done]
    print(f"{args.suite}/{args.arm}: {len(jobs)} episodes", flush=True)
    if not jobs:
        return
    if args.arm == "parse":
        jobs = sorted({(t, 0) for t, _ in jobs})
    policy = pr.Policy(pr.HOST) if args.arm != "parse" else None; max_steps = pr.MAX_STEPS[pr.base_suite(args.suite)]
    env, cur = None, None
    for tid, trial in jobs:
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        if cur != tid:
            if env is not None:
                env.close()
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224); env.seed(7)
            inits = suite.get_task_init_states(tid); cur = tid
        env.reset(); env.set_init_state(inits[trial % len(inits)])
        e = env.env; prompt = task.language if args.lang == "task" else pr.bddl_language(bddl)
        obs = None
        for _ in range(pr.SETTLE):
            obs, _, _, _ = env.step(NOOP)
        tg = L.parse_targets(env, prompt)
        target = tg[0][0] if tg else None
        dest, why = dest_name(env, prompt.lower(), target) if target else (None, "no_target")
        if args.arm == "parse":
            rec = {"suite": args.suite, "arm": "parse", "task_id": tid, "trial": trial, "prompt": prompt,
                   "target": target, "dest": dest, "why": why}
            open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
            continue
        if args.arm == "base":
            dest = None
        rest = L.pos(env, target)[2] if target else None
        ev, frames, plan = [], [], collections.deque()
        t, ok, carried = 0, False, False
        state = {"obs": obs, "t": 0, "ok": False}

        def servo(goal_fn, grip, label):
            n = 0
            while state["t"] < max_steps and n < MAXM:
                err = goal_fn() - state["obs"]["robot0_eef_pos"]
                if np.linalg.norm(err) < TOL:
                    break
                a = np.clip(K * err, -1, 1).tolist() + [0.0, 0.0, 0.0, grip]
                frames.append(state["obs"]["agentview_image"][::-1, ::-1])
                state["obs"], _, d, _ = env.step(a); state["t"] += 1; n += 1
                if d:
                    state["ok"] = True; break
            ev.append((label, state["t"], n))

        if args.arm == "both" and target:
            servo(lambda: L.pos(env, target) + np.array([0, 0, PRE_H]), -1.0, "pre")
        while state["t"] < max_steps and not state["ok"]:
            if target and dest and not carried and L.pos(env, target)[2] - rest >= L.LIFT_M:
                carried = True
                # v2: clearance is referenced to the OBJECT (its centre rides at rest+CARRY_H), not the gripper,
                # so a tall object hanging below the fingers still clears the destination.
                xy0 = np.array(state["obs"]["robot0_eef_pos"][:2])
                need = lambda: max(0.0, rest + CARRY_H - L.pos(env, target)[2])
                servo(lambda: np.array([*xy0, state["obs"]["robot0_eef_pos"][2] + need()]), 1.0, "raise")
                z = float(state["obs"]["robot0_eef_pos"][2])
                servo(lambda: np.array([*L.pos(env, dest)[:2], z]), 1.0, "carry")
                # frozen rule (arm=layer): scripted lower+release only when the destination is a MOVABLE object
                # (its body origin is its placement centre); for fixtures hand back to the policy from the hover.
                do_place = args.arm == "place" or (args.arm == "layer" and dest in e.objects_dict)
                if do_place and not state["ok"]:
                    # lower until the object stops descending (contact with the support), release, retreat
                    n, zs = 0, [float(L.pos(env, target)[2])]
                    while state["t"] < max_steps and n < PLACE_MAX and not state["ok"]:
                        exy = np.clip(K * (L.pos(env, dest)[:2] - L.pos(env, target)[:2]), -0.3, 0.3)
                        frames.append(state["obs"]["agentview_image"][::-1, ::-1])
                        state["obs"], _, d, _ = env.step([float(exy[0]), float(exy[1]), -DESC_A, 0.0, 0.0, 0.0, 1.0])
                        state["t"] += 1; n += 1; state["ok"] = bool(d)
                        zs.append(float(L.pos(env, target)[2]))
                        if n >= 6 and zs[-4] - zs[-1] < 0.001:
                            break
                    ev.append(("lower", state["t"], n))
                    for a in [NOOP] * 8 + [[0.0, 0.0, 0.6, 0.0, 0.0, 0.0, -1.0]] * 10:
                        if state["t"] >= max_steps or state["ok"]:
                            break
                        frames.append(state["obs"]["agentview_image"][::-1, ::-1])
                        state["obs"], _, d, _ = env.step(a); state["t"] += 1; state["ok"] = bool(d)
                    ev.append(("release", state["t"], 0))
                plan.clear()
                continue
            frames.append(state["obs"]["agentview_image"][::-1, ::-1])
            if not plan:
                plan.extend(policy.infer(pr.build_obs(state["obs"], prompt))["actions"][:pr.REPLAN])
            state["obs"], _, d, _ = env.step(plan.popleft().tolist()); state["t"] += 1
            if d:
                state["ok"] = True
        stem = f"{args.suite}_{args.arm}_t{tid}_k{trial}"
        imageio.mimwrite(os.path.join(out_dir, stem + ".mp4"), [np.asarray(f) for f in frames], fps=30)
        rec = {"suite": args.suite, "arm": args.arm, "task_id": tid, "trial": trial, "prompt": prompt,
               "target": target, "dest": dest, "why": why, "success": state["ok"], "steps": state["t"], "events": ev,
               "final_lift": (round(float(L.pos(env, target)[2] - rest), 3) if target else None)}
        open(out, "a").write(json.dumps(rec) + "\n")
        print(f"[{args.arm} w{args.worker}] t{tid} k{trial} {'OK ' if state['ok'] else 'FAIL'} {state['t']} tgt={target} dest={dest} {ev}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
