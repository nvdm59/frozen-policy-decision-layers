"""P2 — coordination-layer campaign: diagnosis + dev baseline (Modal A10G).

Modes:
  modal run modal_flower_p2.py --mode diagnose --shards 3
      Exact replay of P1 rep-0 (seed 300000+10*ci) for the 21 systematic-core
      chains. At every subtask start: precondition check derived from vendor
      tasks.py predicates evaluated on env.get_info(); named contact sets;
      door/light states. Frames via env.render(mode="rgb_array") at each
      subtask start + every 30 steps. Answers: is the failing subtask
      unwinnable at t=0 (precondition violated) or an execution failure?

  modal run modal_flower_p2.py --mode devbase --shards 4
      Baseline pass over the 200 dev-split chains (splits.json), per-chain
      recorded seed 400000+ci, forced-llvmpipe renderer. Harvests untainted
      dev failures for P2 iteration + renderer-consistent dev baseline.

INFRA RULES honored (see JOURNAL):
- Imports ONLY client-side objects (image, vol) from modal_flower_eval;
  every container-side helper is self-contained here (stale baked module).
- Renderer forced to Mesa llvmpipe on every host via
  __EGL_VENDOR_LIBRARY_FILENAMES so all compared arms are renderer-identical.
- torch imported before numpy/cv2 use (P0-proven order).
"""
import json
import pathlib
import time

import modal

from modal_flower_eval import image, vol  # client-side config objects ONLY

app = modal.App("flower-calvin-p2")

GPU_KIND = "A10G"
GPU_RATE_PER_HR = 1.10
RESULTS_ROOT = "/data/results"
CKPT_DIR = "/data/flower_calvin_abc"
DATASET_DIR = "/data/calvin_debug_dataset"
ROOM = pathlib.Path.home() / "cleanroom_calvin"

DIAG_TAG = "p2_diag"
DEVBASE_TAG = "p2_devbase"

MESA_ICD = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


# --------------------------------------------------------------------------
# container-side helpers (self-contained)
# --------------------------------------------------------------------------

def _shims():
    import collections
    import collections.abc
    for _n in dir(collections.abc):
        if not _n.startswith("_") and not hasattr(collections, _n):
            setattr(collections, _n, getattr(collections.abc, _n))
    import fractions
    import math
    if not hasattr(fractions, "gcd"):
        fractions.gcd = math.gcd
    import sys
    if "/opt/flower/calvin_env" not in sys.path:
        sys.path.insert(0, "/opt/flower/calvin_env")
    import calvin_env  # noqa: F401
    assert calvin_env.__file__ is not None, "calvin_env namespace-shadowed"


def _force_llvmpipe():
    """Pin the Mesa EGL ICD so every host renders with llvmpipe (arm parity)."""
    import os
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = MESA_ICD
    print(f"RENDERER_FORCE mesa icd={MESA_ICD}", flush=True)


def _load_model_and_env():
    from unittest.mock import patch

    from omegaconf import OmegaConf
    import hydra
    from pytorch_lightning import seed_everything
    from transformers.dynamic_module_utils import get_imports

    seed_everything(0, workers=True)

    from flower.evaluation.utils import get_default_mode_and_env

    eval_cfg_overwrite = OmegaConf.create({
        "datamodule": {"datasets": {"lang_dataset": {
            "lang_folder": "lang_annotations",
            "use_extracted_rel_actions": False,
        }}},
    })

    def _no_flash_attn(filename):
        return [i for i in get_imports(filename) if i != "flash_attn"]

    t0 = time.time()
    print("RENDERER_CHECK_BEGIN (env init follows)", flush=True)
    with patch("transformers.dynamic_module_utils.get_imports", _no_flash_attn):
        model, env, _, _ = get_default_mode_and_env(
            CKPT_DIR, DATASET_DIR, CKPT_DIR, env=None,
            lang_embeddings="unused-goals-are-lang_text",
            eval_cfg_overwrite=eval_cfg_overwrite, device_id=0,
        )
    print("RENDERER_CHECK_END", flush=True)

    model = model.to(0)
    model.num_sampling_steps = 4
    model.multistep = 10
    model.eval()

    hf_cfg = OmegaConf.load(f"{CKPT_DIR}/config.yaml")
    task_oracle = hydra.utils.instantiate(hf_cfg.callbacks.rollout_lh.tasks)
    val_annotations = hf_cfg.callbacks.rollout_lh.val_annotations

    rollout_cfg = OmegaConf.create({"ep_len": 360, "debug": False})
    print(f"model+env loaded in {time.time() - t0:.1f}s", flush=True)
    return model, env, task_oracle, val_annotations, rollout_cfg


def _standard_chains():
    from flower.evaluation.multistep_sequences import get_sequences
    return dict(enumerate(get_sequences(1000)))


# ---- precondition checker (mirrors vendor tasks.py + new_playtable_tasks) ----

PRE_SPECS = {
    # rotation: no hard t0 precondition (end needs non-robot contact, <0.1m move)
    **{f"rotate_{c}_block_{d}": ("rotate", f"block_{c}")
       for c in ("red", "blue", "pink") for d in ("right", "left")},
    # pushing: block needs a non-robot contact at t0
    **{f"push_{c}_block_{d}": ("push", f"block_{c}")
       for c in ("red", "blue", "pink") for d in ("right", "left")},
    # doors: travel headroom
    "move_slider_left": ("door", "base__slide", 0.15, (0.0, 0.56)),
    "move_slider_right": ("door", "base__slide", -0.15, (0.0, 0.56)),
    "open_drawer": ("door", "base__drawer", 0.12, (0.0, 0.24)),
    "close_drawer": ("door", "base__drawer", -0.12, (0.0, 0.24)),
    # lifting: block must touch the named (surface, link) at t0
    **{f"lift_{c}_block_table": ("lift", f"block_{c}", "table", "base_link")
       for c in ("red", "blue", "pink")},
    **{f"lift_{c}_block_slider": ("lift", f"block_{c}", "table", "plank_link")
       for c in ("red", "blue", "pink")},
    **{f"lift_{c}_block_drawer": ("lift", f"block_{c}", "table", "drawer_link")
       for c in ("red", "blue", "pink")},
    # placing: robot holds exactly one block, block touches only robot
    "place_in_slider": ("place", "table", "plank_link"),
    "place_in_drawer": ("place", "table", "drawer_link"),
    # stacking
    "stack_block": ("stack",),
    "unstack_block": ("unstack",),
    # lights: required start state
    "turn_on_lightbulb": ("light", "lightbulb", 0),
    "turn_off_lightbulb": ("light", "lightbulb", 1),
    "turn_on_led": ("light", "led", 0),
    "turn_off_led": ("light", "led", 1),
    # push into drawer: >=1 block whose sole contact is (table, base_link),
    # robot not touching it
    "push_into_drawer": ("push_into",
                         ["block_red", "block_blue", "block_pink"],
                         "table", "base_link"),
}


def _door_val(info, joint):
    import numpy as np
    return float(np.asarray(
        info["scene_info"]["doors"][joint]["current_state"]).flatten()[0])


def _precheck(subtask, info):
    """(ok, detail) — is `subtask` winnable given start_info == info?"""
    spec = PRE_SPECS[subtask]
    kind = spec[0]
    rob = info["robot_info"]["uid"]
    mov = info["scene_info"]["movable_objects"]
    fix = info["scene_info"]["fixed_objects"]

    def uids(o):
        return set(c[2] for c in o["contacts"])

    def pairs(o):
        return set((c[2], c[4]) for c in o["contacts"])

    if kind == "rotate":
        resting = len(uids(mov[spec[1]]) - {rob}) > 0
        return True, ("resting" if resting else
                      "note: block airborne/held (rotate end needs non-robot contact)")
    if kind == "push":
        ok = len(uids(mov[spec[1]]) - {rob}) > 0
        return ok, ("" if ok else
                    f"{spec[1]} has no non-robot contact at t0 -> push unwinnable")
    if kind == "door":
        _, joint, thr, (lo, hi) = spec
        cur = _door_val(info, joint)
        ok = (cur < hi - thr) if thr > 0 else (cur > lo - thr)
        return ok, f"joint={cur:.3f}" + ("" if ok else
                                         f" -> insufficient travel for {thr:+}")
    if kind == "lift":
        _, blk, sb, sl = spec
        tgt = (fix[sb]["uid"], fix[sb]["links"][sl])
        ok = tgt in pairs(mov[blk])
        return ok, ("" if ok else
                    f"{blk} not on {sb}/{sl} at t0 -> lift unwinnable")
    if kind == "place":
        rc = set(c[2] for c in info["robot_info"]["contacts"])
        if len(rc) != 1:
            return False, (f"robot touches {sorted(rc)} ({len(rc)} bodies) at "
                           f"t0; place needs exactly 1 held block")
        u = next(iter(rc))
        names = [k for k, v in mov.items() if v["uid"] == u]
        if len(names) != 1:
            return False, f"robot's single contact uid={u} is not a movable block"
        blk = names[0]
        extra = uids(mov[blk]) - {rob}
        ok = not extra
        return ok, (f"held={blk}" if ok else
                    f"held {blk} also touches uids {sorted(extra)} at t0")
    if kind == "push_into":
        _, blks, sb, sl = spec
        src = (fix[sb]["uid"], fix[sb]["links"][sl])
        cands = [b for b in blks
                 if rob not in uids(mov[b]) and pairs(mov[b]) == {src}]
        drawer = _door_val(info, "base__drawer")
        ok = len(cands) > 0
        return ok, (f"cands={cands} drawer={drawer:.3f}"
                    + ("" if ok else " -> no valid source block"))
    if kind == "stack":
        movu = set(v["uid"] for v in mov.values())
        free = [b for b, v in mov.items() if not (uids(v) & movu)]
        held = [b for b, v in mov.items() if uids(v) == {rob}]
        return len(free) > 0, (f"free={free} held={held}"
                               + ("" if free else " -> no stackable block"))
    if kind == "unstack":
        movu = set(v["uid"] for v in mov.values())
        stacked = [b for b, v in mov.items()
                   if uids(v) and uids(v) <= movu]
        return len(stacked) > 0, (f"stacked={stacked}"
                                  + ("" if stacked else " -> nothing stacked"))
    if kind == "light":
        _, name, s0 = spec
        cur = int(info["scene_info"]["lights"][name]["logical_state"])
        return cur == s0, f"state={cur}" + ("" if cur == s0 else f" needs {s0}")
    return True, "no-precheck"


def _state_summary(info):
    """Named, JSON-able snapshot of the coordination-relevant state."""
    import numpy as np
    rob = info["robot_info"]["uid"]
    mov = info["scene_info"]["movable_objects"]
    fix = info["scene_info"]["fixed_objects"]
    uid_name = {v["uid"]: k for k, v in mov.items()}
    uid_name.update({v["uid"]: k for k, v in fix.items()})
    uid_name[rob] = "robot"
    link_name = {}
    for fk, fv in fix.items():
        for ln, li in fv.get("links", {}).items():
            link_name[(fv["uid"], li)] = f"{fk}/{ln}"

    def named(contacts):
        out = []
        for c in contacts:
            key = (c[2], c[4])
            out.append(link_name.get(key, f"{uid_name.get(c[2], c[2])}"))
        return sorted(set(out))

    return {
        "doors": {j: round(_door_val(info, j), 3)
                  for j in info["scene_info"]["doors"]},
        "lights": {k: int(v["logical_state"])
                   for k, v in info["scene_info"]["lights"].items()},
        "robot_contacts": named(info["robot_info"]["contacts"]),
        "blocks": {
            b: {"pos": [round(float(x), 3) for x in np.asarray(v["current_pos"])],
                "contacts": named(v["contacts"])}
            for b, v in mov.items()},
    }


def _save_frame(env, path):
    import cv2
    import numpy as np
    try:
        fr = np.asarray(env.render(mode="rgb_array"))
        if fr.dtype != np.uint8:
            fr = np.clip(fr, 0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        return True
    except Exception as e:
        print(f"frame save failed: {e}", flush=True)
        return False


def _rollout_instrumented(env, model, task_oracle, subtask, val_annotations,
                          cfg, frames_dir, sub_idx, frame_every=30):
    """Vendor rollout() semantics + frame capture. Returns (success, steps)."""
    import os
    obs = env.get_obs()
    goal = {"lang_text": val_annotations[subtask][0]}
    model.reset()
    start_info = env.get_info()
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        _save_frame(env, f"{frames_dir}/sub{sub_idx:02d}_{subtask}_s0000.jpg")
    for step in range(cfg.ep_len):
        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        if frames_dir and (step + 1) % frame_every == 0:
            _save_frame(env,
                        f"{frames_dir}/sub{sub_idx:02d}_{subtask}_s{step + 1:04d}.jpg")
        if task_oracle.get_task_info_for_set(start_info, current_info,
                                             {subtask}):
            if frames_dir:
                _save_frame(env, f"{frames_dir}/sub{sub_idx:02d}_{subtask}"
                                 f"_s{step + 1:04d}_SUCCESS.jpg")
            return True, step + 1
    if frames_dir:
        _save_frame(env,
                    f"{frames_dir}/sub{sub_idx:02d}_{subtask}_sFAIL.jpg")
    return False, cfg.ep_len


# --------------------------------------------------------------------------
# mode: diagnose
# --------------------------------------------------------------------------

@app.function(image=image, gpu=GPU_KIND, volumes={"/data": vol},
              timeout=43200, memory=24576)
def diagnose_chains(chain_indices, shard: int):
    import os

    _force_llvmpipe()
    _shims()
    import torch  # noqa: F401

    from flower.evaluation.utils import get_env_state_for_initial_condition
    from pytorch_lightning import seed_everything

    model, env, task_oracle, val_annotations, cfg = _load_model_and_env()
    all_chains = _standard_chains()

    out_dir = f"{RESULTS_ROOT}/{DIAG_TAG}"
    os.makedirs(out_dir, exist_ok=True)

    records = []
    t0_run = time.time()
    for ci in chain_indices:
        initial_state, sequence = all_chains[ci]
        seed = 300000 + ci * 10  # P1 rep-0 replay
        seed_everything(seed, workers=True)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        frames_dir = f"{out_dir}/frames/chain_{ci:04d}"

        subrecs = []
        n_success = 0
        t0 = time.time()
        for k, subtask in enumerate(sequence):
            info0 = env.get_info()
            ok, detail = _precheck(subtask, info0)
            summ = _state_summary(info0)
            success, steps = _rollout_instrumented(
                env, model, task_oracle, subtask, val_annotations, cfg,
                frames_dir, k)
            subrecs.append({
                "sub_idx": k, "subtask": subtask,
                "precheck_ok": bool(ok), "precheck_detail": detail,
                "state": summ, "success": bool(success), "steps": steps,
            })
            print(f"[diag s{shard}] chain {ci} sub{k} {subtask}: "
                  f"pre={'OK' if ok else 'VIOLATED'} ({detail}) -> "
                  f"{'success' if success else 'FAIL'} in {steps}", flush=True)
            if success:
                n_success += 1
            else:
                break
        records.append({
            "chain_idx": ci, "seed": seed, "sequence": list(sequence),
            "n_success": n_success, "p1_rep0_replay": True,
            "wall_s": round(time.time() - t0, 1), "subtasks": subrecs,
        })
        with open(f"{out_dir}/diag_shard{shard}.json", "w") as f:
            json.dump({"shard": shard, "records": records}, f, indent=1)
        vol.commit()

    total_s = time.time() - t0_run
    print(f"[diag s{shard}] done {len(chain_indices)} chains in "
          f"{total_s / 3600:.2f} h", flush=True)
    return {"shard": shard, "records": records, "shard_wall_s": total_s}


# --------------------------------------------------------------------------
# arms: baseline / control (blind retry) / layer (precondition gate)
# --------------------------------------------------------------------------

def _color_of(subtask):
    for c in ("red", "blue", "pink"):
        if c in subtask:
            return c
    return None


def _lift_instruction_for(block_color, info, val_annotations):
    """Pick the lift annotation matching where the block currently rests."""
    blk = f"block_{block_color}"
    mov = info["scene_info"]["movable_objects"]
    fix = info["scene_info"]["fixed_objects"]
    pairs = set((c[2], c[4]) for c in mov[blk]["contacts"])
    tuid = fix["table"]["uid"]
    links = fix["table"]["links"]
    if (tuid, links.get("plank_link")) in pairs:
        task = f"lift_{block_color}_block_slider"
    elif (tuid, links.get("drawer_link")) in pairs:
        task = f"lift_{block_color}_block_drawer"
    else:
        task = f"lift_{block_color}_block_table"
    return str(val_annotations[task][0])


def _recovery_plan(subtask, detail, info, prev_subtask, val_annotations,
                   dropped_block=None):
    """Deterministic rule: violation -> recovery action.

    Returns ("script_release", None) | ("script_release_lateral", None) |
            ("script_lift_clear", None) | ("instruct", text) |
            (None, None) if no recovery known.
    """
    kind = PRE_SPECS[subtask][0]
    rob = info["robot_info"]["uid"]
    mov = info["scene_info"]["movable_objects"]

    def uids(o):
        return set(c[2] for c in o["contacts"])

    if kind == "place":
        rc = set(c[2] for c in info["robot_info"]["contacts"])
        movu = set(v["uid"] for v in mov.values())
        touched_movable = [b for b, v in mov.items() if rob in uids(v)]
        if rc and rc <= movu and touched_movable:
            # touching a block (possibly dirtily) -> lift the grasp clear
            return "script_lift_clear", None
        # not holding any block (arm may rest on table/fixed bodies) ->
        # re-lift: the block dropped in a doomed attempt if known, else
        # prev subtask's color, else any free block
        color = (dropped_block.split("_")[1] if dropped_block
                 else _color_of(prev_subtask or ""))
        if color is None:
            free = [b for b, v in mov.items()
                    if not (uids(v) & movu) and rob not in uids(v)]
            if not free:
                return None, None
            color = free[0].split("_")[1]
        return "instruct", _lift_instruction_for(color, info,
                                                 val_annotations)
    if kind in ("push", "lift"):
        # held-in-air / wrong-surface signatures -> carry toward table
        # center, release, retract
        return "script_release_lateral", None
    if kind == "push_into":
        # most common violation: robot still touching the pushed block
        return "script_release", None
    if kind == "light":
        _, name, s0 = PRE_SPECS[subtask]
        opposite = {
            "turn_on_lightbulb": "turn_off_lightbulb",
            "turn_off_lightbulb": "turn_on_lightbulb",
            "turn_on_led": "turn_off_led",
            "turn_off_led": "turn_on_led",
        }[subtask]
        return "instruct", str(val_annotations[opposite][0])
    if kind == "door":
        opposite = {
            "move_slider_left": "move_slider_right",
            "move_slider_right": "move_slider_left",
            "open_drawer": "close_drawer",
            "close_drawer": "open_drawer",
        }[subtask]
        return "instruct", str(val_annotations[opposite][0])
    return None, None


def _run_scripted(env, info, variant, track):
    """Scripted primitive. Vendor gripper convention: +1 = open, -1 = close
    (robot.py apply_action). HulcWrapper.step expects a torch tensor and
    thresholds the gripper itself (>0 -> 1 else -1)."""
    import numpy as np
    import torch
    OPEN, CLOSE = 1.0, -1.0

    def act(dx=0.0, dy=0.0, dz=0.0, g=OPEN):
        return torch.tensor([dx, dy, dz, 0, 0, 0, g], dtype=torch.float32)

    if variant == "script_release":
        # detach (open) then retract straight up
        seq = [act(g=OPEN)] * 6 + [act(dz=0.7, g=OPEN)] * 10
    elif variant == "script_release_lateral":
        # carry the held block over open table (toward table center-front),
        # then release and retract up
        seq = []
        try:
            tcp = np.asarray(info["robot_info"]["tcp_pos"], dtype=float)
            tgt = np.array([0.05, -0.12])
            d = tgt - tcp[:2]
            n = float(np.linalg.norm(d))
            if n > 0.03:
                step2 = np.clip(d / max(n, 1e-6), -1, 1) * 0.6
                n_lat = int(min(12, max(4, n / 0.012)))
                seq += [act(dx=float(step2[0]), dy=float(step2[1]),
                            g=CLOSE)] * n_lat
        except Exception:
            pass
        seq += [act(g=OPEN)] * 6 + [act(dz=0.7, g=OPEN)] * 10
    else:  # script_lift_clear: raise the held block clear of surfaces
        seq = [act(dz=0.6, g=CLOSE)] * 8
    steps = 0
    for a in seq:
        env.step(a)
        steps += 1
        track["steps"] += 1
    return steps


def _run_instructed(env, model, text, stop_fn, track, max_steps=120,
                    settle_steps=15):
    """Policy-driven corrective burst; stops when stop_fn(info) holds, then
    keeps following the same instruction settle_steps more (grasp
    stabilization) and re-verifies. Re-enters the loop if the condition
    broke during settling."""
    obs = env.get_obs()
    goal = {"lang_text": text}
    model.reset()
    steps = 0
    while steps < max_steps:
        action = model.step(obs, goal)
        obs, _, _, info = env.step(action)
        steps += 1
        track["steps"] += 1
        if stop_fn(info):
            # settle: same instruction, then re-verify
            for _ in range(settle_steps):
                action = model.step(obs, goal)
                obs, _, _, info = env.step(action)
                steps += 1
                track["steps"] += 1
            if stop_fn(info):
                break
            # else: condition broke during settling; keep trying
    return steps


def _held_block(info):
    """Name of the block the robot holds cleanly, else None."""
    rob = info["robot_info"]["uid"]
    mov = info["scene_info"]["movable_objects"]
    rc = set(c[2] for c in info["robot_info"]["contacts"])
    if len(rc) != 1:
        return None
    u = next(iter(rc))
    names = [k for k, v in mov.items() if v["uid"] == u]
    if len(names) != 1:
        return None
    blk = names[0]
    if set(c[2] for c in mov[blk]["contacts"]) == {rob}:
        return blk
    return None


def _make_place_doom_fn(subtask, start_info):
    """Doom monitor for place tasks: fires when the initially-held block has
    been dropped somewhere other than the destination (resting, robot-free,
    no dest contact) for 12 consecutive steps. Returns (fn, held_block)."""
    held = _held_block(start_info)
    if held is None:
        return None, None
    _, dest_body, dest_link = PRE_SPECS[subtask]
    counter = {"n": 0}

    def doom(info):
        rob = info["robot_info"]["uid"]
        fix = info["scene_info"]["fixed_objects"]
        dest = (fix[dest_body]["uid"], fix[dest_body]["links"][dest_link])
        o = info["scene_info"]["movable_objects"][held]
        uids = set(c[2] for c in o["contacts"])
        pairs = set((c[2], c[4]) for c in o["contacts"])
        lost = bool(uids) and rob not in uids and dest not in pairs
        counter["n"] = counter["n"] + 1 if lost else 0
        return counter["n"] >= 12

    return doom, held


def _plain_rollout(env, model, task_oracle, subtask, val_annotations, cfg,
                   track, doom_fn=None):
    """Vendor rollout() semantics + step tracking + optional doom-abort.

    Returns (status, steps): status in {"success", "fail", "aborted"}.
    """
    obs = env.get_obs()
    goal = {"lang_text": val_annotations[subtask][0]}
    model.reset()
    start_info = env.get_info()
    for step in range(cfg.ep_len):
        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        track["steps"] += 1
        if task_oracle.get_task_info_for_set(start_info, current_info,
                                             {subtask}):
            return "success", step + 1
        if doom_fn is not None and doom_fn(current_info):
            return "aborted", step + 1
    return "fail", cfg.ep_len


@app.function(image=image, gpu=GPU_KIND, volumes={"/data": vol},
              timeout=43200, memory=24576)
def arm_chains(chain_indices, arm: str, budget: int, tag: str, shard: int,
               seed_base: int):
    """One arm over a chain list. arm in {baseline, control, layer}.

    Budget = extra rollout-attempt tokens per chain. Control spends them on
    blind retries of the failed subtask (re-baselined start_info). Layer
    spends the SAME pool on precondition recoveries + retries. Baseline
    ignores budget (single attempt per subtask, vendor protocol).
    """
    import os

    _force_llvmpipe()
    _shims()
    import torch  # noqa: F401

    from flower.evaluation.utils import get_env_state_for_initial_condition
    from pytorch_lightning import seed_everything

    model, env, task_oracle, val_annotations, cfg = _load_model_and_env()
    all_chains = _standard_chains()

    out_dir = f"{RESULTS_ROOT}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{arm}_shard{shard}.json"

    records = []
    t_run0 = time.time()
    for ci in chain_indices:
        initial_state, sequence = all_chains[ci]
        seed = seed_base + ci
        seed_everything(seed, workers=True)
        robot_obs, scene_obs = get_env_state_for_initial_condition(
            initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        track = {"steps": 0}
        tokens = 0 if arm == "baseline" else budget
        n_success = 0
        events = []
        t0 = time.time()
        prev_subtask = None
        dropped_block = None
        for k, subtask in enumerate(sequence):
            success = False
            dropped_block = None
            while True:
                doom_fn = None
                if arm == "layer":
                    info0 = env.get_info()
                    ok, detail = _precheck(subtask, info0)
                    if not ok and tokens > 0:
                        rtype, rtext = _recovery_plan(
                            subtask, detail, info0, prev_subtask,
                            val_annotations, dropped_block=dropped_block)
                        if rtype is not None:
                            tokens -= 1
                            if rtype.startswith("script"):
                                rsteps = _run_scripted(env, info0, rtype,
                                                       track)
                            else:
                                def _stop(info, _st=subtask):
                                    return _precheck(_st, info)[0]
                                rsteps = _run_instructed(
                                    env, model, rtext, _stop, track)
                            ok2, detail2 = _precheck(subtask, env.get_info())
                            events.append({
                                "sub_idx": k, "type": "recovery",
                                "recovery": rtype, "text": rtext,
                                "violation": detail, "steps": rsteps,
                                "fixed": bool(ok2), "post": detail2})
                            print(f"[{arm} s{shard}] chain {ci} sub{k} "
                                  f"{subtask}: RECOVERY {rtype} "
                                  f"({detail}) -> fixed={ok2}", flush=True)
                    if PRE_SPECS[subtask][0] == "place":
                        doom_fn, held0 = _make_place_doom_fn(
                            subtask, env.get_info())
                status, steps = _plain_rollout(
                    env, model, task_oracle, subtask, val_annotations, cfg,
                    track, doom_fn=doom_fn)
                success = status == "success"
                events.append({"sub_idx": k, "type": "attempt",
                               "subtask": subtask, "status": status,
                               "steps": steps})
                if status == "aborted":
                    dropped_block = held0
                    print(f"[{arm} s{shard}] chain {ci} sub{k} {subtask}: "
                          f"DOOM-ABORT at {steps} (dropped {held0})",
                          flush=True)
                if success or tokens <= 0:
                    break
                tokens -= 1  # retry (both non-baseline arms)
            if success:
                n_success += 1
                prev_subtask = subtask
            else:
                break
        records.append({
            "chain_idx": ci, "arm": arm, "seed": seed, "budget": budget,
            "sequence": list(sequence), "n_success": n_success,
            "tokens_left": tokens, "env_steps": track["steps"],
            "wall_s": round(time.time() - t0, 1), "events": events,
        })
        print(f"[{arm} s{shard}] chain {ci}: {n_success}/5 "
              f"(tokens left {tokens}, {track['steps']} steps)", flush=True)
        with open(out_path, "w") as f:
            json.dump({"shard": shard, "arm": arm, "records": records},
                      f, indent=1)
        vol.commit()

    total_s = time.time() - t_run0
    print(f"[{arm} s{shard}] done {len(chain_indices)} in "
          f"{total_s / 3600:.2f} h", flush=True)
    return {"shard": shard, "arm": arm, "records": records,
            "shard_wall_s": total_s}


# --------------------------------------------------------------------------
# mode: devbase
# --------------------------------------------------------------------------

@app.function(image=image, gpu=GPU_KIND, volumes={"/data": vol},
              timeout=43200, memory=24576)
def devbase_chains(chain_indices, shard: int):
    import os

    _force_llvmpipe()
    _shims()
    import torch  # noqa: F401

    from flower.evaluation import flower_evaluate as fe
    from pytorch_lightning import seed_everything

    model, env, task_oracle, val_annotations, cfg = _load_model_and_env()
    all_chains = _standard_chains()

    out_dir = f"{RESULTS_ROOT}/{DEVBASE_TAG}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/devbase_shard{shard}.json"

    # lightweight boundary + step hooks (P1-proven pattern; no obs capture)
    orig_model_reset = model.reset
    orig_env_step = env.step
    state = {"step": 0, "bounds": []}

    def hook_reset(*a, **kw):
        state["bounds"].append(state["step"])
        return orig_model_reset(*a, **kw)

    def hook_step(action):
        r = orig_env_step(action)
        state["step"] += 1
        return r

    model.reset = hook_reset
    env.step = hook_step

    records = []
    t0_run = time.time()
    for ci in chain_indices:
        initial_state, sequence = all_chains[ci]
        seed = 400000 + ci
        seed_everything(seed, workers=True)
        state["step"], state["bounds"] = 0, []
        t0 = time.time()
        n_success = fe.evaluate_sequence(
            env, model, task_oracle, initial_state, list(sequence),
            None, val_annotations, cfg, False, None, ci)
        records.append({
            "chain_idx": ci, "seed": seed, "sequence": list(sequence),
            "n_success": int(n_success),
            "wall_s": round(time.time() - t0, 1),
            "subtask_start_steps": list(state["bounds"]),
            "total_steps": state["step"],
        })
        print(f"[devbase s{shard}] chain {ci}: {n_success}/5 "
              f"({len(records)}/{len(chain_indices)})", flush=True)
        if len(records) % 10 == 0 or len(records) == len(chain_indices):
            with open(out_path, "w") as f:
                json.dump({"shard": shard, "records": records}, f, indent=1)
            vol.commit()

    total_s = time.time() - t0_run
    print(f"[devbase s{shard}] done {len(chain_indices)} in "
          f"{total_s / 3600:.2f} h", flush=True)
    return {"shard": shard, "records": records, "shard_wall_s": total_s}


# --------------------------------------------------------------------------
# local entrypoint
# --------------------------------------------------------------------------

CORE_CHAINS = [19, 25, 33, 37, 38, 45, 47, 52, 96, 98, 112, 115, 120, 130,
               131, 138, 141, 148, 180, 181, 183]


@app.local_entrypoint()
def main(mode: str = "diagnose", shards: int = 3, arm: str = "layer",
         budget: int = 2, tag: str = "p2_it1", seed_base: int = 510000,
         chains: str = "core"):
    t0 = time.time()
    out_dir = ROOM / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "diagnose":
        idxs = CORE_CHAINS
        fn, tag = diagnose_chains, DIAG_TAG
    elif mode == "devbase":
        splits = json.load(open(ROOM / "splits.json"))
        idxs = sorted(splits["dev"])
        fn, tag = devbase_chains, DEVBASE_TAG
    elif mode == "arm":
        if chains == "core":
            idxs = CORE_CHAINS
        elif chains == "devfail":
            db = json.load(open(ROOM / "results" / "p2_devbase_summary.json"))
            idxs = db["summary"]["failing_chains"]
        elif chains in ("val", "heldout"):
            splits = json.load(open(ROOM / "splits.json"))
            idxs = sorted(splits[chains])
        else:  # comma-separated list
            idxs = sorted(int(x) for x in chains.split(","))
        fn = arm_chains
        tag = tag
    else:
        raise SystemExit(f"unknown mode {mode}")

    shard_lists = [idxs[s::shards] for s in range(shards)]
    if mode == "arm":
        args = [(shard_lists[s], arm, budget, tag, s, seed_base)
                for s in range(shards)]
    else:
        args = [(shard_lists[s], s) for s in range(shards)]
    print(f"{mode}({arm if mode == 'arm' else ''}): {len(idxs)} chains in "
          f"{shards} shards")

    all_records, hours = [], 0.0
    for res in fn.starmap(args, order_outputs=True):
        all_records.extend(res["records"])
        hours += res["shard_wall_s"] / 3600
    all_records.sort(key=lambda r: r["chain_idx"])

    summary = {
        "tag": tag, "num_chains": len(all_records),
        "gpu_container_hours": round(hours, 2),
        "approx_gpu_cost_usd": round(hours * GPU_RATE_PER_HR, 2),
    }
    out_name = f"{tag}_summary.json"
    if mode == "arm":
        ns = [r["n_success"] for r in all_records]
        summary.update({
            "arm": arm, "budget": budget, "seed_base": seed_base,
            "avg_seq_len": round(sum(ns) / len(ns), 4),
            "per_chain": {str(r["chain_idx"]): r["n_success"]
                          for r in all_records},
            "total_env_steps": sum(r["env_steps"] for r in all_records),
        })
        out_name = f"{tag}_{arm}_summary.json"
    elif mode == "devbase":
        ns = [r["n_success"] for r in all_records]
        summary["avg_seq_len"] = round(sum(ns) / len(ns), 4)
        summary["chain_success_rates"] = {
            i: round(sum(1 for n in ns if n >= i) / len(ns), 4)
            for i in range(1, 6)}
        summary["failing_chains"] = sorted(
            r["chain_idx"] for r in all_records if r["n_success"] < 5)
    else:
        summary["replay_match_note"] = "compare n_success vs P1 rep0"

    print(json.dumps(summary, indent=2))
    with open(out_dir / out_name, "w") as f:
        json.dump({"summary": summary, "records": all_records}, f, indent=1)
    print(f"written {out_dir}/{out_name}; "
          f"local wall {(time.time() - t0) / 60:.1f} min")
