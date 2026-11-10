"""P1 failure taxonomy — re-rollout of the 42 P0-breaking chains, k repeats.

Purpose (see ~/cleanroom_calvin/JOURNAL.md, P1 section):
- Separate systematic breaks from sampling luck (LIBERO vaccination): each
  breaking chain re-rolled k=3 times with fresh, recorded policy-noise seeds.
- Frame + state capture on the FIRST repeat of each chain (rgb_static jpeg
  every 12 env-steps; robot_obs/scene_obs every step; subtask boundaries via
  a model.reset hook) so surviving systematic failures can be mechanism-tagged.
- Vendor code untouched: all eval semantics remain
  flower.evaluation.flower_evaluate.evaluate_sequence; hooks are runtime
  monkeypatches on the env/model objects (wrapper-side only).

STALE-MODULE HAZARD (learned the hard way, see JOURNAL "P1 launch #1"):
image.run_function() baked a pre-shim-fix copy of modal_flower_eval.py into
the image at /root; importing modal_flower_eval INSIDE the container resolves
to that stale copy. Therefore this file imports ONLY client-side objects
(image, vol, constants) from modal_flower_eval and carries its own runtime
helpers. Nothing from modal_flower_eval may be called container-side.

Run:
  modal run modal_flower_p1.py --repeats 3 --shards 4
"""
import json
import pathlib
import time

import modal

# CLIENT-SIDE ONLY imports (image/vol are lazy config objects; importing them
# here keeps the exact P0 image cache key). Only names guaranteed present in
# ANY vintage of the module (the container resolves a stale baked copy).
from modal_flower_eval import image, vol

app = modal.App("flower-calvin-p1")

GPU_KIND = "A10G"
GPU_RATE_PER_HR = 1.10
RESULTS_ROOT = "/data/results"
FAILS_PATH = pathlib.Path.home() / "cleanroom_calvin" / "results" / "p0_failures.json"
TAG = "p1_taxonomy"
CKPT_DIR = "/data/flower_calvin_abc"
DATASET_DIR = "/data/calvin_debug_dataset"


# --------------------------------------------------------------------------
# runtime helpers (self-contained copies; P0-proven logic + shim fix)
# --------------------------------------------------------------------------

def _p1_shims():
    """py3.9-era vendor code on py3.10 + calvin_env namespace-shadow fix."""
    import collections
    import collections.abc
    for _n in dir(collections.abc):
        if not _n.startswith("_") and not hasattr(collections, _n):
            setattr(collections, _n, getattr(collections.abc, _n))
    import fractions
    import math
    if not hasattr(fractions, "gcd"):
        fractions.gcd = math.gcd
    # real calvin_env package must win over the /opt/flower namespace portion
    import sys
    if "/opt/flower/calvin_env" not in sys.path:
        sys.path.insert(0, "/opt/flower/calvin_env")
    import calvin_env  # noqa: F401
    assert calvin_env.__file__ is not None, "calvin_env namespace-shadowed"


def _p1_load_model_and_env():
    """Load FLOWER + CALVIN env via vendor loaders (P0-proven settings)."""
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
    with patch("transformers.dynamic_module_utils.get_imports", _no_flash_attn):
        model, env, _, _ = get_default_mode_and_env(
            CKPT_DIR,
            DATASET_DIR,
            CKPT_DIR,
            env=None,
            lang_embeddings="unused-goals-are-lang_text",
            eval_cfg_overwrite=eval_cfg_overwrite,
            device_id=0,
        )
    load_s = time.time() - t0

    model = model.to(0)
    model.num_sampling_steps = 4
    model.multistep = 10
    model.eval()

    hf_cfg = OmegaConf.load(f"{CKPT_DIR}/config.yaml")
    task_oracle = hydra.utils.instantiate(hf_cfg.callbacks.rollout_lh.tasks)
    val_annotations = hf_cfg.callbacks.rollout_lh.val_annotations

    rollout_cfg = OmegaConf.create({"ep_len": 360, "debug": False})
    print(f"model+env loaded in {load_s:.1f}s "
          f"(sampling_steps={model.num_sampling_steps}, "
          f"multistep={model.multistep})", flush=True)
    return model, env, task_oracle, val_annotations, rollout_cfg


def _p1_standard_chains():
    """{idx: (initial_state, sequence)} for the standard 1000-chain protocol."""
    from flower.evaluation.multistep_sequences import get_sequences
    return dict(enumerate(get_sequences(1000)))


@app.function(image=image, gpu=GPU_KIND, volumes={"/data": vol},
              timeout=43200, memory=24576)
def taxonomy_chains(chain_indices, repeats: int, tag: str, shard: int):
    import os

    _p1_shims()
    import torch  # noqa: F401  (P0-proven order: torch before cv2/numpy use)

    from flower.evaluation import flower_evaluate as fe

    import numpy as np
    import cv2
    from pytorch_lightning import seed_everything

    model, env, task_oracle, val_annotations, cfg = _p1_load_model_and_env()
    all_chains = _p1_standard_chains()

    out_dir = f"{RESULTS_ROOT}/{tag}"
    os.makedirs(f"{out_dir}/states", exist_ok=True)
    out_path = f"{out_dir}/repeats_shard{shard}.json"

    # ---- recording hooks (wrapper-side; vendor objects, methods rebound) ----
    orig_env_step = env.step
    orig_model_reset = model.reset

    class _Rec:
        step_i = 0
        boundaries = []
        robot = []
        scene = []
        capture = False
        frames_dir = None
    rec = _Rec()

    def rec_model_reset(*a, **kw):
        # rollout() calls model.reset() once per subtask -> boundary marker
        rec.boundaries.append(rec.step_i)
        return orig_model_reset(*a, **kw)

    def rec_env_step(action):
        obs, r, d, info = orig_env_step(action)
        rec.step_i += 1
        try:
            rec.robot.append(np.round(np.asarray(
                obs["robot_obs"], dtype=float), 3).tolist())
            rec.scene.append(np.round(np.asarray(
                obs["scene_obs"], dtype=float), 3).tolist())
        except Exception as e:
            if rec.step_i == 1:
                print(f"state capture unavailable: {e}", flush=True)
        if rec.capture and rec.frames_dir and rec.step_i % 12 == 0:
            try:
                fr = np.asarray(obs["rgb_obs"]["rgb_static"])
                if fr.dtype != np.uint8:
                    fr = np.clip(fr, 0, 255).astype(np.uint8)
                cv2.imwrite(f"{rec.frames_dir}/step_{rec.step_i:05d}.jpg",
                            cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            except Exception as e:
                if rec.step_i <= 12:
                    print(f"frame capture unavailable: {e}", flush=True)
        return obs, r, d, info

    model.reset = rec_model_reset
    env.step = rec_env_step

    results = []
    t_run0 = time.time()
    for ci in chain_indices:
        initial_state, eval_sequence = all_chains[ci]
        for rep in range(repeats):
            # fresh, recorded noise stream per repeat (env init state stays
            # vendor-deterministic from initial_state; only policy noise moves)
            seed = 300000 + ci * 10 + rep
            seed_everything(seed, workers=True)
            rec.step_i, rec.boundaries, rec.robot, rec.scene = 0, [], [], []
            rec.capture = rep == 0
            if rec.capture:
                rec.frames_dir = f"{out_dir}/frames/chain_{ci:04d}"
                os.makedirs(rec.frames_dir, exist_ok=True)
            t0 = time.time()
            n_success = fe.evaluate_sequence(
                env, model, task_oracle, initial_state, list(eval_sequence),
                None, val_annotations, cfg, False, None, ci,
            )
            dt = time.time() - t0
            results.append({
                "chain_idx": ci, "repeat": rep, "seed": seed,
                "n_success": int(n_success), "sequence": list(eval_sequence),
                "wall_s": round(dt, 1),
                "subtask_start_steps": list(rec.boundaries),
                "total_steps": rec.step_i,
            })
            if rec.capture:
                with open(f"{out_dir}/states/chain_{ci:04d}.json", "w") as f:
                    json.dump({
                        "chain_idx": ci, "seed": seed,
                        "sequence": list(eval_sequence),
                        "n_success": int(n_success),
                        "subtask_start_steps": list(rec.boundaries),
                        "robot_obs": rec.robot, "scene_obs": rec.scene,
                    }, f)
            print(f"[{TAG} s{shard}] chain {ci} rep {rep}: {n_success}/5 "
                  f"in {dt:.0f}s", flush=True)
        with open(out_path, "w") as f:
            json.dump({"shard": shard, "results": results}, f, indent=1)
        vol.commit()

    total_s = time.time() - t_run0
    print(f"[{TAG} s{shard}] done: {len(chain_indices)} chains x {repeats} "
          f"in {total_s / 3600:.2f} h", flush=True)
    return {"shard": shard, "results": results, "shard_wall_s": total_s}


@app.local_entrypoint()
def main(repeats: int = 3, shards: int = 4):
    fails = json.load(open(FAILS_PATH))
    idxs = sorted(f["chain_idx"] for f in fails)
    print(f"{len(idxs)} breaking chains, {repeats} repeats, {shards} shards")
    # round-robin so slow chains spread across shards
    shard_lists = [idxs[s::shards] for s in range(shards)]
    args = [(shard_lists[s], repeats, TAG, s) for s in range(shards)]

    all_results, hours = [], 0.0
    for res in taxonomy_chains.starmap(args, order_outputs=True):
        all_results.extend(res["results"])
        hours += res["shard_wall_s"] / 3600

    all_results.sort(key=lambda r: (r["chain_idx"], r["repeat"]))
    p0_n = {f["chain_idx"]: f["n_success"] for f in fails}
    per_chain = {}
    for r in all_results:
        per_chain.setdefault(r["chain_idx"], []).append(r["n_success"])
    summary = {
        "tag": TAG, "repeats": repeats,
        "num_chains": len(per_chain),
        "chains_all_repeats_broke": sorted(
            ci for ci, ns in per_chain.items() if all(n < 5 for n in ns)),
        "chains_any_repeat_full": sorted(
            ci for ci, ns in per_chain.items() if any(n == 5 for n in ns)),
        "per_chain": {str(ci): {"p0_n_success": p0_n[ci], "repeat_n_success": ns}
                      for ci, ns in sorted(per_chain.items())},
        "gpu_container_hours": round(hours, 2),
        "approx_gpu_cost_usd": round(hours * GPU_RATE_PER_HR, 2),
    }
    summary["n_systematic"] = len(summary["chains_all_repeats_broke"])
    summary["n_luck_recoverable"] = len(summary["chains_any_repeat_full"])
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chain"},
                     indent=2))
    out_dir = pathlib.Path.home() / "cleanroom_calvin" / "results"
    with open(out_dir / f"{TAG}_summary.json", "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=1)
    print(f"summary written to {out_dir}/{TAG}_summary.json")
