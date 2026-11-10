"""FLOWER VLA evaluation on CALVIN chains — Modal A10G app.

P0 goal: reproduce the published ABC->D baseline (avg seq len 4.54;
chain successes 99.3/95.9/90.5/84.8/77.5) on a 200-chain slice of the
standard 1000-chain protocol.

Design notes (see ~/cleanroom_calvin/JOURNAL.md):
- Vendor eval code (flower/evaluation/flower_evaluate.py) is used for ALL
  eval semantics: get_sequences, evaluate_sequence, rollout, task oracle,
  initial-state construction (pyhash-seeded). We only replace their
  wandb-entangled main()/print_and_save with plain JSON logging.
- Checkpoint + debug dataset live in Modal volume `calvin-data`:
  /data/flower_calvin_abc (HF format), /data/calvin_debug_dataset
  (validation = scene D -> sufficient for rollout eval; no big download).
- Chains = get_sequences(1000)[start:end]  (standard protocol, sliced).

Run:
  modal run modal_flower_eval.py --mode smoke            # 1 chain + timing
  modal run modal_flower_eval.py --mode eval --n 10      # 10-chain slice
  modal run modal_flower_eval.py --mode p0 --shards 4    # 200-chain P0
"""
import json
import time

import modal

app = modal.App("flower-calvin-eval")

VOL_NAME = "calvin-data"
CKPT_DIR = "/data/flower_calvin_abc"
DATASET_DIR = "/data/calvin_debug_dataset"
RESULTS_ROOT = "/data/results"
GPU_KIND = "A10G"
GPU_RATE_PER_HR = 1.10  # approx, for the spend ledger


def _download_florence():
    """Bake Florence-2-large into the image's HF cache (build-time, CPU)."""
    from unittest.mock import patch

    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_imports

    def _no_flash_attn(filename):
        return [i for i in get_imports(filename) if i != "flash_attn"]

    with patch("transformers.dynamic_module_utils.get_imports", _no_flash_attn):
        AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True)
        AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True)
    print("Florence-2-large cached")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "unzip", "ffmpeg", "build-essential", "cmake",
        "libgl1", "libglib2.0-0", "libegl1", "libgles2", "libosmesa6",
        "freeglut3-dev", "libegl1-mesa-dev", "libgles2-mesa-dev",
    )
    .env({
        "HF_HOME": "/root/hf_cache",
        "WANDB_MODE": "disabled",
        "CI": "1",  # keeps HulcDataModule.prepare_data non-interactive
        # default Modal caps exclude 'graphics' -> EGL rendering would fail
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
    })
    .pip_install("torch==2.2.2", "torchvision==0.17.2")
    .run_commands(
        "git clone --recurse-submodules --depth 1 "
        "https://github.com/intuitive-robots/flower_vla_calvin.git /opt/flower "
        "&& git -C /opt/flower rev-parse HEAD | tee /opt/flower_commit.txt",
        "pip install -e /opt/flower/calvin_env/tacto",
        "pip install -e /opt/flower/calvin_env",
        "pip install -r /opt/flower/requirements.txt",
        "pip install -e /opt/flower",
        "pip install transformers==4.44.2",
        # vendored pyhash (their patched 0.9.3) needs old setuptools to build
        "pip install setuptools==57.5.0 && cd /opt/flower/pyhash-0.9.3 "
        "&& python setup.py build && python setup.py install",
        # compat pins LAST so nothing bumps them (np.float removal, numba ABI)
        "pip install numpy==1.23.5 numba==0.58.1 scipy==1.10.1 "
        "opencv-python-headless==4.8.1.78",
        # EGL probe helper used by HulcWrapper.set_egl_device (best effort)
        "cd /opt/flower/calvin_env/egl_check && bash build.sh || true",
    )
    .run_function(_download_florence)
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


# ---------------------------------------------------------------------------
# remote helpers (only imported inside container)
# ---------------------------------------------------------------------------

def _apply_py310_shims():
    """py3.9-era vendor code on py3.10 (pattern from modal_calvin_env.py)."""
    import collections
    import collections.abc
    for _n in dir(collections.abc):
        if not _n.startswith("_") and not hasattr(collections, _n):
            setattr(collections, _n, getattr(collections.abc, _n))
    import fractions
    import math
    if not hasattr(fractions, "gcd"):
        fractions.gcd = math.gcd
    # IMPORTANT: make the real calvin_env package win over the namespace
    # shadow. flower's PEP-660 editable installs leave no sys.path entry for
    # the calvin_env package dir, while /opt/flower (added by flower's own
    # editable .pth, and again by flower_evaluate's sys.path.insert) exposes
    # /opt/flower/calvin_env as an __init__-less namespace portion
    # (__file__=None -> PlayTableSimEnv crashes). A regular package found on
    # sys.path always beats a namespace portion (PEP 420), so expose the
    # submodule repo root and import it first.
    import sys
    if "/opt/flower/calvin_env" not in sys.path:
        sys.path.insert(0, "/opt/flower/calvin_env")
    import calvin_env  # noqa: F401
    assert calvin_env.__file__ is not None, "calvin_env namespace-shadowed"


def _load_model_and_env():
    """Load FLOWER + CALVIN env via vendor loaders. Returns (model, env, cfg_bits)."""
    from unittest.mock import patch

    from omegaconf import OmegaConf
    import hydra
    from pytorch_lightning import seed_everything
    from transformers.dynamic_module_utils import get_imports

    # vendor protocol seeds once, before model construction
    seed_everything(0, workers=True)

    from flower.evaluation.utils import get_default_mode_and_env

    eval_cfg_overwrite = OmegaConf.create({
        "datamodule": {"datasets": {"lang_dataset": {
            "lang_folder": "lang_annotations",       # debug dataset layout
            "use_extracted_rel_actions": False,      # no extracted/ files
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
    # same post-load settings as vendor main() / eval_calvin.yaml
    model.num_sampling_steps = 4
    model.multistep = 10
    model.eval()

    # task oracle + language annotations from the checkpoint's own config
    hf_cfg = OmegaConf.load(f"{CKPT_DIR}/config.yaml")
    task_oracle = hydra.utils.instantiate(hf_cfg.callbacks.rollout_lh.tasks)
    val_annotations = hf_cfg.callbacks.rollout_lh.val_annotations

    rollout_cfg = OmegaConf.create({"ep_len": 360, "debug": False})
    print(f"model+env loaded in {load_s:.1f}s "
          f"(sampling_steps={model.num_sampling_steps}, multistep={model.multistep})")
    return model, env, task_oracle, val_annotations, rollout_cfg


def _standard_chains(start, end):
    """Indices [start:end] of the standard 1000-chain protocol."""
    from flower.evaluation.multistep_sequences import get_sequences
    chains = get_sequences(1000)
    return list(enumerate(chains))[start:end]


@app.function(image=image, gpu=GPU_KIND, volumes={"/data": vol},
              timeout=43200, memory=24576)
def eval_chains(start: int, end: int, tag: str):
    """Evaluate standard chains [start:end); save per-chain JSON to volume."""
    import os

    _apply_py310_shims()
    import torch  # noqa: F401 (ensure CUDA context inside shims scope)

    from flower.evaluation import flower_evaluate as fe

    model, env, task_oracle, val_annotations, cfg = _load_model_and_env()
    chains = _standard_chains(start, end)

    out_dir = f"{RESULTS_ROOT}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/chains_{start:04d}_{end:04d}.json"

    records = []
    t_run0 = time.time()
    for k, (idx, (initial_state, eval_sequence)) in enumerate(chains):
        t0 = time.time()
        n_success = fe.evaluate_sequence(
            env, model, task_oracle, initial_state, list(eval_sequence),
            None, val_annotations, cfg, False, None, idx,
        )
        dt = time.time() - t0
        rec = {
            "chain_idx": idx,
            "sequence": list(eval_sequence),
            "initial_state": {k2: v for k2, v in initial_state.items()},
            "n_success": int(n_success),
            "wall_s": round(dt, 1),
        }
        records.append(rec)
        done = [r["n_success"] for r in records]
        print(f"[{tag}] chain {idx} ({k + 1}/{len(chains)}): "
              f"{n_success}/5 in {dt:.0f}s | running avg len "
              f"{sum(done) / len(done):.3f}", flush=True)
        if (k + 1) % 10 == 0 or k + 1 == len(chains):
            with open(out_path, "w") as f:
                json.dump({"start": start, "end": end, "records": records,
                           "gpu": GPU_KIND}, f, indent=1)
            vol.commit()

    total_s = time.time() - t_run0
    print(f"[{tag}] shard done: {len(records)} chains in {total_s / 3600:.2f} h")
    return {"start": start, "end": end, "records": records,
            "shard_wall_s": total_s}


@app.function(image=image, gpu=GPU_KIND, volumes={"/data": vol},
              timeout=7200, memory=24576)
def smoke():
    """Load checkpoint, run ONE standard chain end-to-end, report timing."""
    _apply_py310_shims()
    import torch

    from flower.evaluation import flower_evaluate as fe

    model, env, task_oracle, val_annotations, cfg = _load_model_and_env()

    # per-step timing on a short burst before the real chain
    env.reset()
    obs = env.get_obs()
    goal = {"lang_text": "push the sliding door to the left side"}
    model.reset()
    torch.cuda.synchronize()
    t0 = time.time()
    n_burst = 30
    for _ in range(n_burst):
        action = model.step(obs, goal)
        obs, _, _, _ = env.step(action)
    torch.cuda.synchronize()
    per_step = (time.time() - t0) / n_burst
    print(f"timing burst: {per_step * 1000:.0f} ms/env-step "
          f"(model {model.num_sampling_steps} sampling steps, replan every "
          f"{model.multistep}) -> worst-case subtask (360 steps) "
          f"~{per_step * 360:.0f}s")

    idx, (initial_state, eval_sequence) = _standard_chains(0, 1)[0]
    print(f"chain 0 sequence: {list(eval_sequence)}; initial: {initial_state}")
    t0 = time.time()
    n_success = fe.evaluate_sequence(
        env, model, task_oracle, initial_state, list(eval_sequence),
        None, val_annotations, cfg, False, None, idx,
    )
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"SMOKE_OK chain0 result {n_success}/5 in {dt:.0f}s; "
          f"peak GPU mem {mem:.1f} GiB")
    return {"n_success": int(n_success), "chain_wall_s": dt,
            "ms_per_env_step": per_step * 1000, "gpu_mem_gib": mem}


def _summarize(records):
    n = len(records)
    results = [r["n_success"] for r in records]
    chain_sr = {}
    for i in range(1, 6):
        chain_sr[i] = sum(1 for r in results if r >= i) / n
    return {
        "num_chains": n,
        "avg_seq_len": sum(results) / n,
        "chain_success_rates": chain_sr,
        "published_abc_d": {"avg_seq_len": 4.54,
                            "chain_success_rates": [0.993, 0.959, 0.905,
                                                    0.848, 0.775]},
    }


@app.local_entrypoint()
def main(mode: str = "smoke", n: int = 10, start: int = 0,
         shards: int = 1, tag: str = None):
    t0 = time.time()
    if mode == "smoke":
        out = smoke.remote()
        print(json.dumps(out, indent=2))
    elif mode in ("eval", "p0"):
        if mode == "p0":
            start, n = 0, 200
        tag = tag or f"{mode}_{start}_{start + n}_{time.strftime('%Y%m%d_%H%M')}"
        bounds = []
        per = (n + shards - 1) // shards
        s = start
        while s < start + n:
            bounds.append((s, min(s + per, start + n), tag))
            s += per
        print(f"running {n} chains in {len(bounds)} shard(s): {bounds}")
        all_records = []
        shard_hours = 0.0
        for res in eval_chains.starmap(bounds, order_outputs=True):
            all_records.extend(res["records"])
            shard_hours += res["shard_wall_s"] / 3600
        all_records.sort(key=lambda r: r["chain_idx"])
        summary = _summarize(all_records)
        summary["tag"] = tag
        summary["gpu_container_hours"] = round(shard_hours, 2)
        summary["approx_gpu_cost_usd"] = round(shard_hours * GPU_RATE_PER_HR, 2)
        print(json.dumps(summary, indent=2))
        import pathlib
        out_dir = pathlib.Path.home() / "cleanroom_calvin" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{tag}_summary.json", "w") as f:
            json.dump({"summary": summary, "records": all_records}, f, indent=1)
        print(f"summary written to {out_dir}/{tag}_summary.json")
    else:
        raise SystemExit(f"unknown mode {mode}")
    print(f"local wall time {(time.time() - t0) / 60:.1f} min")
