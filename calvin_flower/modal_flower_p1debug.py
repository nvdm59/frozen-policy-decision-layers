"""Staged import diagnostics for the P1 SIGILL + calvin_env-shadow failures.

Stage prints are flushed so a SIGILL death pinpoints the killing import.
Two modes: order 'p1' (numpy/cv2 before torch — the failing P1 order) and
order 'p0' (torch first — P0's proven order).
"""
import json
import sys
import time

import modal

from modal_flower_eval import (  # noqa: F401
    GPU_KIND, image, vol, _apply_py310_shims, _load_model_and_env,
)

app = modal.App("flower-calvin-p1debug")


@app.function(image=image, gpu=GPU_KIND, volumes={"/data": vol},
              timeout=3600, memory=24576)
def debug_imports(order: str):
    import subprocess

    def mod_state(label):
        m = sys.modules.get("calvin_env")
        print(f"[{label}] calvin_env in sys.modules={m is not None} "
              f"file={getattr(m, '__file__', 'ABSENT')} "
              f"path_entry={'/opt/flower/calvin_env' in sys.path}", flush=True)

    cpu = subprocess.run(["grep", "-m1", "flags", "/proc/cpuinfo"],
                         capture_output=True, text=True).stdout
    print("CPU flags:", " ".join(w for w in cpu.split()
                                 if "avx" in w or "sse4" in w), flush=True)
    mod_state("pre-shim")
    _apply_py310_shims()
    mod_state("post-shim")

    if order == "p1":
        print("STAGE import numpy", flush=True)
        import numpy  # noqa: F401
        print("STAGE import cv2", flush=True)
        import cv2  # noqa: F401
        print("STAGE import torch", flush=True)
        import torch  # noqa: F401
        print("STAGE import pytorch_lightning", flush=True)
        from pytorch_lightning import seed_everything  # noqa: F401
    else:
        print("STAGE import torch", flush=True)
        import torch  # noqa: F401
        print("STAGE import numpy", flush=True)
        import numpy  # noqa: F401
        print("STAGE import cv2", flush=True)
        import cv2  # noqa: F401
    mod_state("post-imports")

    print("STAGE import flower_evaluate", flush=True)
    from flower.evaluation import flower_evaluate as fe  # noqa: F401
    mod_state("post-flower")

    print("STAGE import play_table_env", flush=True)
    import calvin_env.envs.play_table_env as pte
    print("play_table_env module file:", pte.__file__, flush=True)
    print("play_table_env's calvin_env.__file__:",
          pte.calvin_env.__file__ if hasattr(pte, "calvin_env") else "no attr",
          flush=True)
    print("nvidia-smi:", subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip(),
        flush=True)

    print("STAGE _load_model_and_env", flush=True)
    model, env, task_oracle, val_annotations, cfg = _load_model_and_env()
    print("LOAD_OK", flush=True)
    obs = env.get_obs()
    print("obs keys:", list(obs.keys()),
          "rgb:", {k: getattr(v, 'shape', None)
                   for k, v in obs.get("rgb_obs", {}).items()},
          "robot_obs:", getattr(obs.get("robot_obs"), 'shape', None),
          "scene_obs:", getattr(obs.get("scene_obs"), 'shape', None),
          flush=True)
    info = env.get_info()
    print("info keys:", list(info.keys()), flush=True)
    return {"order": order, "ok": True}


@app.local_entrypoint()
def main(order: str = "p1"):
    t0 = time.time()
    try:
        out = debug_imports.remote(order)
        print(json.dumps(out))
    except Exception as e:
        print("REMOTE FAILED:", type(e).__name__, e)
    print(f"wall {(time.time() - t0) / 60:.1f} min")
