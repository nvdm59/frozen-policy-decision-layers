"""CALVIN environment on Modal Linux — bring-up smoke test.

Builds an image with calvin_env (pybullet sim) and pulls the official debug
dataset (~1.3GB) into a Volume. Smoke = import calvin_env, create an env
headless, reset, step. Policy serving (FLOWER) comes later, separate app.

Run: modal run modal_calvin_env.py
"""
import modal

app = modal.App("calvin-env-smoke")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "unzip", "libgl1", "libglib2.0-0", "libegl1",
                 "libgles2", "libosmesa6", "freeglut3-dev", "build-essential",
                 "cmake", "ffmpeg")
    .run_commands(
        "git clone --recurse-submodules --depth 1 https://github.com/mees/calvin.git /opt/calvin",
        "pip install -e /opt/calvin/calvin_env/tacto",
        "pip install -e /opt/calvin/calvin_env",
        "pip install 'numpy<1.24' hydra-core==1.1.1 opencv-python-headless",
    )
)

vol = modal.Volume.from_name("calvin-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch_debug_dataset():
    import os
    import subprocess
    if os.path.exists("/data/calvin_debug_dataset"):
        return "dataset already present"
    subprocess.run(
        ["wget", "-q", "http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip",
         "-O", "/data/debug.zip"], check=True)
    subprocess.run(["unzip", "-q", "/data/debug.zip", "-d", "/data"], check=True)
    os.remove("/data/debug.zip")
    vol.commit()
    return "debug dataset fetched"


@app.function(image=image, volumes={"/data": vol}, timeout=1200)
def smoke_env():
    # py3.9-era code compat: collections.* aliases removed in py3.10
    import collections
    import collections.abc
    for _n in dir(collections.abc):
        if not _n.startswith("_") and not hasattr(collections, _n):
            setattr(collections, _n, getattr(collections.abc, _n))
    import fractions
    import math
    if not hasattr(fractions, "gcd"):
        fractions.gcd = math.gcd

    import hydra
    from omegaconf import OmegaConf
    import calvin_env  # noqa
    # minimal headless env instantiation from the debug dataset's own config
    import os
    root = "/data/calvin_debug_dataset"
    assert os.path.exists(root), "dataset missing — run fetch_debug_dataset first"
    cfg_path = os.path.join(root, "validation/.hydra/merged_config.yaml")
    cfg = OmegaConf.load(cfg_path)
    env_cfg = cfg.callbacks.rollout.env_cfg if "callbacks" in cfg else cfg.env
    from calvin_env.envs.play_table_env import PlayTableSimEnv
    OmegaConf.set_struct(env_cfg, False)
    if "cameras" in env_cfg:
        env_cfg["cameras"] = OmegaConf.create(
            {k: v for k, v in env_cfg["cameras"].items() if "tactile" not in k})
    kwargs = {k: env_cfg[k] for k in env_cfg
              if k not in ("_target_", "_recursive_")}
    kwargs.update(show_gui=False, use_vr=False, use_egl=False)
    env = PlayTableSimEnv(**kwargs)
    obs = env.reset()
    print("env reset OK; obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))
    return "CALVIN_ENV_SMOKE_OK"


@app.local_entrypoint()
def main():
    print(fetch_debug_dataset.remote())
    print(smoke_env.remote())
