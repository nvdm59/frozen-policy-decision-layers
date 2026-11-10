"""LIBERO-PRO simulator on Modal (CPU containers, OSMesa offscreen rendering). π0.5 stays on the
GPU server app pi05-libero-pro-server; each sim container opens its own websocket to it.
Nothing runs on the laptop except the launcher.

  modal deploy modal_sim.py
  python launch_sim.py <job>      # spawns shards; results persist per episode on volume pi05-liberopro
"""
import os
import pathlib
import subprocess

import modal

HERE = pathlib.Path(__file__).parent
PRO = HERE / "vendor" / "LIBERO-PRO"
R = "/opt/proj/vendor/LIBERO-PRO/libero/libero"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libosmesa6-dev", "libgl1", "libglib2.0-0", "ffmpeg", "libegl1")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(HERE / "requirements_sim.txt"))
    .pip_install("openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git"
                 "@15a9616a00943ada6c20a0f158e3adb39df2ccac#subdirectory=packages/openpi-client")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
          "LIBERO_CONFIG_PATH": "/opt/proj/.libero_container",
          "PYTHONPATH": "/opt/proj/vendor/LIBERO-PRO:/opt/proj"})
    .run_commands(
        "mkdir -p /opt/proj/.libero_container",
        f"printf 'assets: {R}/assets\\nbddl_files: {R}/bddl_files\\nbenchmark_root: {R}\\n"
        f"datasets: {R}/../datasets\\ninit_states: {R}/init_files\\n' > /opt/proj/.libero_container/config.yaml",
    )
    .add_local_dir(str(PRO), "/opt/proj/vendor/LIBERO-PRO",
                   ignore=[".git", "**/.git", "**/__pycache__", "**/*.mp4", "images", "notebooks"])
    .add_local_file(str(HERE / "pro_runner.py"), "/opt/proj/pro_runner.py")
    .add_local_file(str(HERE / "switch_probe.py"), "/opt/proj/switch_probe.py")
    .add_local_file(str(HERE / "safe_init.py"), "/opt/proj/safe_init.py")
    .add_local_file(str(HERE / "layer_v1.py"), "/opt/proj/layer_v1.py")
    .add_local_file(str(HERE / "layer_runner.py"), "/opt/proj/layer_runner.py")
    .add_local_file(str(HERE / "diag_runner.py"), "/opt/proj/diag_runner.py")
    .add_local_file(str(HERE / "p2_runner.py"), "/opt/proj/p2_runner.py")
    .add_local_file(str(HERE / "layer_v2.py"), "/opt/proj/layer_v2.py")
    .add_local_file(str(HERE / "layer_v3.py"), "/opt/proj/layer_v3.py")
    .add_local_file(str(HERE / "train_pool.json"), "/opt/proj/train_pool.json")
)
vol = modal.Volume.from_name("pi05-liberopro", create_if_missing=True)
app = modal.App("pi05-liberopro-sim", image=image)


def _run(cmd, results_subdir):
    # scripts write to /opt/proj/results/<tag>/...; point that at the volume so every episode persists
    os.makedirs("/data/results", exist_ok=True)
    link = pathlib.Path("/opt/proj/results")
    if not link.exists():
        link.symlink_to("/data/results")
    p = subprocess.Popen(cmd, cwd="/opt/proj", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    import time
    last = time.time()
    for line in p.stdout:
        if "WARNING" in line or "setup_macros" in line:
            continue
        print(line, end="", flush=True)
        if time.time() - last > 60:
            vol.commit()
            last = time.time()
    rc = p.wait()
    vol.commit()
    return rc


@app.function(volumes={"/data": vol}, cpu=4, memory=8192, timeout=6 * 3600)
def pro_shard(suite: str, trials: int, worker: int, workers: int, tag: str, seed: int = 7,
              videos: bool = False):
    cmd = ["python", "-u", "pro_runner.py", "--suite", suite, "--trials", str(trials),
           "--worker", str(worker), "--workers", str(workers), "--tag", tag, "--seed", str(seed)]
    if videos:
        cmd.append("--videos")
    return _run(cmd, tag)


@app.function(volumes={"/data": vol}, cpu=4, memory=8192, timeout=6 * 3600)
def switch_shard(trials: int, worker: int, workers: int, tag: str = "p2_switch"):
    return _run(["python", "-u", "switch_probe.py", "--trials", str(trials), "--worker", str(worker),
                 "--workers", str(workers), "--tag", tag], tag)


@app.function(volumes={"/data": vol}, cpu=4, memory=8192, timeout=1800)
def smoke():
    return _run(["python", "-u", "pro_runner.py", "--suite", "libero_goal_task", "--trials", "1",
                 "--worker", "0", "--workers", "10", "--tag", "smoke_cloud"], "smoke_cloud")


@app.function(volumes={"/data": vol}, cpu=4, memory=8192, timeout=8 * 3600)
def arm_shard(suite: str, arm: str, trials: int, worker: int, workers: int, tag: str,
              trial0: int = 0, videos: bool = False, layer: str = "layer_v1", tasks: str = ""):
    os.environ["LAYER"] = layer
    if tasks:
        os.environ["TASKS"] = tasks
    cmd = ["python", "-u", "layer_runner.py", "--suite", suite, "--arm", arm, "--trials", str(trials),
           "--trial0", str(trial0), "--worker", str(worker), "--workers", str(workers), "--tag", tag]
    if videos:
        cmd.append("--videos")
    return _run(cmd, tag)


@app.function(volumes={"/data": vol}, cpu=2, memory=4096, timeout=1800)
def parse_check(suite: str):
    import subprocess
    r = subprocess.run(["python", "-u", "layer_runner.py", "--suite", suite, "--arm", "layer",
                        "--tag", "parse", "--parse_only"], cwd="/opt/proj", capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.startswith("{")] or [r.stdout[-2000:], r.stderr[-3000:]]


@app.function(volumes={"/data": vol}, cpu=2, memory=4096, timeout=900)
def debug_cand(suite: str, task_id: int):
    import subprocess
    code = r'''
import json, os, sys
sys.path.insert(0, "/opt/proj")
import pro_runner as pr, layer_v2 as L
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()[sys.argv[1]](); tid = int(sys.argv[2])
task = suite.get_task(tid)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64); env.seed(7); env.reset()
e = env.env
pp = e.parsed_problem
scene = sorted({n for sec in ("fixtures", "objects") for v in pp[sec].values() for n in v})
print("FILE", L.__file__, "POOL", len(L.POOL))
print("SCENE", scene)
print("POOL_SCENES_SAME_SUITE", [q["scene"] for q in L.POOL if q["suite"] == "libero_goal"][:1])
print("TARGETS", L.parse_targets(env, pr.bddl_language(bddl)))
print("CAND", [L.candidate(env, t[0]) for t in L.parse_targets(env, pr.bddl_language(bddl))])
'''
    r = subprocess.run(["python", "-c", code, suite, str(task_id)], cwd="/opt/proj", capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l[:4] in ("FILE", "SCEN", "POOL", "TARG", "CAND")] + [r.stderr[-1500:]]


@app.function(volumes={"/data": vol}, cpu=4, memory=8192, timeout=8 * 3600)
def diag_shard(suite: str, trials: int, worker: int, workers: int, tag: str, tasks: str = ""):
    cmd = ["python", "-u", "diag_runner.py", "--suite", suite, "--trials", str(trials), "--worker", str(worker),
           "--workers", str(workers), "--tag", tag]
    if tasks:
        cmd += ["--tasks", tasks]
    return _run(cmd, tag)


@app.function(volumes={"/data": vol}, cpu=4, memory=8192, timeout=8 * 3600)
def p2_shard(suite: str, arm: str, trials: int, worker: int, workers: int, tag: str, tasks: str = ""):
    cmd = ["python", "-u", "p2_runner.py", "--suite", suite, "--arm", arm, "--trials", str(trials),
           "--worker", str(worker), "--workers", str(workers), "--tag", tag]
    if tasks:
        cmd += ["--tasks", tasks]
    return _run(cmd, tag)
