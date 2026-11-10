"""π0 LIBERO policy server on a Modal GPU — openpi's own websocket serving
pattern, hosted. The Mac runs the LIBERO sim + coordinator and talks to this
over openpi-client.

    modal deploy modal_pi0_server.py          # -> wss URL, scale-to-zero
    modal serve  modal_pi0_server.py          # dev

Checkpoint (π0 fine-tuned on LIBERO, from openpi's published assets) is
downloaded on first boot into a Volume, so cold starts after the first are
~1 min not ~10.

NOTE: this is batch-eval infra, NOT the demo API — min_containers=0 on
purpose (A10G ≈ $1.10/hr only while an eval is running).
"""
from __future__ import annotations

import modal

APP = "pi05-libero-pro-server"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "clang", "build-essential")
    .run_commands(
        "git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git /root/openpi",
        # openpi pins deps via uv; GPU extras resolve on linux.
        "pip install uv",
        "cd /root/openpi && uv pip install --system -e . 2>&1 | tail -5",
    )
    .env({"OPENPI_DATA_HOME": "/cache/openpi", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"})
)

app = modal.App(APP, image=image)
cache = modal.Volume.from_name("openpi-cache", create_if_missing=True)


@app.function(
    gpu="A10G",
    volumes={"/cache/openpi": cache},
    timeout=60 * 60 * 6,          # long eval sessions
    min_containers=0,             # scale to zero between eval runs
    max_containers=16,            # one container per eval worker
)
@modal.web_server(PORT, startup_timeout=60 * 15)  # first boot downloads the checkpoint
def policy_server():
    import subprocess
    # openpi's stock server: env=LIBERO -> pi0 LIBERO fine-tune from openpi-assets.
    subprocess.Popen(
        ["python", "scripts/serve_policy.py", "--env", "LIBERO",
         "--port", str(PORT)],
        cwd="/root/openpi",
    )
