#!/usr/bin/env bash
# Step 1 setup for π0.5 × LIBERO-PRO. Creates an ISOLATED venv; the July openpi_poc/.venv is untouched.
# Installs: the same packages as openpi_poc/.venv (pinned freeze), openpi-client (PI's GitHub, pinned commit),
# and LIBERO-PRO (github.com/Zxy-MLlab/LIBERO-PRO, editable, --no-deps).
set -euo pipefail
P=~/claude-code/python/pi05_liberopro
S=${SCRATCH:-/tmp/pi_libero}
mkdir -p "$P/vendor"
[ -d "$P/vendor/LIBERO-PRO" ] || cp -R "$S/LIBERO-PRO" "$P/vendor/"
cd "$P"
uv venv -q -p 3.12 .venv
uv pip install -q -p .venv/bin/python -r "$S/freeze.txt"
uv pip install -q -p .venv/bin/python \
  "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git@15a9616a00943ada6c20a0f158e3adb39df2ccac#subdirectory=packages/openpi-client"
uv pip install -q -p .venv/bin/python --no-deps -e vendor/LIBERO-PRO
.venv/bin/python -c "import libero.libero.benchmark as b; print('libero_goal_task registered:', 'libero_goal_task' in b.get_benchmark_dict())"
