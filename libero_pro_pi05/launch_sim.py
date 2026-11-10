"""Spawn cloud shards (deployed app; survives laptop sleep). Per-episode resume on volume."""
import json, pathlib, sys, time
import modal
ROOM = pathlib.Path.home() / "cleanroom_pi05_liberopro" / "results"
pro = modal.Function.from_name("pi05-liberopro-sim", "pro_shard")
sw = modal.Function.from_name("pi05-liberopro-sim", "switch_shard")
calls = []
# Step 2: instruction-switch feasibility (goal suite = dev), 4 shards
for w in range(4):
    calls.append(("switch", w, sw.spawn(10, w, 4, "p2_switch").object_id))
# Step 3 (layer-independent part): π0.5-alone baselines on HELD-OUT changed-task suites, 20 trials/task
for suite in ("libero_spatial_task", "libero_object_task", "libero_10_task"):
    for w in range(4):
        calls.append((suite, w, pro.spawn(suite, 20, w, 4, "p3_base", 7, True).object_id))
json.dump(calls, open(ROOM / "cloud_calls.json", "w"), indent=1)
print(f"spawned {len(calls)} shards at {time.strftime('%H:%M')}")
