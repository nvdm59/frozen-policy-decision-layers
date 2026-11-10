"""spawn arm shards: python launch_arms.py <tag> <suite1,suite2> <arms> <trials> <trial0> <shards>"""
import sys, json, time, pathlib, modal
tag, suites, arms, trials, trial0, shards = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
f = modal.Function.from_name("pi05-liberopro-sim", "arm_shard")
calls = []
for s in suites:
    for a in arms:
        for w in range(shards):
            c = f.spawn(s, a, trials, w, shards, tag, trial0, False, (sys.argv[7] if len(sys.argv) > 7 else "layer_v1"), (sys.argv[8] if len(sys.argv) > 8 else ""))
            calls.append([s, f"{a}_w{w}", c.object_id])
p = pathlib.Path.home() / "cleanroom_pi05_liberopro" / "results" / f"calls_{tag}.json"
json.dump(calls, open(p, "w"))
print(f"spawned {len(calls)} shards -> {p.name} at", time.strftime("%H:%M"))
