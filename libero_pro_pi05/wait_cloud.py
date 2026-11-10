import json, pathlib, time, modal
ROOM = pathlib.Path.home() / "cleanroom_pi05_liberopro" / "results"
import sys; calls = json.load(open(ROOM / (sys.argv[1] if len(sys.argv) > 1 else "cloud_calls.json")))
pending = {c[2]: c for c in calls}; out = []
while pending:
    for cid, c in list(pending.items()):
        try:
            rc = modal.FunctionCall.from_id(cid).get(timeout=0); out.append((c[0], c[1], rc)); del pending[cid]
        except TimeoutError:
            pass
        except Exception as e:
            out.append((c[0], c[1], f"ERR {type(e).__name__}")); del pending[cid]
    if pending:
        time.sleep(120)
print("ALL SHARDS DONE:", out)
