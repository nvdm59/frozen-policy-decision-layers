import modal, json
f = modal.Function.from_name("pi05-liberopro-sim", "parse_check")
suites = ["libero_goal_task", "libero_spatial_task", "libero_object_task", "libero_10_task"]
for s, res in zip(suites, f.map(suites)):
    for l in res:
        try:
            d = json.loads(l); print(d["suite"][:-5], d["task"], "|", d["instr"], "->", d["targets"], "| goal:", d["goal"])
        except Exception:
            print(l)
