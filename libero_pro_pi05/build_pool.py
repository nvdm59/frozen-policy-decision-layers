"""Training-instruction pool for layer v2: the 40 ORIGINAL LIBERO-4 instructions (π0.5's fine-tuning set),
each with its scene signature (object+fixture instance names) and first goal object. Pure text parsing."""
import glob, json, os, re, collections
ORIG = os.path.expanduser("~/claude-code/python/openpi_poc/vendor/LIBERO/libero/libero/bddl_files")

def parse(path):
    s = open(path).read()
    names = set()
    for sec in ("fixtures", "objects"):
        m = re.search(r"\(:%s(.*?)\)" % sec, s, re.S)
        for line in m.group(1).strip().splitlines():
            if " - " in line:
                names.update(line.split(" - ")[0].split())
    goal = s.split("(:goal")[1]
    preds = re.findall(r"\((On|In) (\S+) (\S+)\)", goal)
    lang = re.search(r"\(:language\s+([^)]*)\)", s).group(1).strip()
    return {"instr": lang, "scene": sorted(names), "goal_first": preds[0][1] if preds else None,
            "n_preds": len(re.findall(r"\((?:On|In|Open|Close|Turnon|Turnoff) ", goal)), "file": os.path.basename(path)}

# reliability prior: π0.5 success on the ORIGINAL goal suite from P0 (dev data only)
rel = collections.defaultdict(list)
for f in glob.glob("results/p0/libero_goal_w*.jsonl"):
    for x in map(json.loads, open(f)):
        rel[x["prompt"].lower()].append(x["success"])
pool = []
for suite in ("libero_goal", "libero_spatial", "libero_object", "libero_10"):
    for p in sorted(glob.glob(f"{ORIG}/{suite}/*.bddl")):
        d = parse(p); d["suite"] = suite
        r = rel.get(d["instr"].lower())
        d["reliability"] = sum(r) / len(r) if r else None
        pool.append(d)
json.dump(pool, open("train_pool.json", "w"), indent=1)
print(len(pool), "training instructions;", sum(d["reliability"] is not None for d in pool), "with reliability")

# offline applicability on changed tasks (uses goal objects ONLY for this coverage report)
PRO = "vendor/LIBERO-PRO/libero/libero/bddl_files"
for suite in ("libero_goal_task", "libero_spatial_task", "libero_object_task", "libero_10_task"):
    for p in sorted(glob.glob(f"{PRO}/{suite}/*.bddl")):
        d = parse(p); s = open(p).read().split("(:goal")[1]
        objs = [o for _, o, _ in re.findall(r"\((On|In) (\S+) (\S+)\)", s)]
        cov = []
        for o in objs:
            c = [q["instr"] for q in pool if q["scene"] == d["scene"] and q["goal_first"] == o]
            cov.append((o, c[:2]))
        print(suite[:-5], "|", d["instr"], "->", cov)
