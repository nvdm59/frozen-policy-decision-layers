"""Layer v2 (designed on dev after v1 failed): training-free grasp-verify-switch decision layer around a FROZEN π0.5 (no weights touched).

Generic: no per-task prompt tables. Everything is derived from (instruction text, scene object list,
simulator object poses) — the same information class as the CALVIN layer's contact checks.

  1. parse_targets(): grasp targets = movable scene objects named in the instruction that are NOT
     preceded by a spatial preposition (those are destinations / references). Duplicate instances
     (two bowls, two moka pots, two plates) are resolved from the relation phrase after the noun
     (between / not between / next to / on / from table center / left / right) against sim poses.
  2. Phase A prompt  = a TRAINED instruction (train_pool.json: the 40 original LIBERO-4 instructions) whose
     scene signature equals this scene and whose first goal object is the target; put/pick verbs only;
     highest dev reliability (P0 original-suite success), ties alphabetical. No candidate -> instruction
     unchanged for that target (layer == base there).
  3. Precondition check = the TARGET is lifted >= LIFT_M -> switch. A different object lifted first, or
     A_TIMEOUT steps without a target lift -> switch to the instruction (no retries, no release macro).
  4. On verified grasp -> switch to the ORIGINAL (changed) instruction and replan immediately.
  5. Multi-target instructions: after the gripper has been open for RELEASE_N steps post-switch,
     advance to the next target's phase A.
Arms: base (instruction throughout) | blind (phase A for BLIND_T steps, then switch, no checks) | layer.
Budget: identical step budget for every arm (the release macro counts against it)."""
import collections
import json
import os
import re

import numpy as np

LIFT_M = 0.05
A_TIMEOUT = 150
MAX_RETRY = 0
POOL = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pool.json")))
BLIND_T = 52            # median verified-switch step in the step-2 dev probe
RELEASE_N = 10
PREPS = ("on top of", "on the top of", "to the right of", "to the left of", "next to", "between",
         "in front of", "in the", "on the", "into the", "onto the", "inside", "of the", "on", "in",
         "into", "onto", "to")
SYN = {"mug": ["cup"], "frypan": ["pan", "frying pan"], "porcelain mug": ["white mug"],
       "white yellow mug": ["yellow and white mug", "white and yellow mug"],
       "red coffee mug": ["red mug"], "wine bottle": ["bottle"], "akita black bowl": ["black bowl"],
       "black book": ["book"], "cream cheese": ["cream cheese box"], "plate": ["dish"], "cookies": ["cookie box", "cookies box"]}


def _base(name):
    return re.sub(r"_\d+$", "", name).replace("_", " ")


def aliases(name):
    b = re.sub(r"^chefmate \d+ ", "", _base(name))
    out = {b, b.split()[-1]}
    if len(b.split()) > 1:
        out.add(" ".join(b.split()[-2:]))
    for k, v in SYN.items():
        if b == k or b.endswith(k):
            out.update(v)
    return sorted(out, key=len, reverse=True)


def _mentions(text, names):
    """[(start, end, name_category, alias)] longest-match, non-overlapping."""
    hits = []
    for n in names:
        for a in aliases(n):
            for m in re.finditer(r"\b" + re.escape(a) + r"\b", text):
                hits.append((m.start(), m.end(), _base(n), a))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out, last = [], -1
    for h in hits:
        if h[0] >= last:
            out.append(h)
            last = h[1]
    return out


def _strip_tail(before):
    while True:
        b2 = re.sub(r"\b(the|a|both|and|left|right)\s*$", "", before).rstrip()
        if b2 == before:
            return before
        before = b2


def _preceded_by_prep(text, start):
    words = _strip_tail(text[:start].rstrip()).split()
    return any(" ".join(words[-k:]) in PREPS for k in (1, 2, 3, 4) if len(words) >= k)


def pos(env, name):
    e = env.env if hasattr(env, "env") else env
    return np.array(e.sim.data.body_xpos[e.obj_body_id[name]])


def _resolve(env, cat, rel, movable, all_names):
    cands = [n for n in movable if _base(n) == cat]
    if len(cands) <= 1:
        return cands[0] if cands else None
    P = {n: pos(env, n) for n in cands}
    refs = [h for h in _mentions(rel, [n for n in all_names if _base(n) != cat])]
    refpos = []
    for h in refs:
        inst = [n for n in all_names if _base(n) == h[2]]
        if inst:
            refpos.append(pos(env, inst[0]))
    xy = lambda a, b: float(np.linalg.norm(a[:2] - b[:2]))
    if rel.startswith("LEFT "):
        return max(cands, key=lambda n: P[n][1])
    if rel.startswith("RIGHT "):
        return min(cands, key=lambda n: P[n][1])
    if "between" in rel and len(refpos) >= 2:
        A, B = refpos[0][:2], refpos[1][:2]

        def seg(n):   # xy distance from the instance to segment A-B
            p = P[n][:2]
            u = float(np.clip(np.dot(p - A, B - A) / max(np.dot(B - A, B - A), 1e-9), 0, 1))
            return float(np.linalg.norm(p - (A + u * (B - A))))
        f = (lambda n: -seg(n)) if re.search(r"\bnot between\b", rel) else seg
        return min(cands, key=f)
    if refpos:
        return min(cands, key=lambda n: xy(P[n], refpos[0]))
    if "center" in rel:
        c = np.mean([pos(env, n) for n in movable], axis=0)
        return min(cands, key=lambda n: xy(P[n], c))
    if re.search(r"\bleft\b", rel):
        return max(cands, key=lambda n: P[n][1])
    if re.search(r"\bright\b", rel):
        return min(cands, key=lambda n: P[n][1])
    return sorted(cands)[0]


def parse_targets(env, instr):
    """-> [(object_name, phrase)] in instruction order; [] if not a pick/put instruction."""
    e = env.env if hasattr(env, "env") else env
    text = instr.lower().strip()
    if not re.search(r"\b(put|pick|place)\b", text):
        return []
    movable = list(e.objects_dict)
    all_names = movable + list(e.fixtures_dict)
    hits = _mentions(text, movable)
    out, rel_end = [], -1
    for i, (s, en, cat, a) in enumerate(hits):
        if s < rel_end or _preceded_by_prep(text, s):
            continue
        nxt = re.search(r"\b(and (put|place|close)|and place|and put)\b", text[en:])
        rel = text[en: en + nxt.start()] if nxt else text[en:]
        if re.match(r"\s*(not between|between|next to)\b", rel):
            rel_end = en + len(rel)
        if not text.startswith("pick"):   # 'put X on Y': Y is a destination, keep explicit locatives only
            m = re.match(r"\s*(not between .*|between .*|next to .*|from .*)", rel)
            rel = m.group(1) if m else ""
        adj = re.search(r"\b(left|right)\s+$", text[:s])
        if adj:
            rel = adj.group(1).upper() + " " + rel
        name = _resolve(env, cat, rel, movable, all_names)
        if name and name not in [o[0] for o in out]:
            loc = rel.strip()
            loc = loc.split(" ", 1)[1] if loc.startswith(("LEFT ", "RIGHT ")) else loc
            pre = (adj.group(1) + " ") if adj else ""
            phrase = pre + text[s:en] + ((" " + loc) if loc.startswith(
                ("between", "not between", "next to", "on ", "from", "in the")) else "")
            out.append((name, re.sub(r"\s+", " ", phrase).strip()))
    return out


def candidate(env, target):
    e = env.env if hasattr(env, "env") else env
    pp = e.parsed_problem   # scene signature from the parsed bddl (same source as train_pool.json)
    scene = sorted({n for sec in ("fixtures", "objects") for v in pp[sec].values() for n in v})
    c = [q for q in POOL if q["scene"] == scene and q["goal_first"] == target
         and re.match(r"\s*(put|pick)\b", q["instr"].lower())]
    c.sort(key=lambda q: (-(q["reliability"] if q["reliability"] is not None else 0.5), q["instr"]))
    return c[0]["instr"] if c else None


class Layer:
    """Stateful per-episode controller; call prompt_for(t, obs, env) before each replan decision.
    It may return a macro action list (release) that the runner must execute first."""

    def __init__(self, env, instr, arm):
        self.instr, self.arm = instr, arm
        e = env.env if hasattr(env, "env") else env
        self.movable = list(e.objects_dict)
        self.targets = parse_targets(env, instr) if arm != "base" else []
        self.cands = [candidate(env, t[0]) for t in self.targets]
        self.idx, self.mode, self.t_phase, self.retries = 0, ("A" if self.targets else "B"), 0, 0
        self.rest = {n: pos(env, n)[2] for n in self.movable}
        self.done_t, self.open_run, self.events = set(), 0, []

    def _lifted(self, env):
        return [n for n in self.movable
                if n not in self.done_t and pos(env, n)[2] - self.rest[n] >= LIFT_M]

    def prompt(self):
        if self.mode == "A":
            return self.cands[self.idx]
        return self.instr

    def step(self, t, env, last_action):
        """Update state; returns (changed_prompt: bool, macro: list|None)."""
        if not self.targets:
            return False, None
        if self.mode == "A" and self.cands[self.idx] is None:
            self.mode = "B"
            self.events.append(("no_candidate", t, self.targets[self.idx][0]))
            return t > 0, None
        if self.mode == "A":
            tgt = self.targets[self.idx][0]
            if self.arm == "blind":
                if t >= BLIND_T:
                    self.mode = "B"
                    self.events.append(("switch_blind", t))
                    return True, None
                return False, None
            lifted = self._lifted(env)
            if tgt in lifted:
                self.mode = "B"
                self.events.append(("switch", t, tgt))
                self.t_phase, self.open_run = t, 0
                return True, None
            if lifted and self.retries < MAX_RETRY:
                self.retries += 1
                self.events.append(("wrong_grasp", t, lifted[0]))
                return True, ([[0.0] * 6 + [-1.0]] * 10 + [[0.0, 0.0, 0.6, 0.0, 0.0, 0.0, -1.0]] * 10
                              + [[0.0] * 6 + [-1.0]] * 5)
            if t - self.t_phase >= A_TIMEOUT or (lifted and self.retries >= MAX_RETRY):
                self.mode = "B"
                self.events.append(("switch_fallback", t))
                return True, None
            return False, None
        # mode B, layer only: advance to next target once released
        if self.arm == "layer" and self.idx + 1 < len(self.targets):
            self.open_run = self.open_run + 1 if (last_action is not None and last_action[-1] < 0) else 0
            if self.open_run >= RELEASE_N and t - self.t_phase >= 20:
                self.done_t.add(self.targets[self.idx][0])
                self.idx += 1
                self.mode, self.t_phase, self.retries = "A", t, 0
                self.events.append(("next_target", t, self.targets[self.idx][0]))
                return True, None
        return False, None

    def rebaseline(self, env):
        for n in self.movable:
            if n not in self.done_t:
                self.rest[n] = pos(env, n)[2]
