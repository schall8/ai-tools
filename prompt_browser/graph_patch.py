"""
Patch a stored ComfyUI workflow graph before re-submitting:
  - set / randomize the seed
  - (best-effort) override the positive prompt

Reuses the trace helpers in lora_utilities/extract_render_metadata.py so prompt
location matches how the prompt was originally extracted.
"""
import os
import sys
import copy
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lora_utilities"))
import extract_render_metadata as ext  # noqa: E402

MAX_SEED = 2 ** 63 - 1


def set_seed(graph, seed):
    """Set every literal seed/noise_seed input. Returns the seed used."""
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs", {})
        for key in ("noise_seed", "seed"):
            if key in ins and not ext.is_link(ins[key]):
                ins[key] = seed
    return seed


def randomize_seed(graph):
    return set_seed(graph, random.randint(0, MAX_SEED))


def _string_target(graph, ref, depth=0, seen=None):
    """Find (node_id, input_key) holding the literal positive-prompt string, or
    None if it can't be set safely (e.g. built by a Text Concatenate node)."""
    if seen is None:
        seen = set()
    if depth > 24 or not ext.is_link(ref):
        return None
    nid = str(ref[0])
    if nid in seen:
        return None
    seen.add(nid)
    node = graph.get(nid)
    if not isinstance(node, dict):
        return None
    ins = node.get("inputs", {})
    ct = ext.node_ct(node)

    if any(ct == t or ct.startswith(t) for t in ext.TEXT_ENCODE):
        for key in ("text", "t5xxl", "t5xl", "clip_l", "text_g", "text_l"):
            if key in ins:
                v = ins[key]
                if isinstance(v, str):
                    return (nid, key)
                if ext.is_link(v):
                    r = _literal_target(graph, v, depth + 1, seen)
                    if r:
                        return r
        return None

    # passthrough conditioning node -> follow its conditioning input(s)
    for k, v in ins.items():
        if any(s in k.lower() for s in ("conditioning", "positive", "cond")) and ext.is_link(v):
            r = _string_target(graph, v, depth + 1, seen)
            if r:
                return r
    for k, v in ins.items():
        if ext.is_link(v):
            r = _string_target(graph, v, depth + 1, seen)
            if r:
                return r
    return None


def _literal_target(graph, ref, depth, seen):
    """Follow a link to a node whose literal text/string input can be set."""
    if depth > 24 or not ext.is_link(ref):
        return None
    nid = str(ref[0])
    node = graph.get(nid)
    if not isinstance(node, dict):
        return None
    ct = ext.node_ct(node)
    if "Concatenate" in ct:
        return None  # composed from multiple inputs -- don't clobber
    ins = node.get("inputs", {})
    for key in ("text", "string", "value", "Text", "String", "text_multiline", "prompt"):
        if key in ins:
            v = ins[key]
            if isinstance(v, str):
                return (nid, key)
            if ext.is_link(v):
                r = _literal_target(graph, v, depth + 1, seen)
                if r:
                    return r
    return None


def set_positive_prompt(graph, text):
    """Best-effort override of the positive prompt. Returns True on success."""
    _sid, snode = ext.find_output_sampler(graph)
    if not snode:
        return False
    ins = snode.get("inputs", {})
    ref = None
    if "positive" in ins and ext.is_link(ins["positive"]):
        ref = ins["positive"]
    elif "guider" in ins and ext.is_link(ins["guider"]):
        g = graph.get(str(ins["guider"][0]), {})
        gi = g.get("inputs", {}) if isinstance(g, dict) else {}
        for key in ("conditioning", "positive", "cond"):
            if key in gi and ext.is_link(gi[key]):
                ref = gi[key]
                break
    if not ref:
        return False
    target = _string_target(graph, ref)
    if not target:
        return False
    nid, key = target
    graph[nid]["inputs"][key] = text
    return True


def prepare(workflow, new_seed=None, randomize=False, positive_prompt=None):
    """Return (patched_graph_copy, info_dict). Never mutates the input."""
    graph = copy.deepcopy(workflow)
    info = {"seed": None, "prompt_set": None}
    if randomize:
        info["seed"] = randomize_seed(graph)
    elif new_seed is not None:
        info["seed"] = set_seed(graph, int(new_seed))
    if positive_prompt is not None:
        info["prompt_set"] = set_positive_prompt(graph, positive_prompt)
    return graph, info
