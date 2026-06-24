"""
extract_render_metadata.py

Walk a directory tree of ComfyUI-generated PNGs, read the embedded `prompt`
graph (PNG tEXt/iTXt/zTXt chunks) and extract generation metadata into a CSV
suitable for importing into a database or spreadsheet.

Columns:
    file_path, folder, filename, model_family, model_file, gen_type,
    positive_prompt, negative_prompt, loras, lora_count,
    width, height, steps, sampler, scheduler, guidance, seed, denoise

Pure standard library -- no extra dependencies, runs in any Python 3.8+ env.

Usage:
    python extract_render_metadata.py -i "E:/DATA/renders" -o renders.csv
    python extract_render_metadata.py -i "E:/DATA/renders" --limit 50   (test run)
"""

import os
import re
import csv
import json
import zlib
import struct
import argparse


# --------------------------------------------------------------------------- #
# PNG text-chunk reading
# --------------------------------------------------------------------------- #
def read_png_text_chunks(path):
    """Return {keyword: text} for tEXt/iTXt/zTXt chunks, or None if not a PNG."""
    out = {}
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                length, ctype = struct.unpack(">I4s", hdr)
                data = f.read(length)
                f.read(4)  # crc
                ctype = ctype.decode("latin1")
                try:
                    if ctype == "tEXt":
                        k, _, v = data.partition(b"\x00")
                        out[k.decode("latin1")] = v.decode("utf-8", "replace")
                    elif ctype == "zTXt":
                        k, rest = data.split(b"\x00", 1)
                        comp = rest[1:]  # skip 1-byte compression method
                        out[k.decode("latin1")] = zlib.decompress(comp).decode("utf-8", "replace")
                    elif ctype == "iTXt":
                        k, rest = data.split(b"\x00", 1)
                        comp_flag = rest[0]
                        rest = rest[2:]  # skip comp_flag + comp_method
                        _lang, rest = rest.split(b"\x00", 1)
                        _trans, rest = rest.split(b"\x00", 1)
                        txt = zlib.decompress(rest) if comp_flag == 1 else rest
                        out[k.decode("latin1")] = txt.decode("utf-8", "replace")
                except Exception:
                    pass
                if ctype == "IEND":
                    break
    except Exception:
        return None
    return out


def get_prompt_graph(path):
    """Return the parsed ComfyUI `prompt` graph dict, or None."""
    chunks = read_png_text_chunks(path)
    if not chunks or "prompt" not in chunks:
        return None
    try:
        return json.loads(chunks["prompt"])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Graph helpers
# --------------------------------------------------------------------------- #
def is_link(v):
    """A ComfyUI input link is [node_id, output_slot]."""
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int))


def node_ct(node):
    return node.get("class_type", "") if isinstance(node, dict) else ""


# Class types that pass conditioning straight through (positive stays positive).
COND_PASSTHROUGH = (
    "FluxGuidance", "ConditioningZeroOut", "ConditioningSetTimestepRange",
    "ConditioningConcat", "ConditioningCombine", "ConditioningSetArea",
    "ConditioningSetAreaPercentage", "ConditioningAverage", "ControlNetApply",
    "ControlNetApplyAdvanced", "ConditioningSetTimestepRangeFromSigmas",
)
# Class types that hold the actual prompt string.
TEXT_ENCODE = ("CLIPTextEncode", "CLIPTextEncodeFlux", "CLIPTextEncodeSDXL",
               "BNK_CLIPTextEncodeAdvanced", "smZ CLIPTextEncode")
# Class types whose output is a plain string (primitive/utility nodes).
STRING_NODES = ("Text Multiline", "Text", "PrimitiveString", "PrimitiveStringMultiline",
                "String", "ttN text", "ShowText|pysssss", "Text Concatenate",
                "JWStringMultiline", "CR Prompt Text")


def resolve_string(graph, val, depth=0):
    """Resolve a value that should be text: literal string, or follow a link to
    a string/utility node and reconstruct its text."""
    if depth > 12:
        return ""
    if isinstance(val, str):
        return val
    if not is_link(val):
        return ""
    node = graph.get(str(val[0]))
    if not isinstance(node, dict):
        return ""
    ins = node.get("inputs", {})
    ct = node_ct(node)
    if "Concatenate" in ct:
        parts = []
        for key in ("text_a", "text_b", "text_c", "text_d"):
            if key in ins:
                parts.append(resolve_string(graph, ins[key], depth + 1))
        delim = ins.get("delimiter", " ")
        delim = delim if isinstance(delim, str) else " "
        joined = delim.join(p for p in parts if p)
        if joined:
            return joined
    # Common string-bearing input names.
    for key in ("text", "string", "value", "String", "Text", "text_multiline", "prompt"):
        if key in ins:
            r = resolve_string(graph, ins[key], depth + 1)
            if r:
                return r
    return ""


def trace_prompt_text(graph, ref, depth=0, visited=None):
    """Follow a conditioning link backwards to the CLIPTextEncode node(s) and
    return the prompt string."""
    if visited is None:
        visited = set()
    if depth > 24 or not is_link(ref):
        return ""
    nid = str(ref[0])
    if nid in visited:
        return ""
    visited.add(nid)
    node = graph.get(nid)
    if not isinstance(node, dict):
        return ""
    ins = node.get("inputs", {})
    ct = node_ct(node)

    if any(ct == t or ct.startswith(t) for t in TEXT_ENCODE):
        # CLIPTextEncodeFlux / SDXL split prompt across t5xxl/clip_l/text_g/text_l.
        for key in ("text", "t5xxl", "t5xl", "clip_l", "text_g", "text_l", "clip_g"):
            if key in ins:
                r = resolve_string(graph, ins[key], depth)
                if r:
                    return r
        return ""

    # Passthrough / unknown conditioning node: follow its conditioning inputs.
    follow_keys = [k for k in ins
                   if any(s in k.lower() for s in ("conditioning", "positive", "cond"))]
    if not follow_keys:
        # As a fallback follow any link input.
        follow_keys = [k for k, v in ins.items() if is_link(v)]
    for k in follow_keys:
        if is_link(ins[k]):
            r = trace_prompt_text(graph, ins[k], depth + 1, visited)
            if r:
                return r
    return ""


# --------------------------------------------------------------------------- #
# Field extractors
# --------------------------------------------------------------------------- #
SAMPLER_CTS = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
               "KSampler (Efficient)", "KSampler //Inspire")
LATENT_EMPTY_HINTS = ("EmptyLatentImage", "Empty Latent", "EmptySD3LatentImage",
                      "SDXL Empty Latent", "EmptyLatent", "ModelSamplingFlux")
MODEL_LOADER_CTS = ("UNETLoader", "UnetLoaderGGUF", "CheckpointLoaderSimple",
                    "CheckpointLoader", "CheckpointLoaderNF4", "UNETLoaderGGUF",
                    "NunchakuFluxDiTLoader", "Load Diffusion Model")


def find_first(graph, cts):
    for nid, node in graph.items():
        if isinstance(node, dict) and node_ct(node) in cts:
            return nid, node
    return None, None


def extract_model(graph):
    """Return (model_file, model_family)."""
    model_file = ""
    for nid, node in graph.items():
        if not isinstance(node, dict):
            continue
        ct = node_ct(node)
        ins = node.get("inputs", {})
        if ct in MODEL_LOADER_CTS:
            for key in ("unet_name", "ckpt_name", "model_path", "model_name", "model"):
                v = ins.get(key)
                if isinstance(v, str) and v and v != "None":
                    model_file = v
                    break
        if model_file:
            break
    fam = detect_family(graph, model_file)
    return model_file, fam


def detect_family(graph, model_file):
    name = (model_file or "").lower()
    keymap = [
        (("z_image", "z-image", "zimage"), "z-image"),
        (("chroma",), "chroma"),
        (("flux",), "flux"),
        (("qwen",), "qwen-image"),
        (("wan",), "wan"),
        (("hunyuan",), "hunyuan"),
        (("sd3", "sd_3", "sd35", "sd3.5"), "sd3"),
        (("pony",), "pony"),
        (("sdxl", "_xl", "-xl", "juggernaut", "illustrious", "noobai"), "sdxl"),
        (("sd15", "sd_15", "v1-5"), "sd1.5"),
    ]
    for keys, fam in keymap:
        if any(k in name for k in keys):
            return fam
    # Fall back to node-type hints when the filename is uninformative.
    cts = {node_ct(n) for n in graph.values() if isinstance(n, dict)}
    if any("Flux" in c for c in cts):
        return "flux"
    if any("Wan" in c for c in cts):
        return "wan"
    if "EmptySD3LatentImage" in cts:
        return "sd3"
    if any("HunyuanVideo" in c for c in cts):
        return "hunyuan"
    if "SDXL Empty Latent Image (rgthree)" in cts or any("SDXL" in c for c in cts):
        return "sdxl"
    return "unknown"


def extract_loras(graph):
    """Return list of 'name@strength' for enabled loras."""
    loras = []
    for nid, node in graph.items():
        if not isinstance(node, dict):
            continue
        ct = node_ct(node)
        ins = node.get("inputs", {})
        if ct == "Power Lora Loader (rgthree)":
            for k, v in ins.items():
                if k.startswith("lora_") and isinstance(v, dict):
                    if v.get("on") and v.get("lora") and v.get("lora") != "None":
                        loras.append("%s@%s" % (v["lora"], v.get("strength", 1)))
        elif ct == "Lora Loader Stack (rgthree)":
            i = 1
            while ("lora_%d" % i) in ins:
                name = ins.get("lora_%d" % i)
                if isinstance(name, str) and name and name != "None":
                    st = ins.get("strength_%d" % i, 1)
                    loras.append("%s@%s" % (name, st))
                i += 1
        elif ct in ("LoraLoader", "LoraLoaderModelOnly", "LoraLoaderTagsQuery"):
            name = ins.get("lora_name")
            if isinstance(name, str) and name and name != "None":
                st = ins.get("strength_model", ins.get("strength", 1))
                loras.append("%s@%s" % (name, st))
    return loras


def extract_prompts(graph):
    """Return (positive, negative)."""
    # Prefer tracing from a sampler's conditioning so we get the real positive.
    pos = neg = ""
    sid, snode = find_first(graph, SAMPLER_CTS)
    if snode:
        ins = snode.get("inputs", {})
        if "positive" in ins and is_link(ins["positive"]):
            pos = trace_prompt_text(graph, ins["positive"])
        if "negative" in ins and is_link(ins["negative"]):
            neg = trace_prompt_text(graph, ins["negative"])
        if not pos and "guider" in ins and is_link(ins["guider"]):
            g = graph.get(str(ins["guider"][0]), {})
            gi = g.get("inputs", {}) if isinstance(g, dict) else {}
            for key in ("conditioning", "positive", "cond"):
                if key in gi and is_link(gi[key]):
                    pos = trace_prompt_text(graph, gi[key])
                    if pos:
                        break
            if "negative" in gi and is_link(gi["negative"]):
                neg = trace_prompt_text(graph, gi["negative"])
    # Fallback: collect CLIPTextEncode texts directly; longest = positive.
    if not pos:
        texts = []
        for nid, node in graph.items():
            if isinstance(node, dict) and any(node_ct(node).startswith(t) for t in TEXT_ENCODE):
                t = resolve_string(graph, node.get("inputs", {}).get("text", ""))
                if t:
                    texts.append(t)
        if texts:
            texts.sort(key=len, reverse=True)
            pos = texts[0]
            if not neg and len(texts) > 1:
                neg = texts[-1]
    return pos.strip(), neg.strip()


VIDEO_CTS_I2V = ("WanImageToVideo", "SVD_img2vid_Conditioning", "CogVideoImageEncode",
                 "HunyuanVideoImageToVideo")
VIDEO_CTS_ANY = ("WanImageToVideo", "WanVideoSampler", "VHS_VideoCombine",
                 "SVD_img2vid_Conditioning", "CogVideoSampler", "CogVideoDecode",
                 "HunyuanVideoSampler", "LTXVideo", "EmptyHunyuanLatentVideo",
                 "WanVideoModelLoader", "SaveAnimatedWEBP", "SaveAnimatedPNG")


def pixels_to_origin(graph, ref, depth=0, visited=None):
    """Trace a pixel/image link back: 'load' (user LoadImage) or 'decode'
    (internally generated, i.e. an upscale/refine pass)."""
    if visited is None:
        visited = set()
    if depth > 24 or not is_link(ref):
        return ""
    nid = str(ref[0])
    if nid in visited:
        return ""
    visited.add(nid)
    node = graph.get(nid)
    if not isinstance(node, dict):
        return ""
    ct = node_ct(node)
    if ct.startswith("LoadImage") or ct in ("LoadImageMask", "Image Load", "LoadImageOutput"):
        return "load"
    if ct == "VAEDecode" or ct.startswith("VAEDecode"):
        return "decode"
    ins = node.get("inputs", {})
    # prefer image-ish inputs first, then any link
    ordered = [k for k in ("image", "images", "pixels", "IMAGE") if k in ins]
    ordered += [k for k in ins if k not in ordered]
    for k in ordered:
        if is_link(ins.get(k)):
            r = pixels_to_origin(graph, ins[k], depth + 1, visited)
            if r:
                return r
    return ""


def trace_latent_source(graph, ref, depth=0, visited=None):
    """Follow a latent link back; return 'empty', 'i2i' (VAEEncode of a loaded
    image), 'refine' (VAEEncode of an internally generated image), or ''."""
    if visited is None:
        visited = set()
    if depth > 24 or not is_link(ref):
        return ""
    nid = str(ref[0])
    if nid in visited:
        return ""
    visited.add(nid)
    node = graph.get(nid)
    if not isinstance(node, dict):
        return ""
    ct = node_ct(node)
    ins = node.get("inputs", {})
    if ct in ("VAEEncode", "VAEEncodeForInpaint", "VAEEncodeTiled"):
        origin = pixels_to_origin(graph, ins.get("pixels") or ins.get("image"))
        return "i2i" if origin == "load" else "refine"
    if any(h in ct for h in ("EmptyLatent", "Empty Latent", "EmptySD3", "SDXL Empty Latent")):
        return "empty"
    for k in ("latent_image", "latent", "samples", "LATENT"):
        if k in ins and is_link(ins[k]):
            r = trace_latent_source(graph, ins[k], depth + 1, visited)
            if r:
                return r
    for k, v in ins.items():
        if is_link(v):
            r = trace_latent_source(graph, v, depth + 1, visited)
            if r:
                return r
    return ""


def find_back(graph, ref, target_cts, depth=0, visited=None):
    """BFS-ish backward search following link inputs; return first node whose
    class_type is in target_cts."""
    if visited is None:
        visited = set()
    if depth > 30 or not is_link(ref):
        return None, None
    nid = str(ref[0])
    if nid in visited:
        return None, None
    visited.add(nid)
    node = graph.get(nid)
    if not isinstance(node, dict):
        return None, None
    if node_ct(node) in target_cts:
        return nid, node
    for k, v in node.get("inputs", {}).items():
        if is_link(v):
            r_id, r_node = find_back(graph, v, target_cts, depth + 1, visited)
            if r_node:
                return r_id, r_node
    return None, None


def find_output_sampler(graph):
    """Find the sampler that feeds the saved image (trace back from SaveImage)."""
    for nid, node in graph.items():
        if isinstance(node, dict) and node_ct(node) in ("SaveImage", "SaveImageWebsocket"):
            imgs = node.get("inputs", {}).get("images")
            if is_link(imgs):
                sid, snode = find_back(graph, imgs, set(SAMPLER_CTS))
                if snode:
                    return sid, snode
    return find_first(graph, SAMPLER_CTS)


def extract_gen_type(graph):
    cts = {node_ct(n) for n in graph.values() if isinstance(n, dict)}
    is_video = any(c in cts for c in VIDEO_CTS_ANY)
    if is_video:
        if any(c in cts for c in VIDEO_CTS_I2V):
            return "I2V"
        if "LoadImage" in cts:  # image-driven video
            return "I2V"
        return "T2V"
    # image: inspect the OUTPUT sampler's latent source
    sid, snode = find_output_sampler(graph)
    if snode:
        ins = snode.get("inputs", {})
        ref = ins.get("latent_image") or ins.get("latent")
        if is_link(ref) and trace_latent_source(graph, ref) == "i2i":
            return "i2i"
        return "t2i"
    # no sampler found: only call it i2i if a loaded image is VAE-encoded
    for nid, node in graph.items():
        if isinstance(node, dict) and node_ct(node) in ("VAEEncode", "VAEEncodeForInpaint"):
            ins = node.get("inputs", {})
            if pixels_to_origin(graph, ins.get("pixels") or ins.get("image")) == "load":
                return "i2i"
    return "t2i"


def extract_dims_and_sampling(graph):
    """Best-effort width/height/steps/sampler/scheduler/guidance/seed/denoise."""
    out = dict(width="", height="", steps="", sampler="", scheduler="",
               guidance="", seed="", denoise="")
    for nid, node in graph.items():
        if not isinstance(node, dict):
            continue
        ct = node_ct(node)
        ins = node.get("inputs", {})
        if ct in ("BasicScheduler", "KSampler", "KSamplerAdvanced"):
            if isinstance(ins.get("steps"), (int, float)):
                out["steps"] = ins["steps"]
            if isinstance(ins.get("scheduler"), str):
                out["scheduler"] = ins["scheduler"]
            if isinstance(ins.get("denoise"), (int, float)):
                out["denoise"] = ins["denoise"]
            if isinstance(ins.get("sampler_name"), str):
                out["sampler"] = ins["sampler_name"]
        if ct == "KSamplerSelect" and isinstance(ins.get("sampler_name"), str):
            out["sampler"] = ins["sampler_name"]
        if ct == "FluxGuidance" and isinstance(ins.get("guidance"), (int, float)):
            out["guidance"] = ins["guidance"]
        if ct in ("RandomNoise", "KSampler", "KSamplerAdvanced"):
            v = ins.get("noise_seed", ins.get("seed"))
            if isinstance(v, (int, float)):
                out["seed"] = v
        if isinstance(ins.get("width"), (int, float)) and not out["width"]:
            out["width"] = ins["width"]
            out["height"] = ins.get("height", "")
        # rgthree SDXL Empty Latent packs "  896 x 1152  (portrait)"
        if ct == "SDXL Empty Latent Image (rgthree)" and not out["width"]:
            dim = ins.get("dimensions", "")
            if isinstance(dim, str) and "x" in dim:
                try:
                    w, h = dim.split("(")[0].split("x")
                    out["width"] = int(w.strip())
                    out["height"] = int(h.strip())
                except Exception:
                    pass
    return out


# --------------------------------------------------------------------------- #
# SFW / NSFW classification
# --------------------------------------------------------------------------- #
# Whole-word terms (matched with word boundaries to avoid false hits like
# "class"/"grass"). Nudity + explicit sexual content only -- glamour terms such
# as "sexy", "cleavage", "lingerie", "bikini" are intentionally NOT included.
_NSFW_WORDS = [
    "nude", "nudes", "naked", "nudity", "topless", "bottomless",
    "breast", "breasts", "boob", "boobs", "tit", "tits", "titties",
    "nipple", "nipples", "areola", "areolae",
    "pussy", "vagina", "vulva", "labia", "clitoris", "clit", "cunt", "twat",
    "penis", "cock", "erection", "phallus",
    "cum", "cumming", "semen", "ejaculate", "ejaculation", "precum",
    "sex", "intercourse", "fuck", "fucking", "fucked", "fellatio",
    "blowjob", "handjob", "rimjob", "footjob", "titjob",
    "deepthroat", "bukkake", "gangbang", "threesome", "orgy",
    "anal", "anus", "asshole",
    "orgasm", "masturbate", "masturbating", "masturbation", "fingering",
    "doggystyle", "missionary", "cunnilingus",
    "erotic", "porn", "porno", "pornographic", "hardcore", "xxx", "nsfw",
    "slut", "whore", "milf", "upskirt", "genital", "genitals", "genitalia",
    "bdsm", "bondage", "dildo", "vibrator", "strapon", "buttplug",
    "cumshot", "creampie", "squirting",
    "pubic", "crotch", "buttocks",
    # Nudity phrases. Body-part-specific so clothed "bare shoulders / bare legs /
    # wearing only overalls" stay SFW.
    "no clothes", "without clothes", "wearing nothing", "wearing no clothes",
    "bare chest", "bare breast", "bare breasts", "bare ass", "bare butt",
    "bare bottom", "bare nipple", "bare nipples", "bare pussy",
    "exposed body", "exposed breast", "exposed breasts", "exposed chest",
    "exposed nipple", "exposed nipples",
    "fully exposing", "exposing her body", "exposing her breasts",
    "exposing her chest",
    "spread legs", "spread her legs", "spreading her legs", "legs spread apart",
    "panty reveal", "transparent",
]
# Substrings (compound words / lora filenames) matched anywhere.
_NSFW_SUBSTR = [
    "blowjob", "deepthroat", "handjob", "footjob", "titjob", "titfuck",
    "cumshot", "creampie", "masturbat", "gangbang", "doggystyle",
    "pussydiffusion", "nipplediffusion", "sexposition", "ultimatedeepthroat",
    "bareback", "nude-mod", "nude_mod", "nudemod",
]
_NSFW_WORD_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _NSFW_WORDS) + r")\b", re.I)
_NSFW_SUBSTR_RE = re.compile("|".join(re.escape(s) for s in _NSFW_SUBSTR), re.I)


def classify_image_type(positive_prompt, loras=""):
    """Return 'NSFW' if the prompt or enabled lora names indicate nudity/sexual
    content, else 'SFW'. Negative prompt is deliberately ignored."""
    text = "%s\n%s" % (positive_prompt or "", loras or "")
    if _NSFW_WORD_RE.search(text) or _NSFW_SUBSTR_RE.search(text):
        return "NSFW"
    return "SFW"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
FIELDS = ["file_path", "folder", "filename", "model_family", "model_file",
          "gen_type", "image_type", "positive_prompt", "negative_prompt",
          "loras", "lora_count", "width", "height", "steps", "sampler",
          "scheduler", "guidance", "seed", "denoise"]

# Fields whose combination defines one unique "generation" (batch/sequential
# repeats share these; filename / path / seed / dims differ and are ignored).
DEDUP_KEY_FIELDS = ["model_family", "model_file", "gen_type",
                    "positive_prompt", "negative_prompt", "loras"]

# Columns in the deduplicated output. Representative (first-seen) values are
# kept for the sampling fields. `loras` keeps name@strength so the lora name and
# its strength stay tied together.
UNIQUE_FIELDS = ["model_family", "model_file", "gen_type", "image_type",
                 "positive_prompt", "negative_prompt", "loras", "lora_count",
                 "width", "height", "steps", "sampler", "scheduler", "seed"]


def dedup_key(row):
    return tuple(row.get(k, "") for k in DEDUP_KEY_FIELDS)


def write_unique_csv(rows, out_path):
    """Collapse rows by generation params; write one row per unique key with an
    image_count. `rows` may be a list or any iterable of dict rows. Returns
    (unique_count, collapsed_total)."""
    groups = {}          # key -> aggregate row
    order = []           # preserve first-seen order
    collapsed = 0
    for row in rows:
        if row.get("gen_type") == "no-metadata":
            continue     # nothing meaningful to dedup on
        collapsed += 1
        k = dedup_key(row)
        agg = groups.get(k)
        if agg is None:
            agg = {f: "" for f in UNIQUE_FIELDS}
            for f in DEDUP_KEY_FIELDS:
                agg[f] = row.get(f, "")
            for f in ("lora_count", "width", "height", "steps",
                      "sampler", "scheduler", "seed"):
                agg[f] = row.get(f, "")
            # image_type is derived from the prompt/loras; carry it if present,
            # otherwise compute (lets --dedup-only run on an older CSV).
            agg["image_type"] = row.get("image_type") or classify_image_type(
                row.get("positive_prompt", ""), row.get("loras", ""))
            groups[k] = agg
            order.append(k)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=UNIQUE_FIELDS)
        writer.writeheader()
        for k in order:
            writer.writerow(groups[k])
    return len(order), collapsed


def unique_path_for(output):
    base, ext = os.path.splitext(output)
    return base + "_unique" + (ext or ".csv")


def process_file(path, root):
    graph = get_prompt_graph(path)
    rel_folder = os.path.relpath(os.path.dirname(path), root)
    row = {f: "" for f in FIELDS}
    row["file_path"] = path
    row["folder"] = rel_folder if rel_folder != "." else ""
    row["filename"] = os.path.basename(path)
    if not graph:
        row["gen_type"] = "no-metadata"
        return row, False
    model_file, fam = extract_model(graph)
    pos, neg = extract_prompts(graph)
    loras = extract_loras(graph)
    row["model_file"] = model_file
    row["model_family"] = fam
    row["gen_type"] = extract_gen_type(graph)
    row["positive_prompt"] = pos
    row["negative_prompt"] = neg
    row["loras"] = "; ".join(loras)
    row["lora_count"] = len(loras)
    row["image_type"] = classify_image_type(pos, row["loras"])
    row.update(extract_dims_and_sampling(graph))
    # Keep the raw ComfyUI graph for downstream replay (DB import / regeneration).
    # Not a CSV column -- the underscore key is ignored by the CSV writers.
    row["_graph"] = graph
    return row, True


def iter_pngs(root):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".png"):
                yield os.path.join(dirpath, fn)


def main():
    ap = argparse.ArgumentParser(description="Extract ComfyUI PNG generation metadata to CSV.")
    ap.add_argument("-i", "--input", default="E:/DATA/renders",
                    help="Root directory to scan recursively for *.png (default: E:/DATA/renders)")
    ap.add_argument("-o", "--output", default="render_metadata.csv",
                    help="Output CSV path (default: render_metadata.csv)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N files (0 = all). Useful for testing.")
    ap.add_argument("--jsonl", default=None,
                    help="Also write a JSON Lines file: one record per PNG with all "
                         "fields PLUS the raw ComfyUI workflow graph under 'workflow'. "
                         "This is the import source for the prompt_browser database.")
    ap.add_argument("--no-dedup", action="store_true",
                    help="Skip writing the deduplicated *_unique.csv.")
    ap.add_argument("--dedup-only", action="store_true",
                    help="Skip scanning; just (re)build the *_unique.csv from an "
                         "existing --output CSV.")
    ap.add_argument("--reclassify", action="store_true",
                    help="Recompute image_type (SFW/NSFW) for an existing --output "
                         "CSV in place and rebuild the *_unique.csv, without "
                         "rescanning PNGs. Use after editing the keyword lists.")
    args = ap.parse_args()

    uniq_out = unique_path_for(args.output)

    # Reclassify mode: recompute image_type from existing prompt/loras columns.
    if args.reclassify:
        if not os.path.isfile(args.output):
            print("CSV not found: %s" % args.output)
            return
        with open(args.output, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        changed = 0
        for row in rows:
            if row.get("gen_type") == "no-metadata":
                continue
            new = classify_image_type(row.get("positive_prompt", ""), row.get("loras", ""))
            if row.get("image_type") != new:
                changed += 1
            row["image_type"] = new
        with open(args.output, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        n_unique, _ = write_unique_csv(rows, uniq_out)
        print("Reclassified %d rows (%d changed) -> %s" % (len(rows), changed, args.output))
        print("Rebuilt %d unique generations -> %s" % (n_unique, uniq_out))
        return

    # Dedup-only mode: read the existing CSV and rebuild the unique file.
    if args.dedup_only:
        if not os.path.isfile(args.output):
            print("CSV not found: %s" % args.output)
            return
        with open(args.output, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        n_unique, collapsed = write_unique_csv(rows, uniq_out)
        print("Deduplicated %d rows -> %d unique generations -> %s"
              % (collapsed, n_unique, uniq_out))
        return

    if not os.path.isdir(args.input):
        print("Input directory not found: %s" % args.input)
        return

    total = with_meta = without = 0
    collected = []  # kept for the dedup phase (graph stripped to bound memory)
    jsonl_fh = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    try:
        with open(args.output, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for path in iter_pngs(args.input):
                row, ok = process_file(path, args.input)
                writer.writerow(row)  # extrasaction="ignore" drops _graph
                if jsonl_fh is not None:
                    record = {f: row.get(f, "") for f in FIELDS}
                    record["workflow"] = row.get("_graph")
                    jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                if not args.no_dedup:
                    collected.append({f: row.get(f, "") for f in FIELDS})  # no graph
                total += 1
                with_meta += 1 if ok else 0
                without += 0 if ok else 1
                if total % 500 == 0:
                    print("  processed %d files (%d with metadata)..." % (total, with_meta))
                if args.limit and total >= args.limit:
                    break
    finally:
        if jsonl_fh is not None:
            jsonl_fh.close()

    print("\nDone. %d PNGs scanned -> %s" % (total, args.output))
    print("  with generation metadata: %d" % with_meta)
    print("  without metadata:         %d" % without)
    if args.jsonl:
        print("  workflow graphs -> %s" % args.jsonl)

    if not args.no_dedup:
        n_unique, collapsed = write_unique_csv(collected, uniq_out)
        print("\nDedup phase: %d rows -> %d unique generations -> %s"
              % (collapsed, n_unique, uniq_out))


if __name__ == "__main__":
    main()
