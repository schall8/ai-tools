#!/usr/bin/env python
"""
Stamp LoRA trigger word(s) into trained .safetensors metadata (post-training).

Model-agnostic: works on any musubi/sd-scripts LoRA. Writes the fields the
common ComfyUI trigger tools read, without touching tensor weights:

  ss_tag_frequency        -> rgthree Power Lora Loader "Show Info" trigger list
  ss_training_comment     -> shown by several loaders/managers
  modelspec.trigger_phrase-> ModelSpec-standard trigger field

Multiple triggers may be given comma-separated: --trigger "c0urtney, corset".

Usage:
  # stamp every top-level *.safetensors in an output dir (skips *.bak):
  python write_trigger.py --dir D:/DATA/training/krea2_loras/courtney_krea2 --trigger c0urtney
  # or a single file:
  python write_trigger.py --file path/to/lora.safetensors --trigger c0urtney

Windows note: load_file() memory-maps the file, so we clone tensors into RAM
and release the map before overwriting (otherwise Windows errors 1224).
"""

import argparse
import gc
import json
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
except Exception as e:  # pragma: no cover
    sys.exit(f"ERROR: safetensors/torch not importable ({e}). "
             f"Run inside the musubi conda env.")

TRIGGER_COUNT = 999999  # high enough to sort above real training tags


def stamp(path: Path, triggers: list[str]) -> None:
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})

    # clone into RAM so the mmap can be released before we overwrite (Windows)
    mm = load_file(str(path))
    tensors = {k: v.clone() for k, v in mm.items()}
    del mm
    gc.collect()

    # merge into ss_tag_frequency (rgthree reads this)
    try:
        tf = json.loads(meta["ss_tag_frequency"]) if meta.get("ss_tag_frequency") else {}
        if not isinstance(tf, dict):
            tf = {}
    except Exception:
        tf = {}
    bucket = tf.setdefault("trigger", {})
    if not isinstance(bucket, dict):
        bucket = tf["trigger"] = {}
    for t in triggers:
        bucket[t] = TRIGGER_COUNT
    meta["ss_tag_frequency"] = json.dumps(tf)

    joined = ", ".join(triggers)
    meta["ss_training_comment"] = joined
    meta["modelspec.trigger_phrase"] = joined

    save_file(tensors, str(path), metadata=meta)
    print(f"  stamped {path.name}: {joined}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", help="Stamp every top-level *.safetensors here.")
    g.add_argument("--file", help="Stamp a single .safetensors file.")
    p.add_argument("--trigger", required=True,
                   help='Trigger word, or comma-separated list.')
    args = p.parse_args()

    triggers = [t.strip() for t in args.trigger.split(",") if t.strip()]
    if not triggers:
        sys.exit("ERROR: --trigger resolved to no words.")

    if args.file:
        files = [Path(args.file)]
    else:
        d = Path(args.dir)
        if not d.is_dir():
            sys.exit(f"ERROR: not a directory: {d}")
        files = sorted(f for f in d.glob("*.safetensors") if not f.name.endswith(".bak"))

    if not files:
        print("write_trigger: no .safetensors files found — nothing to stamp.")
        return 0

    for f in files:
        if not f.is_file():
            print(f"  skip (missing): {f}")
            continue
        try:
            stamp(f, triggers)
        except Exception as e:
            print(f"  WARN: failed to stamp {f.name}: {e}")

    print(f"write_trigger: done ({len(files)} file(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
