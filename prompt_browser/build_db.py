"""
build_db.py -- load the extractor's JSONL into PostgreSQL.

1. Generate the JSONL with the workflow graphs (PowerShell, one line):
     python ../lora_utilities/extract_render_metadata.py -i "E:/DATA/renders" -o renders.csv --jsonl renders.jsonl --no-dedup
2. Load it:
     python build_db.py renders.jsonl

Re-running is safe: rows are upserted by file_path, so you can append new
batches (e.g. the next 100k images) without duplicating.
"""
import os
import sys
import json
import argparse

import psycopg2
from psycopg2.extras import execute_batch

from db import connect

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def _int(v):
    try:
        return int(float(v)) if v not in ("", None) else None
    except (ValueError, TypeError):
        return None


def _real(v):
    try:
        return float(v) if v not in ("", None) else None
    except (ValueError, TypeError):
        return None


def _seed(v):
    # seeds are big integers; keep as string for NUMERIC, None if blank/non-numeric
    if v in ("", None):
        return None
    s = str(v).strip()
    return s if s.lstrip("-").isdigit() else None


def parse_loras(loras_str):
    """'a.safetensors@1; b\\c.safetensors@0.8' -> [('a.safetensors', 1.0), ...]"""
    out = []
    for chunk in (loras_str or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, strength = chunk.rpartition("@")
        if not name:                      # no '@' -> whole chunk is the name
            name, strength = chunk, ""
        out.append((name.strip(), _real(strength)))
    return out


def ensure_schema(cur):
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        cur.execute(f.read())


UPSERT_GEN = """
INSERT INTO generations
    (file_path, folder, filename, model_family, model_file, gen_type, image_type,
     positive_prompt, negative_prompt, width, height, steps, sampler, scheduler,
     guidance, seed, denoise, lora_count, workflow)
VALUES (%(file_path)s, %(folder)s, %(filename)s, %(model_family)s, %(model_file)s,
        %(gen_type)s, %(image_type)s, %(positive_prompt)s, %(negative_prompt)s,
        %(width)s, %(height)s, %(steps)s, %(sampler)s, %(scheduler)s, %(guidance)s,
        %(seed)s, %(denoise)s, %(lora_count)s, %(workflow)s)
ON CONFLICT (file_path) DO UPDATE SET
    folder=EXCLUDED.folder, filename=EXCLUDED.filename,
    model_family=EXCLUDED.model_family, model_file=EXCLUDED.model_file,
    gen_type=EXCLUDED.gen_type, image_type=EXCLUDED.image_type,
    positive_prompt=EXCLUDED.positive_prompt, negative_prompt=EXCLUDED.negative_prompt,
    width=EXCLUDED.width, height=EXCLUDED.height, steps=EXCLUDED.steps,
    sampler=EXCLUDED.sampler, scheduler=EXCLUDED.scheduler, guidance=EXCLUDED.guidance,
    seed=EXCLUDED.seed, denoise=EXCLUDED.denoise, lora_count=EXCLUDED.lora_count,
    workflow=EXCLUDED.workflow
RETURNING id;
"""


def main():
    ap = argparse.ArgumentParser(description="Load extractor JSONL into PostgreSQL.")
    ap.add_argument("jsonl", help="Path to the JSONL produced by extract_render_metadata.py --jsonl")
    ap.add_argument("--batch", type=int, default=500, help="Commit every N records.")
    args = ap.parse_args()

    if not os.path.isfile(args.jsonl):
        raise SystemExit("JSONL not found: %s" % args.jsonl)

    conn = connect()
    conn.autocommit = False
    cur = conn.cursor()
    ensure_schema(cur)
    conn.commit()

    lora_cache = {}  # name -> id

    def get_lora_id(name):
        if name in lora_cache:
            return lora_cache[name]
        cur.execute(
            "INSERT INTO loras (name) VALUES (%s) ON CONFLICT (name) DO UPDATE "
            "SET name=EXCLUDED.name RETURNING id", (name,))
        lid = cur.fetchone()[0]
        lora_cache[name] = lid
        return lid

    total = skipped = 0
    with open(args.jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("gen_type") == "no-metadata" or not rec.get("workflow"):
                skipped += 1
                continue
            params = {
                "file_path": rec.get("file_path"),
                "folder": rec.get("folder"),
                "filename": rec.get("filename"),
                "model_family": rec.get("model_family"),
                "model_file": rec.get("model_file"),
                "gen_type": rec.get("gen_type"),
                "image_type": rec.get("image_type"),
                "positive_prompt": rec.get("positive_prompt"),
                "negative_prompt": rec.get("negative_prompt"),
                "width": _int(rec.get("width")),
                "height": _int(rec.get("height")),
                "steps": _int(rec.get("steps")),
                "sampler": rec.get("sampler") or None,
                "scheduler": rec.get("scheduler") or None,
                "guidance": _real(rec.get("guidance")),
                "seed": _seed(rec.get("seed")),
                "denoise": _real(rec.get("denoise")),
                "lora_count": _int(rec.get("lora_count")),
                "workflow": json.dumps(rec.get("workflow")),
            }
            cur.execute(UPSERT_GEN, params)
            gen_id = cur.fetchone()[0]
            # resync loras for this generation
            cur.execute("DELETE FROM generation_loras WHERE generation_id=%s", (gen_id,))
            links = [(gen_id, get_lora_id(n), s) for n, s in parse_loras(rec.get("loras"))]
            if links:
                execute_batch(
                    cur,
                    "INSERT INTO generation_loras (generation_id, lora_id, strength) "
                    "VALUES (%s, %s, %s)", links)
            total += 1
            if total % args.batch == 0:
                conn.commit()
                print("  loaded %d records..." % total)

    conn.commit()
    cur.close()
    conn.close()
    print("\nDone. %d generations loaded/updated, %d skipped (no workflow)." % (total, skipped))


if __name__ == "__main__":
    main()
