# prompt_browser

Browse the prompts/metadata extracted from your renders (stored in PostgreSQL),
pick one, optionally tweak the seed or prompt, and re-submit the original
ComfyUI workflow to generate a new image.

```
extract_render_metadata.py --jsonl   ->   build_db.py   ->   PostgreSQL
                                                                  |
                                                          app.py (Streamlit)
                                                                  |
                                                       ComfyUI  /prompt  API
```

The key idea: each render's **raw ComfyUI workflow graph** is stored alongside
its metadata, so "generate" just replays that exact graph (optionally with a new
seed/prompt) — no per-workflow templates to maintain, and any past image is
reproducible.

## Prerequisites

- **PostgreSQL** running. Native install, or quick container:
  ```bash
  docker run -d --name prompts-pg -p 5432:5432 \
    -e POSTGRES_USER=prompts -e POSTGRES_PASSWORD=prompts -e POSTGRES_DB=prompts \
    postgres:16
  ```
- **ComfyUI** running with its API (default `http://127.0.0.1:8188`).
- The **`render` conda environment** (this project's env):
  ```bash
  conda activate render
  pip install -r requirements.txt
  ```

## Setup

> All commands below assume `conda activate render` is active.

1. Copy config and edit:
   ```bash
   cp .env.example .env       # set DATABASE_URL and COMFYUI_URL
   ```
2. Extract metadata **with workflow graphs** (note `--jsonl`):
   ```bash
   python ../lora_utilities/extract_render_metadata.py \
       -i "E:/DATA/renders" -o renders.csv --jsonl renders.jsonl --no-dedup
   ```
3. Load into PostgreSQL (creates the schema on first run, upserts on re-runs):
   ```bash
   python build_db.py renders.jsonl
   ```
4. Launch the UI:
   ```bash
   streamlit run app.py
   ```

To import another batch later (e.g. the next 100k images), repeat steps 2–3 with
the new render directory; rows are upserted by `file_path`, so nothing duplicates.

## Files

| File | Purpose |
| --- | --- |
| `schema.sql` | PostgreSQL schema (generations + loras + generation_loras, trigram prompt index). |
| `db.py` | Connection helper (reads `DATABASE_URL`). |
| `build_db.py` | Loads the extractor JSONL into PostgreSQL. |
| `comfy_client.py` | Minimal ComfyUI API client (queue / progress / fetch images). |
| `graph_patch.py` | Sets/randomizes seed and best-effort overrides the positive prompt in a stored graph. |
| `app.py` | Streamlit browser + generate UI. |

## Notes & limits

- **Assets must exist in ComfyUI.** A replayed graph references model/lora
  *filenames*; those must be installed where ComfyUI looks (`models/loras`, etc.).
  Generation fails with a validation error otherwise.
- **Prompt editing is best-effort.** Workflows that build the prompt from a
  `Text Concatenate` chain (e.g. the multiperson graphs) can't be safely edited;
  the app warns and replays the original prompt. Seed override always works.
- Single-user/local by design — no auth, no queue. ComfyUI serializes jobs.
