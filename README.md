# ai-tools
Python scripts I am building to help with my ai addiction

## Repository structure

| Folder | Contents |
| --- | --- |
| `train_scripts/` | Parameter-driven LoRA cache/train drivers for musubi-tuner (krea2, Klein, Z-Image, WAN 2.2, LTX-2.3). One generic driver per architecture — pass everything as flags. See `train_scripts/README.md`. |
| `lora_utilities/` | Python utilities for inspecting LoRAs, captioning datasets, and extracting metadata from generated images (see below). |
| `prompt_browser/` | PostgreSQL-backed app to browse extracted prompts and re-submit them to the ComfyUI API to generate. See `prompt_browser/README.md`. |

Trained LoRA `.safetensors` weights and sample renders are no longer tracked in
this repository (they live on local/external storage only). This repo now holds
the **tooling** — training drivers, dataset/caption utilities, and the prompt
browser.

## lora_utilities

Utilities for working with LoRA files, LoRA training datasets, and AI-generated
image metadata.

### extract_render_metadata.py

Walks a directory tree of **ComfyUI-generated PNGs**, reads the generation graph
embedded in each PNG (the `prompt` text chunk) and exports the metadata to CSV
for import into a database or spreadsheet. Pure standard library — no extra
dependencies, runs in any Python 3.8+ environment.

**Basic usage**

```bash
python extract_render_metadata.py -i "E:/DATA/renders" -o render_metadata.csv
```

This produces two files:

| File | Contents |
| --- | --- |
| `render_metadata.csv` | One row per PNG. |
| `render_metadata_unique.csv` | Deduplicated — one row per unique *generation* (same model + type + prompt + loras collapse into a single row, ignoring filename/seed/dimensions). |

**Options**

| Flag | Description |
| --- | --- |
| `-i`, `--input` | Root directory scanned recursively for `*.png` (default `E:/DATA/renders`). |
| `-o`, `--output` | Output CSV path (default `render_metadata.csv`). The unique file is named automatically, e.g. `render_metadata_unique.csv`. |
| `--limit N` | Process at most N files (0 = all). Useful for a quick test run. |
| `--no-dedup` | Skip writing the deduplicated `*_unique.csv`. |
| `--dedup-only` | Skip scanning; rebuild the `*_unique.csv` from an existing `--output` CSV. |
| `--reclassify` | Recompute the SFW/NSFW `image_type` for an existing `--output` CSV in place and rebuild the unique CSV, **without rescanning PNGs**. Use after editing the keyword lists. |

**Columns**

Per-file CSV: `file_path, folder, filename, model_family, model_file, gen_type,
image_type, positive_prompt, negative_prompt, loras, lora_count, width, height,
steps, sampler, scheduler, guidance, seed, denoise`

Unique CSV: `model_family, model_file, gen_type, image_type, positive_prompt,
negative_prompt, loras, lora_count, width, height, steps, sampler, scheduler,
seed`

Notes on key columns:

- **`model_family`** — derived from the model filename / node types: `flux`,
  `chroma`, `sdxl`, `sd3`, `wan`, `z-image`, `qwen-image`, `hunyuan`, etc.
- **`gen_type`** — `t2i`, `i2i`, `I2V`, or `T2V`. A `t2i` that re-encodes its own
  output for an upscale/hires pass is correctly kept as `t2i`; `i2i` is only used
  when a user `LoadImage` feeds the latent.
- **`image_type`** — `SFW` or `NSFW`, classified by scanning the positive prompt
  and the enabled lora names for nudity / sexual terms (the negative prompt is
  deliberately ignored).
- **`loras`** — enabled loras only, as `name@strength`, joined by `; `. Disabled
  loras in a Power Lora Loader are skipped.

**Tuning the SFW/NSFW classifier**

The keyword lists `_NSFW_WORDS` and `_NSFW_SUBSTR` live near the top of the
classifier section in the script. After editing them you do **not** need to
rescan the PNGs — just recompute from the existing CSV:

```bash
python extract_render_metadata.py -o render_metadata.csv --reclassify
```

### joycaption_dir.py

Batch-captions every image in a directory with **JoyCaption Beta One** and
writes a `.txt` sidecar next to each image (`ImageName.png` → `ImageName.txt`),
with a trigger word prepended — ready for LoRA training. Prompt presets:
`descriptive`, `short`, `training`, `tags`.

```bash
python joycaption_dir.py ./dataset underbust_corset
python joycaption_dir.py ./dataset underbust_corset --style short --overwrite
```

Requires `torch`, `transformers`, `accelerate`, and `pillow`. The model (~17 GB)
auto-downloads from HuggingFace on first run.

### get_lora_info.py

Interactive inspector for `.safetensors` LoRA files. Lists the files in a
directory, lets you pick which to inspect, and prints (or exports) each file's
metadata and tensor keys. Requires `safetensors`.

```bash
python get_lora_info.py -d "path/to/lora/directory"
```

Then follow the prompts to choose files by number and to output to screen or to
a `json`/`txt` file.
