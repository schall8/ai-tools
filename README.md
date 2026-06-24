# ai-tools
Python scripts I am building to help with my ai addiction

## Repository structure

| Folder | Contents |
| --- | --- |
| `loras/` | Trained LoRA `.safetensors` weights, organized into a subfolder per base model. |
| `lora_utilities/` | Python utilities for inspecting LoRAs and extracting metadata from generated images (see below), plus the exported CSVs. |
| `OneTrainer/trainer_presets/` | [OneTrainer](https://github.com/Nerogar/OneTrainer) LoRA training config presets (`.json`). |
| `fluxgym_mods/` | Modifications for [fluxgym](https://github.com/cocktailpeanut/fluxgym) (e.g. `models.yaml`). |
| `renders/` | Sample ComfyUI-generated PNGs (with embedded generation metadata). |

### loras/

LoRA weights grouped by the base model they were trained for:

| Subfolder | Base model |
| --- | --- |
| `loras/Flux1-dev/` | FLUX.1-dev |
| `loras/Flux2-Klein/` | FLUX.2 Klein |
| `loras/Chroma/` | Chroma |
| `loras/SDXL/` | SDXL |
| `loras/Wan22/` | Wan 2.2 |
| `loras/z-image/` | Z-Image |
| `loras/LTX2.3/` | LTX-Video 2.3 |

## lora_utilities

Utilities for working with LoRA files and AI-generated image metadata.

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

### get_lora_info.py

Interactive inspector for `.safetensors` LoRA files. Lists the files in a
directory, lets you pick which to inspect, and prints (or exports) each file's
metadata and tensor keys. Requires `safetensors`.

```bash
python get_lora_info.py -d "path/to/lora/directory"
```

Then follow the prompts to choose files by number and to output to screen or to
a `json`/`txt` file.
