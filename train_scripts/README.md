# train_scripts — parameter-driven LoRA cache/train drivers

Reusable cache + train scripts for five architectures. Instead of copying and
hand-editing a `.toml` + `-cache.bat` + `-train.bat` per subject, you run one
generic driver per architecture and pass everything as command-line flags.

```
train_scripts/
  setup.bat                     # one-time: writes config.bat (asks for your paths)
  config.example.bat            # template of the config.bat setup.bat generates
  config.bat                    # YOUR machine paths (git-ignored; created by setup)
  render_toml.py                # shared stdlib TOML renderer (no deps)
  krea2/   krea2_cache.bat  krea2_train.bat
  Klein/   klein_cache.bat  klein_train.bat
  z-image/ zimage_cache.bat zimage_train.bat
  wan22/   wan22_cache.bat  wan22_train.bat
  musubi-tuner-ltx/  ltx_cache.bat  ltx_train.bat
  <arch>/_generated/            # auto-generated <name>_<arch>.toml files live here
```

## Setup (run once per machine)

The scripts don't hard-code where your repos and models live — they read those
paths from `config.bat`. Create it once:

```bat
setup.bat
```

`setup.bat` asks for eight locations (press Enter to accept each default) and
writes `config.bat`. Re-run it any time to change a path; your current values
become the new defaults. Prefer editing by hand? Copy `config.example.bat` to
`config.bat` and edit the paths. `config.bat` is git-ignored, so your local
paths never get committed.

The eight locations:

| Variable | What it points at |
|----------|-------------------|
| `MUSUBI_DIR` | `musubi-tuner` repo (krea2 / klein / z-image / wan) |
| `MUSUBI_LTX_DIR` | `musubi-tuner-ltx` repo (LTX-2.3) |
| `COMFY_MODELS` | ComfyUI `models` root (diffusion_models / vae / text_encoders / clip) |
| `FLUX2_DIR` | FLUX.2 folder (Klein dit + ae + text_encoder) |
| `LTX_GEMMA_ROOT` | LTX-2.3 Gemma text-encoder folder |
| `LTX_CHECKPOINT` | LTX-2.3 DiT checkpoint (full path — often a personal finetune) |
| `TRAINING_ROOT` | output root; each arch adds its own `_loras` subfolder |
| `HF_HOME` | HuggingFace cache (krea2) |

Individual model **filenames** (`krea2-raw.safetensors`, `wan2.2_*`,
`z_image_*`, …) are derived from these folders inside each script. If yours
differ, either edit the script default or override per-run with the existing
flags (`--vae`, `--dit`, `--checkpoint`, `--text-encoder`, `--gemma-root`, …).
If you run a script before `config.bat` exists, it stops and tells you to run
`setup.bat`.

Sampling support by architecture: **krea2 / klein / z-image have the `--samples`
toggle; LTX and WAN do not** (those models have no in-training sampling).

## How it works

1. **Cache** renders `_generated/<name>_<arch>.toml` from your flags, then runs
   the two caching steps (VAE latents + text-encoder outputs).
2. **Train** looks up that same `_generated/<name>_<arch>.toml` by `--name` and
   launches training. So **always run cache before train**, and use the same
   `--name` for both.

Cache directories are derived automatically from the subject name, e.g.
`D:/github/musubi-tuner/cache/<name>_krea2/img`.

Add `--dry-run` to any script to render the TOML and print the resolved config
**without** launching caching or training — handy for a sanity check.

> **Note on commas:** `.bat` argument parsing treats a comma as a delimiter, so
> resolution uses an `x` separator (`--res 256x384`). Any comma-separated value
> you override on the command line (e.g. `--video-frames`) **must be quoted**:
> `--video-frames "1,9,17,25"`.

## Krea2

```bat
krea2\krea2_cache.bat --name courtney --dataset "D:/DATA/training/datasets/courtney/img" --repeats 2 --res 256x384
krea2\krea2_train.bat --name courtney --epochs 40 --samples on --sample-prompts "D:\github\musubi-tuner\train\courtney_krea2_samples.txt"
```

Defaults match the courtney template: res `256,384`, repeats `2`, epochs `40`,
dim/alpha `32`, blocks_to_swap `14`.

## Klein (FLUX.2 Klein 9B)

```bat
Klein\klein_cache.bat --name courtney --dataset "D:/DATA/training/datasets/courtney/img" --repeats 3 --res 512x768
Klein\klein_train.bat --name courtney --epochs 16 --samples off
```

Defaults match the courtney template: res `512,768`, repeats `3`, epochs `16`,
dim/alpha `32`, blocks_to_swap `6`.

## Z-Image (De-Turbo)

Main image dataset plus an optional `--extra-dir` image slot (e.g. a nudes set).
Has the `--samples` toggle. Training **auto-resumes**: if a `-state` folder
exists it runs only the remaining epochs to reach `--target-epochs`.

```bat
z-image\zimage_cache.bat --name tammy --dataset "D:/DATA/.../tammy/images" --repeats 3 ^
  --extra-dir "D:/DATA/.../tammy/nude_test" --extra-repeats 2 --res 1024x1024
z-image\zimage_train.bat --name tammy --target-epochs 23 --samples on --sample-prompts "...\tammy_zimage_samples.txt"
```

Defaults match the tammy template: res `1024x1024`, dim/alpha `64/32`, lr `8e-5`,
target-epochs `23`. Trained on De-Turbo; run the LoRA on Z-Image Turbo at
inference with `--guidance_scale 0`.

Z-Image LoRAs need a one-time conversion before ComfyUI can load them (the
train script prints the exact command on completion):

```bat
python src\musubi_tuner\convert_lora.py ^
  --input  "D:\DATA\training\zimage_loras\tammy\tammy_zimage.safetensors" ^
  --output "D:\DATA\training\zimage_loras\tammy\tammy_zimage_comfy.safetensors" ^
  --target other
```

## WAN 2.2 (T2V, dual high+low noise)

Three optional dataset slots (video/image/extra), like LTX. No `--samples` flag.
Uses the same VAE + T5 as WAN 2.1, and `--skip_existing` keeps re-caching cheap.

```bat
wan22\wan22_cache.bat --name tammy ^
  --video-dir "D:/DATA/training/tammy_2025/videos" --video-repeats 3 --video-frames "1,4" ^
  --image-dir "D:/DATA/training/tammy_2025/images" --image-repeats 3 ^
  --extra-dir "D:/DATA/training/tammy_2025/nude_test" --extra-repeats 4 --res 480x640

wan22\wan22_train.bat --name tammy --epochs 16
```

Defaults match the tammy template: res `480x640`, dim/alpha `16/16`,
blocks_to_swap `30`, epochs `16`, task `t2v-A14B`, dual high+low noise DiT.
The WAN video slot uses `target_frames`/`frame_extraction` with **no**
`frame_sample` (pass `--frame-sample 4` to add it).

## LTX-2.3

Three optional dataset slots — pass a directory to enable each. At least one is
required. LTX has **no in-training sampling** (same as WAN), so there is no
`--samples` flag.

```bat
musubi-tuner-ltx\ltx_cache.bat --name tammy ^
  --video-dir "D:/DATA/training/tammy_2025/videos" --video-repeats 4 --video-frames 1,9,17,25 ^
  --image-dir "D:/DATA/training/tammy_2025/images" --image-repeats 2 ^
  --extra-dir "D:/DATA/training/tammy_2025/nude_test" --extra-repeats 4

musubi-tuner-ltx\ltx_train.bat --name tammy --epochs 24
```

Defaults match the tammy_eros template: res `512,768`, dim/alpha `64`,
blocks_to_swap `14`, epochs `24`, preset `t2v`. LTX training **auto-resumes**
(`--autoresume`): re-running with the same `--name`/output picks up from the
latest saved state. Metadata flags (`--meta-title`, `--meta-author`,
`--meta-desc`, `--meta-tags`) are LTX-only and default off the subject name.

## Samples toggle (krea2 / klein / z-image)

- `--samples on` (default) — enables `--sample_prompts` + `--sample_every_n_epochs`.
  krea2/klein also add `--sample_at_first`; z-image does not. Requires
  `--sample-prompts <file>`.
- `--samples off` — disables sampling entirely.
- `--sample-every N` — epochs between samples (default 2; z-image default 1).

WAN and LTX have no in-training sampling, so neither exposes `--samples`.

## Trigger words (all architectures)

Pass `--trigger <word>` to any `*_train.bat`. This does two things:

1. **Native:** passes `--training_comment "<word>"` to the trainer, so every
   checkpoint gets `ss_training_comment` written *during* training (survives even
   if the post-step is skipped).
2. **Post-step:** after a successful run, `write_trigger.py` stamps the output
   dir with the fields ComfyUI trigger tools actually read — `ss_tag_frequency`
   (rgthree Power Lora Loader's *Show Info* keys off this; musubi-tuner does
   **not** write it — there's a `# TODO support tag frequency` in its trainer),
   plus `modelspec.trigger_phrase` and `ss_training_comment`.

Existing training tags are preserved; the trigger is given a dominating count so
it sorts first. Every saved epoch checkpoint in the output dir gets stamped. No
retrain needed for either part.

```bat
krea2\krea2_train.bat --name courtney --trigger c0urtney ...
```

Comma-separate for multiple words: `--trigger "c0urtney, corset"`.

To stamp a LoRA that's **already trained**, run the shared tool directly (from
the `musubi` conda env, which has `safetensors`):

```bat
python train_scripts\write_trigger.py --file "D:\...\courtney_krea2.safetensors" --trigger c0urtney
python train_scripts\write_trigger.py --dir  "D:\...\courtney_krea2"             --trigger c0urtney
```

After stamping, refresh the LoRA in ComfyUI (or restart) so the frontend
re-reads the header.

## Overriding defaults

Every tunable is a flag: `--epochs`, `--dim`, `--alpha`, `--lr`,
`--blocks-to-swap`, `--save-every`, `--output-root`, `--output-name`, and the
model paths (`--vae`, `--text-encoder`, `--checkpoint`, `--gemma-root`, ...).
Run a script with no args to see its usage header.
