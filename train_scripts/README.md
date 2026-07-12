# train_scripts — parameter-driven LoRA cache/train drivers

Reusable cache + train scripts for three architectures. Instead of copying and
hand-editing a `.toml` + `-cache.bat` + `-train.bat` per subject, you run one
generic driver per architecture and pass everything as command-line flags.

```
train_scripts/
  render_toml.py                # shared stdlib TOML renderer (no deps)
  krea2/   krea2_cache.bat  krea2_train.bat
  Klein/   klein_cache.bat  klein_train.bat
  z-image/ zimage_cache.bat zimage_train.bat
  wan22/   wan22_cache.bat  wan22_train.bat
  musubi-tuner-ltx/  ltx_cache.bat  ltx_train.bat
  <arch>/_generated/            # auto-generated <name>_<arch>.toml files live here
```

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
blocks_to_swap `14`, epochs `24`, preset `t2v`.

## Samples toggle (krea2 / klein only)

- `--samples on` (default) — enables `--sample_prompts` + `--sample_every_n_epochs`
  + `--sample_at_first`. Requires `--sample-prompts <file>`.
- `--samples off` — disables sampling entirely.
- `--sample-every N` — epochs between samples (default 2).

## Overriding defaults

Every tunable is a flag: `--epochs`, `--dim`, `--alpha`, `--lr`,
`--blocks-to-swap`, `--save-every`, `--output-root`, `--output-name`, and the
model paths (`--vae`, `--text-encoder`, `--checkpoint`, `--gemma-root`, ...).
Run a script with no args to see its usage header.
