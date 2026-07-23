@echo off
REM =====================================================================
REM  train_scripts machine config (TEMPLATE)
REM
REM  This file is CALLED by every cache/train script to learn where your
REM  repos, models, and output folders live. Do NOT run it directly.
REM
REM  Two ways to create your own config.bat:
REM    1. Run  setup.bat  (interactive - recommended), or
REM    2. Copy this file to  config.bat  and edit the paths below.
REM
REM  config.bat is git-ignored, so your local paths stay out of the repo.
REM
REM  Only these base locations vary per machine. The individual model
REM  filenames (krea2-raw.safetensors, wan2.2_*, z_image_*, ...) are set
REM  inside each script from the folders below, and can still be pointed
REM  elsewhere per-run with flags like --vae / --dit / --checkpoint.
REM =====================================================================

REM ---- musubi-tuner repos (clone locations) ----
set "MUSUBI_DIR=D:\github\musubi-tuner"
set "MUSUBI_LTX_DIR=D:\github\musubi-tuner-ltx"

REM ---- where trained LoRAs are written (each arch adds \<arch>_loras) ----
set "TRAINING_ROOT=D:\DATA\training"

REM ---- HuggingFace cache (krea2 caching/train) ----
set "HF_HOME=D:\hf-cache"

REM ---- ComfyUI models root (krea2 / z-image / wan / ltx checkpoints live under here) ----
set "COMFY_MODELS=D:\comfyui\ComfyUI\models"

REM ---- FLUX.2 model folder (Klein dit + ae + text_encoder) ----
set "FLUX2_DIR=D:\ai\models\FLUX2"

REM ---- LTX-2.3 Gemma text-encoder folder ----
set "LTX_GEMMA_ROOT=D:\ai\models\LTX-2.3\gemma"

REM ---- LTX-2.3 DiT checkpoint (full path; this is often a personal finetune) ----
set "LTX_CHECKPOINT=D:\comfyui\ComfyUI\models\diffusion_models\LTX23\ltx2310eros_v1.safetensors"
