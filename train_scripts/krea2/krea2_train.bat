@echo off
setlocal enabledelayedexpansion
title Musubi Krea2 RAW LoRA Training (generic / parameter-driven)

REM =====================================================================
REM  Generic Krea2 LoRA training. Consumes the TOML produced by
REM  krea2_cache.bat (matched by --name), so run cache first.
REM
REM  USAGE:
REM    krea2_train.bat --name courtney [options]
REM
REM  Common options (defaults match the courtney template):
REM    --epochs 40            --repeats is set at cache time (in the TOML)
REM    --dim 32   --alpha 32  --lr 1e-4   --blocks-to-swap 14
REM    --save-every 1  --save-last 20  --grad-accum 4
REM    --output-name <name>_krea2   --output-root D:\DATA\training\krea2_loras
REM
REM  Sampling toggle (NEW):
REM    --samples on|off      (default: on)
REM    --sample-prompts <file>   (required when --samples on)
REM    --sample-every 2
REM =====================================================================

REM ---- fixed paths / defaults ----
set "MUSUBI_DIR=D:\github\musubi-tuner"
set "GEN_DIR=%~dp0_generated"
set "DIT_RAW=D:\comfyui\ComfyUI\models\diffusion_models\krea2-raw.safetensors"
set "VAE=D:\comfyui\ComfyUI\models\vae\qwen_image_vae.safetensors"
set "TEXT_ENCODER=D:\comfyui\ComfyUI\models\text_encoders\Qwen3-VL-4B-Instruct\model-00001-of-00002.safetensors"

set "NAME="
set "OUTPUT_ROOT=D:\DATA\training\krea2_loras"
set "OUTPUT_NAME="
set "EPOCHS=40"
set "DIM=32"
set "ALPHA=32"
set "LR=1e-4"
set "BLOCKS=14"
set "SAVE_EVERY=1"
set "SAVE_LAST=20"
set "GRAD_ACCUM=4"
set "SAMPLES=on"
set "SAMPLE_PROMPTS="
set "SAMPLE_EVERY=2"
set "DRYRUN=0"

REM ---- parse args ----
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--name"           ( set "NAME=%~2" & shift & shift & goto parse )
if /i "%~1"=="--output-root"    ( set "OUTPUT_ROOT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--output-name"    ( set "OUTPUT_NAME=%~2" & shift & shift & goto parse )
if /i "%~1"=="--epochs"         ( set "EPOCHS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dim"            ( set "DIM=%~2" & shift & shift & goto parse )
if /i "%~1"=="--alpha"          ( set "ALPHA=%~2" & shift & shift & goto parse )
if /i "%~1"=="--lr"             ( set "LR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--blocks-to-swap" ( set "BLOCKS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--save-every"     ( set "SAVE_EVERY=%~2" & shift & shift & goto parse )
if /i "%~1"=="--save-last"      ( set "SAVE_LAST=%~2" & shift & shift & goto parse )
if /i "%~1"=="--grad-accum"     ( set "GRAD_ACCUM=%~2" & shift & shift & goto parse )
if /i "%~1"=="--samples"        ( set "SAMPLES=%~2" & shift & shift & goto parse )
if /i "%~1"=="--sample-prompts" ( set "SAMPLE_PROMPTS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--sample-every"   ( set "SAMPLE_EVERY=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"        ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"=="" ( echo ERROR: --name is required & exit /b 1 )
if "%OUTPUT_NAME%"=="" set "OUTPUT_NAME=%NAME%_krea2"
set "OUTPUT_DIR=%OUTPUT_ROOT%\%OUTPUT_NAME%"
set "TOML=%GEN_DIR%\%NAME%_krea2.toml"

if not exist "%TOML%" (
    echo ERROR: dataset TOML not found: %TOML%
    echo Run krea2_cache.bat --name %NAME% --dataset ... first.
    exit /b 1
)

REM ---- sampling toggle ----
if /i "%SAMPLES%"=="on" (
    if "%SAMPLE_PROMPTS%"=="" ( echo ERROR: --samples on requires --sample-prompts ^<file^> & exit /b 1 )
    if not exist "%SAMPLE_PROMPTS%" ( echo ERROR: sample prompts file not found: %SAMPLE_PROMPTS% & exit /b 1 )
    set "SAMPLE_ARGS=--sample_prompts "%SAMPLE_PROMPTS%" --sample_every_n_epochs %SAMPLE_EVERY% --sample_at_first"
    set "SAMPLE_MSG=on, every %SAMPLE_EVERY% epochs"
) else (
    set "SAMPLE_ARGS="
    set "SAMPLE_MSG=off"
)

echo.
echo Krea2 RAW LoRA training - %NAME%
echo   Config:   %TOML%
echo   Output:   %OUTPUT_DIR%\%OUTPUT_NAME%
echo   Epochs:   %EPOCHS%   dim/alpha: %DIM%/%ALPHA%   lr: %LR%   blocks_to_swap: %BLOCKS%
echo   Sampling: !SAMPLE_MSG!
echo   NOTE: blocks_to_swap=%BLOCKS% for 16GB. Increase if OOM.

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] Sampling args: !SAMPLE_ARGS!
    echo [DRY RUN] Not launching training.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set CUDA_VISIBLE_DEVICES=0
set HF_HOME=D:\hf-cache
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:1024
set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
set CUDA_MODULE_LOADING=LAZY

accelerate launch ^
  --num_cpu_threads_per_process 8 ^
  --mixed_precision bf16 ^
  src\musubi_tuner\krea2_train_network.py ^
  --dataset_config "%TOML%" ^
  --dit "%DIT_RAW%" ^
  --vae "%VAE%" ^
  --mixed_precision bf16 ^
  --timestep_sampling krea2_shift ^
  --weighting_scheme none ^
  --network_module networks.lora_krea2 ^
  --network_dim %DIM% ^
  --network_alpha %ALPHA% ^
  --fp8_base ^
  --fp8_scaled ^
  --gradient_checkpointing ^
  --blocks_to_swap %BLOCKS% ^
  --sdpa ^
  --optimizer_type adafactor ^
  --learning_rate %LR% ^
  --max_grad_norm 1.0 ^
  --gradient_accumulation_steps %GRAD_ACCUM% ^
  --max_train_epochs %EPOCHS% ^
  --save_every_n_epochs %SAVE_EVERY% ^
  --save_last_n_epochs %SAVE_LAST% ^
  --save_state ^
  --text_encoder "%TEXT_ENCODER%" ^
  !SAMPLE_ARGS! ^
  --output_dir "%OUTPUT_DIR%" ^
  --output_name "%OUTPUT_NAME%" ^
  --max_data_loader_n_workers 2 ^
  --persistent_data_loader_workers ^
  --seed 42

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Training failed with error code %ERRORLEVEL%
)

pause
endlocal
