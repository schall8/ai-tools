@echo off
setlocal enabledelayedexpansion
title Musubi LTX-2.3 LoRA Training (generic / parameter-driven)

REM =====================================================================
REM  Generic LTX-2.3 LoRA training. Consumes the TOML produced by
REM  ltx_cache.bat (matched by --name), so run cache first.
REM
REM  NOTE: LTX (like WAN) has no in-training sampling, so there is no
REM  --samples toggle here.
REM
REM  USAGE:
REM    ltx_train.bat --name tammy [options]
REM
REM  Common options (defaults match the tammy_eros template):
REM    --epochs 24   --dim 64   --alpha 64   --lr 1e-4   --blocks-to-swap 14
REM    --save-every 2   --save-last 30   --grad-accum 4   --warmup-steps 100
REM    --target-preset t2v
REM    --output-name <name>_ltx   --output-root D:\DATA\training\ltx_loras
REM    --meta-title / --meta-author / --meta-desc / --meta-tags
REM    --trigger <word>      stamp into output LoRA metadata (comma-sep for many)
REM =====================================================================

REM ---- fixed paths / defaults ----
set "MUSUBI_DIR=D:\github\musubi-tuner-ltx"
set "GEN_DIR=%~dp0_generated"
set "LTX_CHECKPOINT=D:\comfyui\ComfyUI\models\diffusion_models\LTX23\ltx2310eros_v1.safetensors"
set "GEMMA_ROOT=D:\ai\models\LTX-2.3\gemma"

set "NAME="
set "OUTPUT_ROOT=D:\DATA\training\ltx_loras"
set "OUTPUT_NAME="
set "EPOCHS=24"
set "DIM=64"
set "ALPHA=64"
set "LR=1e-4"
set "BLOCKS=14"
set "SAVE_EVERY=2"
set "SAVE_LAST=30"
set "GRAD_ACCUM=4"
set "WARMUP=100"
set "TARGET_PRESET=t2v"
set "META_TITLE="
set "META_AUTHOR=AI Guy"
set "META_DESC="
set "META_TAGS="
set "TRIGGER="
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
if /i "%~1"=="--warmup-steps"   ( set "WARMUP=%~2" & shift & shift & goto parse )
if /i "%~1"=="--target-preset"  ( set "TARGET_PRESET=%~2" & shift & shift & goto parse )
if /i "%~1"=="--checkpoint"     ( set "LTX_CHECKPOINT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--gemma-root"     ( set "GEMMA_ROOT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--meta-title"     ( set "META_TITLE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--meta-author"    ( set "META_AUTHOR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--meta-desc"      ( set "META_DESC=%~2" & shift & shift & goto parse )
if /i "%~1"=="--meta-tags"      ( set "META_TAGS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--trigger"        ( set "TRIGGER=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"        ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"=="" ( echo ERROR: --name is required & exit /b 1 )
if "%OUTPUT_NAME%"=="" set "OUTPUT_NAME=%NAME%_ltx"
if "%META_TITLE%"=="" set "META_TITLE=%NAME% LTX-2.3 LoRA"
if "%META_DESC%"=="" set "META_DESC=LTX-2.3 LoRA trained on %NAME%"
if "%META_TAGS%"=="" set "META_TAGS=%NAME%"
set "OUTPUT_DIR=%OUTPUT_ROOT%\%OUTPUT_NAME%"
set "TOML=%GEN_DIR%\%NAME%_ltx.toml"

if not exist "%TOML%" (
    echo ERROR: dataset TOML not found: %TOML%
    echo Run ltx_cache.bat --name %NAME% ... first.
    exit /b 1
)

echo.
echo LTX-2.3 LoRA training - %NAME%
echo   Config:   %TOML%
echo   Output:   %OUTPUT_DIR%\%OUTPUT_NAME%
echo   Epochs:   %EPOCHS%   dim/alpha: %DIM%/%ALPHA%   lr: %LR%   blocks_to_swap: %BLOCKS%
echo   Preset:   %TARGET_PRESET%   (no in-training sampling for LTX)

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] Not launching training.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set TORCHINDUCTOR_FREEZING=1
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
set CUDA_MODULE_LOADING=LAZY

if not "!TRIGGER!"=="" ( set "COMMENT_ARG=--training_comment "!TRIGGER!"" ) else ( set "COMMENT_ARG=" )

accelerate launch ^
  --num_cpu_threads_per_process 1 ^
  --mixed_precision bf16 ^
  ltx2_train_network.py ^
  --dataset_config "%TOML%" ^
  --ltx2_checkpoint "%LTX_CHECKPOINT%" ^
  --ltx2_mode v ^
  --ltx_version 2.3 ^
  --cuda_allow_tf32 ^
  --gemma_root "%GEMMA_ROOT%" ^
  --gemma_load_in_4bit ^
  --gemma_bnb_4bit_quant_type nf4 ^
  --network_module networks.lora_ltx2 ^
  --network_dim %DIM% ^
  --network_alpha %ALPHA% ^
  --lora_target_preset %TARGET_PRESET% ^
  --fp8_base ^
  --fp8_scaled ^
  --gradient_checkpointing ^
  --blocks_to_swap %BLOCKS% ^
  --sdpa ^
  --optimizer_type adafactor ^
  --optimizer_args relative_step=False scale_parameter=False warmup_init=False ^
  --learning_rate %LR% ^
  --lr_scheduler cosine_with_restarts ^
  --lr_warmup_steps %WARMUP% ^
  --max_grad_norm 1.0 ^
  --shifted_logit_shift 0.5 ^
  --gradient_accumulation_steps %GRAD_ACCUM% ^
  --max_train_epochs %EPOCHS% ^
  --save_every_n_epochs %SAVE_EVERY% ^
  --save_last_n_epochs %SAVE_LAST% ^
  --output_dir "%OUTPUT_DIR%" ^
  --output_name "%OUTPUT_NAME%" ^
  --caption_dropout_rate 0.05 ^
  --seed 42 ^
  !COMMENT_ARG! ^
  --metadata_title "%META_TITLE%" ^
  --metadata_author "%META_AUTHOR%" ^
  --metadata_description "%META_DESC%" ^
  --metadata_tags "%META_TAGS%" ^
  --save_state ^
  --autoresume ^
  --logging_dir "%OUTPUT_DIR%\logs"

set "TRAINRC=%ERRORLEVEL%"
if not "%TRAINRC%"=="0" (
    echo.
    echo Training failed with error code %TRAINRC%
) else (
    if not "!TRIGGER!"=="" (
        echo.
        echo Stamping trigger word "!TRIGGER!" into output LoRAs...
        python "%~dp0..\write_trigger.py" --dir "%OUTPUT_DIR%" --trigger "!TRIGGER!"
    )
)

pause
endlocal
