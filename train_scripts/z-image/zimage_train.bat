@echo off
setlocal enabledelayedexpansion
title Musubi Z-Image LoRA Training (generic / parameter-driven)

REM =====================================================================
REM  Generic Z-Image (De-Turbo) LoRA training. Consumes the TOML produced
REM  by zimage_cache.bat (matched by --name), so run cache first.
REM
REM  DiT = De-Turbo (training-stable); the LoRA runs on Z-Image Turbo at
REM  inference with --guidance_scale 0.
REM
REM  Auto-resume: if a "<name>_zimage-NNNNNN-state" folder exists in the
REM  output dir, it resumes and runs only the REMAINING epochs to reach
REM  --target-epochs (musubi restarts its epoch counter on --resume).
REM
REM  USAGE:
REM    zimage_train.bat --name tammy [options]
REM
REM  Common options (defaults match the tammy template):
REM    --target-epochs 23   --dim 64   --alpha 32   --lr 8e-5   --save-every 1
REM    --output-name <name>_zimage   --output-root D:\DATA\training\zimage_loras
REM
REM  Sampling toggle:
REM    --samples on|off      (default: on)
REM    --sample-prompts <file>   (required when --samples on)
REM    --sample-every 1
REM
REM  Trigger word:
REM    --trigger <word>      stamp into output LoRA metadata after training
REM                          (comma-separated for multiple, e.g. "tammy, corset")
REM =====================================================================

REM ---- load machine paths from config.bat (run setup.bat to create it) ----
set "CONFIG=%~dp0..\config.bat"
if not exist "%CONFIG%" ( echo ERROR: config not found: %CONFIG% & echo Run setup.bat in the train_scripts folder once to create it. & exit /b 1 )
call "%CONFIG%"

REM ---- fixed paths / defaults ----
set "GEN_DIR=%~dp0_generated"
set "DIT=%COMFY_MODELS%\diffusion_models\z_image_de_turbo_v1_bf16.safetensors"
set "VAE=%COMFY_MODELS%\vae\z_image_ae.safetensors"
set "TE=%COMFY_MODELS%\text_encoders\qwen_3_4b.safetensors"

set "NAME="
set "OUTPUT_ROOT=%TRAINING_ROOT%\zimage_loras"
set "OUTPUT_NAME="
set "TARGET_EPOCHS=23"
set "DIM=64"
set "ALPHA=32"
set "LR=8e-5"
set "SAVE_EVERY=1"
set "SAMPLES=on"
set "SAMPLE_PROMPTS="
set "SAMPLE_EVERY=1"
set "TRIGGER="
set "DRYRUN=0"

REM ---- parse args ----
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--name"           ( set "NAME=%~2" & shift & shift & goto parse )
if /i "%~1"=="--output-root"    ( set "OUTPUT_ROOT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--output-name"    ( set "OUTPUT_NAME=%~2" & shift & shift & goto parse )
if /i "%~1"=="--target-epochs"  ( set "TARGET_EPOCHS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dim"            ( set "DIM=%~2" & shift & shift & goto parse )
if /i "%~1"=="--alpha"          ( set "ALPHA=%~2" & shift & shift & goto parse )
if /i "%~1"=="--lr"             ( set "LR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--save-every"     ( set "SAVE_EVERY=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dit"            ( set "DIT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--vae"            ( set "VAE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--text-encoder"   ( set "TE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--samples"        ( set "SAMPLES=%~2" & shift & shift & goto parse )
if /i "%~1"=="--sample-prompts" ( set "SAMPLE_PROMPTS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--sample-every"   ( set "SAMPLE_EVERY=%~2" & shift & shift & goto parse )
if /i "%~1"=="--trigger"        ( set "TRIGGER=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"        ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"=="" ( echo ERROR: --name is required & exit /b 1 )
if "%OUTPUT_NAME%"=="" set "OUTPUT_NAME=%NAME%_zimage"
set "OUT=%OUTPUT_ROOT%\%NAME%"
set "OUTNAME=%OUTPUT_NAME%"
set "TOML=%GEN_DIR%\%NAME%_zimage.toml"

if not exist "%TOML%" (
    echo ERROR: dataset TOML not found: %TOML%
    echo Run zimage_cache.bat --name %NAME% --dataset ... first.
    exit /b 1
)

REM ---- sampling toggle ----
if /i "%SAMPLES%"=="on" (
    if "%SAMPLE_PROMPTS%"=="" ( echo ERROR: --samples on requires --sample-prompts ^<file^> & exit /b 1 )
    if not exist "%SAMPLE_PROMPTS%" ( echo ERROR: sample prompts file not found: %SAMPLE_PROMPTS% & exit /b 1 )
    set "SAMPLE_ARGS=--sample_prompts "%SAMPLE_PROMPTS%" --sample_every_n_epochs %SAMPLE_EVERY%"
    set "SAMPLE_MSG=on, every %SAMPLE_EVERY% epochs"
) else (
    set "SAMPLE_ARGS="
    set "SAMPLE_MSG=off"
)

REM ---- auto-resume: find newest -state and compute remaining epochs ----
set "RESUME="
set "STATEDIR="
for /f "delims=" %%D in ('dir /b /ad /o-n "%OUT%\%OUTNAME%-*-state" 2^>nul') do if not defined RESUME ( set "RESUME=%OUT%\%%D" & set "STATEDIR=%%D" )

set "RESUMEARG="
set "MAXEPOCHS=%TARGET_EPOCHS%"
if defined RESUME (
    set "RESUMEARG=--resume !RESUME!"
    set "EPNUM=!STATEDIR:%OUTNAME%-=!"
    set "EPNUM=!EPNUM:-state=!"
    for /f "tokens=* delims=0" %%n in ("!EPNUM!") do set "EPNUM=%%n"
    if not defined EPNUM set "EPNUM=0"
    set /a "MAXEPOCHS=%TARGET_EPOCHS%-!EPNUM!"
    if !MAXEPOCHS! LSS 1 set "MAXEPOCHS=1"
    echo Resuming from: !RESUME!
    echo   completed epoch !EPNUM! of %TARGET_EPOCHS% - running !MAXEPOCHS! more epoch^(s^)
) else (
    echo No saved state found - training fresh for %TARGET_EPOCHS% epochs.
)

echo.
echo Z-Image LoRA training - %NAME%
echo   Config:   %TOML%
echo   Output:   %OUT%\%OUTNAME%
echo   Epochs:   !MAXEPOCHS! (target %TARGET_EPOCHS%)   dim/alpha: %DIM%/%ALPHA%   lr: %LR%
echo   Sampling: !SAMPLE_MSG!

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] Resume arg:   !RESUMEARG!
    echo [DRY RUN] Sampling args: !SAMPLE_ARGS!
    echo [DRY RUN] Not launching training.
    endlocal & exit /b 0
)

if not exist "%DIT%" ( echo ERROR: DiT not found: %DIT% & exit /b 1 )
if not exist "%VAE%" ( echo ERROR: VAE not found: %VAE% & exit /b 1 )
if not exist "%TE%"  ( echo ERROR: text encoder not found: %TE% & exit /b 1 )

cd /d "%MUSUBI_DIR%"
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_ALLOC_CONF=expandable_segments:True
set PYTHONIOENCODING=utf-8

if not "!TRIGGER!"=="" ( set "COMMENT_ARG=--training_comment "!TRIGGER!"" ) else ( set "COMMENT_ARG=" )

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 zimage_train_network.py ^
  --dit "%DIT%" ^
  --vae "%VAE%" ^
  --text_encoder "%TE%" ^
  --dataset_config "%TOML%" ^
  --sdpa ^
  --mixed_precision bf16 ^
  --timestep_sampling shift ^
  --weighting_scheme none ^
  --discrete_flow_shift 2.0 ^
  --optimizer_type adamw8bit ^
  --learning_rate %LR% ^
  --fp8_base ^
  --fp8_scaled ^
  --fp8_llm ^
  --gradient_checkpointing ^
  --split_attn ^
  --max_data_loader_n_workers 2 ^
  --persistent_data_loader_workers ^
  --network_module networks.lora_zimage ^
  --network_dim %DIM% ^
  --network_alpha %ALPHA% ^
  --max_train_epochs !MAXEPOCHS! ^
  --save_every_n_epochs %SAVE_EVERY% ^
  --save_state ^
  --seed 42 ^
  !RESUMEARG! ^
  !SAMPLE_ARGS! ^
  !COMMENT_ARG! ^
  --output_dir "%OUT%" ^
  --output_name "%OUTNAME%"

if %ERRORLEVEL% NEQ 0 ( echo. & echo Training failed with error code %errorlevel% & pause & exit /b 1 )

if not "!TRIGGER!"=="" (
    echo.
    echo Stamping trigger word "!TRIGGER!" into output LoRAs...
    python "%~dp0..\write_trigger.py" --dir "%OUT%" --trigger "!TRIGGER!"
)

echo.
echo Done. LoRA: %OUT%\%OUTNAME%.safetensors
echo Convert for ComfyUI:
echo   python src\musubi_tuner\convert_lora.py --input "%OUT%\%OUTNAME%.safetensors" --output "%OUT%\%OUTNAME%_comfy.safetensors" --target other
pause
endlocal
