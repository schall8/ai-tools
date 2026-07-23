@echo off
setlocal enabledelayedexpansion
title Musubi FLUX.2 Klein 9B LoRA Training (generic / parameter-driven)

REM =====================================================================
REM  Generic FLUX.2 Klein LoRA training. Consumes the TOML produced by
REM  klein_cache.bat (matched by --name), so run cache first.
REM
REM  USAGE:
REM    klein_train.bat --name courtney [options]
REM
REM  Common options (defaults match the courtney template):
REM    --epochs 16   --dim 32   --alpha 32   --lr 1e-4   --blocks-to-swap 6
REM    --save-every 1   --warmup-steps 100
REM    --output-name <name>_klein   --output-root D:\DATA\training\klein_loras
REM
REM  Sampling toggle (NEW):
REM    --samples on|off      (default: on)
REM    --sample-prompts <file>   (required when --samples on)
REM    --sample-every 2
REM
REM  Trigger word (NEW):
REM    --trigger <word>      stamp into output LoRA metadata after training
REM                          (comma-separated for multiple, e.g. "c0urtney, corset")
REM =====================================================================

REM ---- load machine paths from config.bat (run setup.bat to create it) ----
set "CONFIG=%~dp0..\config.bat"
if not exist "%CONFIG%" ( echo ERROR: config not found: %CONFIG% & echo Run setup.bat in the train_scripts folder once to create it. & exit /b 1 )
call "%CONFIG%"

REM ---- fixed paths / defaults ----
set "GEN_DIR=%~dp0_generated"
set "DIT_CHECKPOINT=%FLUX2_DIR%\flux-2-klein-base-9b.safetensors"
set "VAE_CHECKPOINT=%FLUX2_DIR%\ae.safetensors"
set "TEXT_ENCODER=%FLUX2_DIR%\text_encoder\model-00001-of-00004.safetensors"
set "MODEL_VERSION=klein-base-9b"

set "NAME="
set "OUTPUT_ROOT=%TRAINING_ROOT%\klein_loras"
set "OUTPUT_NAME="
set "EPOCHS=16"
set "DIM=32"
set "ALPHA=32"
set "LR=1e-4"
set "BLOCKS=6"
set "SAVE_EVERY=1"
set "WARMUP=100"
set "SAMPLES=on"
set "SAMPLE_PROMPTS="
set "SAMPLE_EVERY=2"
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
if /i "%~1"=="--warmup-steps"   ( set "WARMUP=%~2" & shift & shift & goto parse )
if /i "%~1"=="--model-version"  ( set "MODEL_VERSION=%~2" & shift & shift & goto parse )
if /i "%~1"=="--samples"        ( set "SAMPLES=%~2" & shift & shift & goto parse )
if /i "%~1"=="--sample-prompts" ( set "SAMPLE_PROMPTS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--sample-every"   ( set "SAMPLE_EVERY=%~2" & shift & shift & goto parse )
if /i "%~1"=="--trigger"        ( set "TRIGGER=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"        ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"=="" ( echo ERROR: --name is required & exit /b 1 )
if "%OUTPUT_NAME%"=="" set "OUTPUT_NAME=%NAME%_klein"
set "OUTPUT_DIR=%OUTPUT_ROOT%\%OUTPUT_NAME%"
set "TOML=%GEN_DIR%\%NAME%_klein.toml"

if not exist "%TOML%" (
    echo ERROR: dataset TOML not found: %TOML%
    echo Run klein_cache.bat --name %NAME% --dataset ... first.
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
echo FLUX.2 Klein 9B LoRA training - %NAME%
echo   Config:   %TOML%
echo   Output:   %OUTPUT_DIR%\%OUTPUT_NAME%
echo   Epochs:   %EPOCHS%   dim/alpha: %DIM%/%ALPHA%   lr: %LR%   blocks_to_swap: %BLOCKS%
echo   Sampling: !SAMPLE_MSG!

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] Sampling args: !SAMPLE_ARGS!
    echo [DRY RUN] Not launching training.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set TORCHINDUCTOR_FREEZING=1
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:1024
set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
set CUDA_MODULE_LOADING=LAZY

if not "!TRIGGER!"=="" ( set "COMMENT_ARG=--training_comment "!TRIGGER!"" ) else ( set "COMMENT_ARG=" )

accelerate launch ^
  --num_cpu_threads_per_process 8 ^
  --mixed_precision bf16 ^
  src\musubi_tuner\flux_2_train_network.py ^
  --dataset_config "%TOML%" ^
  --model_version %MODEL_VERSION% ^
  --dit "%DIT_CHECKPOINT%" ^
  --vae "%VAE_CHECKPOINT%" ^
  --text_encoder "%TEXT_ENCODER%" ^
  --vae_dtype bfloat16 ^
  --mixed_precision bf16 ^
  --timestep_sampling flux2_shift ^
  --weighting_scheme none ^
  --network_module networks.lora_flux_2 ^
  --network_dim %DIM% ^
  --network_alpha %ALPHA% ^
  --fp8_base ^
  --fp8_scaled ^
  --fp8_text_encoder ^
  --gradient_checkpointing ^
  --blocks_to_swap %BLOCKS% ^
  --sdpa ^
  --optimizer_type adafactor ^
  --optimizer_args relative_step=False scale_parameter=False warmup_init=False ^
  --learning_rate %LR% ^
  --lr_scheduler constant_with_warmup ^
  --lr_warmup_steps %WARMUP% ^
  --max_grad_norm 0.0 ^
  --gradient_accumulation_steps 1 ^
  --max_train_epochs %EPOCHS% ^
  --save_every_n_epochs %SAVE_EVERY% ^
  !SAMPLE_ARGS! ^
  !COMMENT_ARG! ^
  --output_dir "%OUTPUT_DIR%" ^
  --output_name "%OUTPUT_NAME%" ^
  --max_data_loader_n_workers 2 ^
  --persistent_data_loader_workers ^
  --seed 42

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
