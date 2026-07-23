@echo off
setlocal enabledelayedexpansion
title Musubi WAN 2.2 LoRA Training (generic / parameter-driven)

REM =====================================================================
REM  Generic WAN 2.2 T2V LoRA training (dual high+low noise DiT).
REM  Consumes the TOML produced by wan22_cache.bat (matched by --name),
REM  so run cache first.
REM
REM  NOTE: WAN (like LTX) has no in-training sampling, so there is no
REM  --samples toggle here.
REM
REM  USAGE:
REM    wan22_train.bat --name tammy [options]
REM
REM  Common options (defaults match the tammy template):
REM    --epochs 16   --dim 16   --alpha 16   --lr 1e-4   --blocks-to-swap 30
REM    --grad-accum 2   --warmup-steps 150   --save-every 1   --task t2v-A14B
REM    --output-name <name>_wan22   --output-root D:\DATA\training\wan_loras
REM    --trigger <word>      stamp into output LoRA metadata (comma-sep for many)
REM =====================================================================

REM ---- load machine paths from config.bat (run setup.bat to create it) ----
set "CONFIG=%~dp0..\config.bat"
if not exist "%CONFIG%" ( echo ERROR: config not found: %CONFIG% & echo Run setup.bat in the train_scripts folder once to create it. & exit /b 1 )
call "%CONFIG%"

REM ---- fixed paths / defaults ----
set "GEN_DIR=%~dp0_generated"
set "DIT_LOW=%COMFY_MODELS%\diffusion_models\wan2.2_t2v_low_noise_14B_fp16.safetensors"
set "DIT_HIGH=%COMFY_MODELS%\diffusion_models\wan2.2_t2v_high_noise_14B_fp16.safetensors"
set "VAE=%COMFY_MODELS%\vae\wan_2.1_vae.safetensors"
set "T5=%COMFY_MODELS%\clip\models_t5_umt5-xxl-enc-bf16.pth"
set "LOGDIR=%MUSUBI_DIR%\logs"

set "NAME="
set "OUTPUT_ROOT=%TRAINING_ROOT%\wan_loras"
set "OUTPUT_NAME="
set "EPOCHS=16"
set "DIM=16"
set "ALPHA=16"
set "LR=1e-4"
set "BLOCKS=30"
set "GRAD_ACCUM=2"
set "WARMUP=150"
set "SAVE_EVERY=1"
set "TASK=t2v-A14B"
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
if /i "%~1"=="--grad-accum"     ( set "GRAD_ACCUM=%~2" & shift & shift & goto parse )
if /i "%~1"=="--warmup-steps"   ( set "WARMUP=%~2" & shift & shift & goto parse )
if /i "%~1"=="--save-every"     ( set "SAVE_EVERY=%~2" & shift & shift & goto parse )
if /i "%~1"=="--task"           ( set "TASK=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dit-low"        ( set "DIT_LOW=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dit-high"       ( set "DIT_HIGH=%~2" & shift & shift & goto parse )
if /i "%~1"=="--vae"            ( set "VAE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--t5"             ( set "T5=%~2" & shift & shift & goto parse )
if /i "%~1"=="--logdir"         ( set "LOGDIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--trigger"        ( set "TRIGGER=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"        ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"=="" ( echo ERROR: --name is required & exit /b 1 )
if "%OUTPUT_NAME%"=="" set "OUTPUT_NAME=%NAME%_wan22"
set "OUTPUT_DIR=%OUTPUT_ROOT%\%OUTPUT_NAME%"
set "TOML=%GEN_DIR%\%NAME%_wan22.toml"

if not exist "%TOML%" (
    echo ERROR: dataset TOML not found: %TOML%
    echo Run wan22_cache.bat --name %NAME% ... first.
    exit /b 1
)

echo.
echo WAN 2.2 T2V LoRA training - %NAME%  (dual high+low noise)
echo   Config:   %TOML%
echo   Output:   %OUTPUT_DIR%\%OUTPUT_NAME%
echo   Task:     %TASK%
echo   Epochs:   %EPOCHS%   dim/alpha: %DIM%/%ALPHA%   lr: %LR%   blocks_to_swap: %BLOCKS%
echo   (no in-training sampling for WAN)

if "%DRYRUN%"=="1" (
    echo.
    echo   DIT low:  %DIT_LOW%
    echo   DIT high: %DIT_HIGH%
    echo [DRY RUN] Not launching training.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_ALLOC_CONF=expandable_segments:True

if not "!TRIGGER!"=="" ( set "COMMENT_ARG=--training_comment "!TRIGGER!"" ) else ( set "COMMENT_ARG=" )

accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 "wan_train_network.py" ^
  --cuda_allow_tf32 ^
  --dataset_config "%TOML%" ^
  --discrete_flow_shift 3 ^
  --task %TASK% ^
  --dit "%DIT_LOW%" ^
  --dit_high_noise "%DIT_HIGH%" ^
  --fp8_base ^
  --gradient_accumulation_steps %GRAD_ACCUM% ^
  --blocks_to_swap %BLOCKS% ^
  --gradient_checkpointing ^
  --learning_rate %LR% ^
  --log_with tensorboard ^
  --logging_dir "%LOGDIR%" ^
  --lr_scheduler cosine ^
  --lr_warmup_steps %WARMUP% ^
  --max_data_loader_n_workers 8 ^
  --max_train_epochs %EPOCHS% ^
  --mixed_precision fp16 ^
  --network_dim %DIM% ^
  --network_alpha %ALPHA% ^
  --network_module networks.lora_wan ^
  --optimizer_type AdamW ^
  --output_dir "%OUTPUT_DIR%" ^
  --output_name "%OUTPUT_NAME%" ^
  !COMMENT_ARG! ^
  --persistent_data_loader_workers ^
  --save_every_n_epochs %SAVE_EVERY% ^
  --save_state ^
  --seed 42 ^
  --t5 "%T5%" ^
  --fp8_t5 ^
  --timestep_sampling sigmoid ^
  --vae "%VAE%" ^
  --vae_cache_cpu ^
  --vae_dtype float16 ^
  --sdpa

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
