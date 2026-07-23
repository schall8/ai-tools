@echo off
setlocal enabledelayedexpansion
title Musubi WAN 2.2 Cache (generic / parameter-driven)

REM =====================================================================
REM  Generic WAN 2.2 T2V latent + text-encoder cache (same VAE + T5 as 2.1).
REM  Three optional dataset slots -- pass a dir to enable each:
REM     --video-dir  (with --video-repeats / --video-frames)
REM     --image-dir  (with --image-repeats)
REM     --extra-dir  (with --extra-repeats)   e.g. a nudes/close-up set
REM  At least one slot is required. --skip_existing makes reruns cheap.
REM
REM  USAGE:
REM    wan22_cache.bat --name tammy ^
REM       --video-dir "D:/DATA/.../videos" --video-repeats 3 --video-frames "1,4" ^
REM       --image-dir "D:/DATA/.../images" --image-repeats 3 ^
REM       --extra-dir "D:/DATA/.../nude_test" --extra-repeats 4 ^
REM       [--res 480x640] [--dry-run]
REM
REM  Run this ONCE per subject before wan22_train.bat.
REM =====================================================================

REM ---- load machine paths from config.bat (run setup.bat to create it) ----
set "CONFIG=%~dp0..\config.bat"
if not exist "%CONFIG%" ( echo ERROR: config not found: %CONFIG% & echo Run setup.bat in the train_scripts folder once to create it. & exit /b 1 )
call "%CONFIG%"

REM ---- fixed paths / defaults ----
set "RENDER=%~dp0..\render_toml.py"
set "GEN_DIR=%~dp0_generated"
set "CACHE_ROOT=%MUSUBI_DIR:\=/%/cache"
set "VAE=%COMFY_MODELS%\vae\wan_2.1_vae.safetensors"
set "T5=%COMFY_MODELS%\clip\models_t5_umt5-xxl-enc-bf16.pth"

set "NAME="
set "RES=480x640"
set "VIDEO_DIR="
set "VIDEO_REPEATS=3"
set "VIDEO_FRAMES=1,4"
set "FRAME_EXTRACTION=uniform"
set "FRAME_SAMPLE=none"
set "IMAGE_DIR="
set "IMAGE_REPEATS=3"
set "EXTRA_DIR="
set "EXTRA_REPEATS=4"
set "DRYRUN=0"

REM ---- parse args ----
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--name"             ( set "NAME=%~2" & shift & shift & goto parse )
if /i "%~1"=="--res"              ( set "RES=%~2" & shift & shift & goto parse )
if /i "%~1"=="--video-dir"        ( set "VIDEO_DIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--video-repeats"    ( set "VIDEO_REPEATS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--video-frames"     ( set "VIDEO_FRAMES=%~2" & shift & shift & goto parse )
if /i "%~1"=="--frame-extraction" ( set "FRAME_EXTRACTION=%~2" & shift & shift & goto parse )
if /i "%~1"=="--frame-sample"     ( set "FRAME_SAMPLE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--image-dir"        ( set "IMAGE_DIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--image-repeats"    ( set "IMAGE_REPEATS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--extra-dir"        ( set "EXTRA_DIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--extra-repeats"    ( set "EXTRA_REPEATS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--vae"              ( set "VAE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--t5"               ( set "T5=%~2" & shift & shift & goto parse )
if /i "%~1"=="--cache-root"       ( set "CACHE_ROOT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"          ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"=="" ( echo ERROR: --name is required & exit /b 1 )
if "%VIDEO_DIR%%IMAGE_DIR%%EXTRA_DIR%"=="" ( echo ERROR: at least one of --video-dir / --image-dir / --extra-dir is required & exit /b 1 )

if not exist "%GEN_DIR%" mkdir "%GEN_DIR%"
set "TOML=%GEN_DIR%\%NAME%_wan22.toml"

echo.
echo Rendering dataset TOML -^> %TOML%
python "%RENDER%" --arch wan22 --name "%NAME%" --out "%TOML%" --cache-root "%CACHE_ROOT%" --res %RES% ^
  --video-dir "%VIDEO_DIR%" --video-repeats %VIDEO_REPEATS% --video-frames %VIDEO_FRAMES% --frame-extraction %FRAME_EXTRACTION% --frame-sample %FRAME_SAMPLE% ^
  --image-dir "%IMAGE_DIR%" --image-repeats %IMAGE_REPEATS% ^
  --extra-dir "%EXTRA_DIR%" --extra-repeats %EXTRA_REPEATS%
if errorlevel 1 ( echo ERROR: TOML render failed & exit /b 1 )

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] TOML rendered. Would cache with:
    echo   VAE: %VAE%
    echo   T5:  %T5%
    echo   video: %VIDEO_DIR%  image: %IMAGE_DIR%  extra: %EXTRA_DIR%
    echo [DRY RUN] Not launching cache.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set CUDA_VISIBLE_DEVICES=0

echo.
echo Caching VAE latents...
python wan_cache_latents.py --dataset_config "%TOML%" --device cuda --num_workers 4 --vae "%VAE%" --batch_size 4 --vae_cache_cpu --vae_dtype float16 --skip_existing
if errorlevel 1 ( echo ERROR: latent caching failed & exit /b 1 )

echo.
echo Caching text encoder outputs (T5, fp8)...
python wan_cache_text_encoder_outputs.py --dataset_config "%TOML%" --device cuda --num_workers 4 --t5 "%T5%" --batch_size 16 --fp8_t5 --skip_existing
if errorlevel 1 ( echo ERROR: text encoder caching failed & exit /b 1 )

echo.
echo Cache complete: %TOML%
endlocal
