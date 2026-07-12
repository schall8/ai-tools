@echo off
setlocal enabledelayedexpansion
title Musubi LTX-2.3 Cache (generic / parameter-driven)

REM =====================================================================
REM  Generic LTX-2.3 latent + text-encoder cache.
REM  Three optional dataset slots -- pass a dir to enable each:
REM     --video-dir  (with --video-repeats / --video-frames)
REM     --image-dir  (with --image-repeats)
REM     --extra-dir  (with --extra-repeats)   e.g. a nudes/close-up set
REM  At least one slot is required.
REM
REM  USAGE:
REM    ltx_cache.bat --name tammy ^
REM       --video-dir "D:/DATA/.../videos" --video-repeats 4 --video-frames 1,9,17,25 ^
REM       --image-dir "D:/DATA/.../images" --image-repeats 2 ^
REM       --extra-dir "D:/DATA/.../nude_test" --extra-repeats 4 ^
REM       [--res 512,768] [--dry-run]
REM
REM  Run this ONCE per subject before ltx_train.bat.
REM =====================================================================

REM ---- fixed paths / defaults ----
set "MUSUBI_DIR=D:\github\musubi-tuner-ltx"
set "RENDER=%~dp0..\render_toml.py"
set "GEN_DIR=%~dp0_generated"
set "CACHE_ROOT=D:/github/musubi-tuner-ltx/cache"
set "LTX_CHECKPOINT=D:\comfyui\ComfyUI\models\diffusion_models\LTX23\ltx2310eros_v1.safetensors"
set "GEMMA_ROOT=D:\ai\models\LTX-2.3\gemma"

set "NAME="
set "RES=512x768"
set "VIDEO_DIR="
set "VIDEO_REPEATS=4"
set "VIDEO_FRAMES=1,9,17,25"
set "FRAME_EXTRACTION=uniform"
set "FRAME_SAMPLE=4"
set "IMAGE_DIR="
set "IMAGE_REPEATS=2"
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
if /i "%~1"=="--checkpoint"       ( set "LTX_CHECKPOINT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--gemma-root"       ( set "GEMMA_ROOT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--cache-root"       ( set "CACHE_ROOT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"          ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"=="" ( echo ERROR: --name is required & exit /b 1 )
if "%VIDEO_DIR%%IMAGE_DIR%%EXTRA_DIR%"=="" ( echo ERROR: at least one of --video-dir / --image-dir / --extra-dir is required & exit /b 1 )

if not exist "%GEN_DIR%" mkdir "%GEN_DIR%"
set "TOML=%GEN_DIR%\%NAME%_ltx.toml"

echo.
echo Rendering dataset TOML -^> %TOML%
python "%RENDER%" --arch ltx --name "%NAME%" --out "%TOML%" --cache-root "%CACHE_ROOT%" --res %RES% ^
  --video-dir "%VIDEO_DIR%" --video-repeats %VIDEO_REPEATS% --video-frames %VIDEO_FRAMES% --frame-extraction %FRAME_EXTRACTION% --frame-sample %FRAME_SAMPLE% ^
  --image-dir "%IMAGE_DIR%" --image-repeats %IMAGE_REPEATS% ^
  --extra-dir "%EXTRA_DIR%" --extra-repeats %EXTRA_REPEATS%
if errorlevel 1 ( echo ERROR: TOML render failed & exit /b 1 )

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] TOML rendered. Would cache with:
    echo   checkpoint: %LTX_CHECKPOINT%
    echo   gemma_root: %GEMMA_ROOT%
    echo   video: %VIDEO_DIR%  image: %IMAGE_DIR%  extra: %EXTRA_DIR%
    echo [DRY RUN] Not launching cache.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set CUDA_VISIBLE_DEVICES=0

echo.
echo Caching VAE latents...
python ltx2_cache_latents.py --dataset_config "%TOML%" --save_dataset_manifest dataset_manifest.json --ltx2_checkpoint "%LTX_CHECKPOINT%" --device cuda --vae_dtype bf16 --ltx2_mode v --vae_chunk_size 16 --vae_spatial_tile_size 256 --vae_spatial_tile_overlap 64
if errorlevel 1 ( echo ERROR: latent caching failed & exit /b 1 )

echo.
echo Caching text encoder outputs (Gemma)...
python ltx2_cache_text_encoder_outputs.py --dataset_config "%TOML%" --ltx2_checkpoint "%LTX_CHECKPOINT%" --gemma_root "%GEMMA_ROOT%" --gemma_load_in_4bit --gemma_bnb_4bit_quant_type nf4 --gemma_bnb_4bit_compute_dtype bf16 --device cuda --mixed_precision bf16 --ltx2_mode v --batch_size 1
if errorlevel 1 ( echo ERROR: text encoder caching failed & exit /b 1 )

echo.
echo Cache complete: %TOML%
endlocal
