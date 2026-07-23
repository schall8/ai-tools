@echo off
setlocal enabledelayedexpansion
title Musubi Z-Image Cache (generic / parameter-driven)

REM =====================================================================
REM  Generic Z-Image latent + text-encoder cache.
REM  Main image dataset (--dataset) plus an optional extra image slot
REM  (--extra-dir), e.g. a nudes/close-up set.
REM
REM  USAGE:
REM    zimage_cache.bat --name tammy --dataset "D:/DATA/.../tammy/images" ^
REM       [--repeats 3] [--extra-dir "D:/DATA/.../tammy/nude_test" --extra-repeats 2] ^
REM       [--res 1024x1024] [--dry-run]
REM
REM  Run this ONCE per subject before zimage_train.bat.
REM =====================================================================

REM ---- load machine paths from config.bat (run setup.bat to create it) ----
set "CONFIG=%~dp0..\config.bat"
if not exist "%CONFIG%" ( echo ERROR: config not found: %CONFIG% & echo Run setup.bat in the train_scripts folder once to create it. & exit /b 1 )
call "%CONFIG%"

REM ---- fixed paths / defaults ----
set "RENDER=%~dp0..\render_toml.py"
set "GEN_DIR=%~dp0_generated"
set "CACHE_ROOT=%MUSUBI_DIR:\=/%/cache"
set "VAE=%COMFY_MODELS%\vae\z_image_ae.safetensors"
set "TEXT_ENCODER=%COMFY_MODELS%\text_encoders\qwen_3_4b.safetensors"

set "NAME="
set "DATASET="
set "REPEATS=3"
set "RES=1024x1024"
set "EXTRA_DIR="
set "EXTRA_REPEATS=2"
set "DRYRUN=0"

REM ---- parse args ----
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--name"          ( set "NAME=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dataset"       ( set "DATASET=%~2" & shift & shift & goto parse )
if /i "%~1"=="--repeats"       ( set "REPEATS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--res"           ( set "RES=%~2" & shift & shift & goto parse )
if /i "%~1"=="--extra-dir"     ( set "EXTRA_DIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--extra-repeats" ( set "EXTRA_REPEATS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--vae"           ( set "VAE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--text-encoder"  ( set "TEXT_ENCODER=%~2" & shift & shift & goto parse )
if /i "%~1"=="--cache-root"    ( set "CACHE_ROOT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"       ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"==""    ( echo ERROR: --name is required & exit /b 1 )
if "%DATASET%"=="" ( echo ERROR: --dataset is required & exit /b 1 )

if not exist "%GEN_DIR%" mkdir "%GEN_DIR%"
set "TOML=%GEN_DIR%\%NAME%_zimage.toml"

echo.
echo Rendering dataset TOML -^> %TOML%
python "%RENDER%" --arch zimage --name "%NAME%" --out "%TOML%" --cache-root "%CACHE_ROOT%" --dataset "%DATASET%" --repeats %REPEATS% --res %RES% --extra-dir "%EXTRA_DIR%" --extra-repeats %EXTRA_REPEATS%
if errorlevel 1 ( echo ERROR: TOML render failed & exit /b 1 )

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] TOML rendered. Would cache with:
    echo   VAE:          %VAE%
    echo   TEXT_ENCODER: %TEXT_ENCODER%
    echo   dataset:      %DATASET%  ^(repeats=%REPEATS%, res=%RES%^)
    echo   extra:        %EXTRA_DIR%
    echo [DRY RUN] Not launching cache.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set CUDA_VISIBLE_DEVICES=0
set PYTHONIOENCODING=utf-8

echo.
echo Caching VAE latents...
python zimage_cache_latents.py --dataset_config "%TOML%" --vae "%VAE%"
if errorlevel 1 ( echo ERROR: latent caching failed & exit /b 1 )

echo.
echo Caching text encoder outputs (Qwen3-4B, fp8)...
python zimage_cache_text_encoder_outputs.py --dataset_config "%TOML%" --text_encoder "%TEXT_ENCODER%" --batch_size 8 --fp8_llm
if errorlevel 1 ( echo ERROR: text encoder caching failed & exit /b 1 )

echo.
echo Cache complete: %TOML%
endlocal
