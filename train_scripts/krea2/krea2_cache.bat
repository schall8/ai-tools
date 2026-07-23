@echo off
setlocal enabledelayedexpansion
title Musubi Krea2 Cache (generic / parameter-driven)

REM =====================================================================
REM  Generic Krea2 latent + text-encoder cache.
REM  Renders the dataset TOML from flags, then caches.
REM
REM  USAGE:
REM    krea2_cache.bat --name courtney --dataset "D:/DATA/.../courtney/img" ^
REM       [--repeats 2] [--res 256,384] [--dry-run]
REM
REM  Run this ONCE per subject before krea2_train.bat.
REM =====================================================================

REM ---- load machine paths from config.bat (run setup.bat to create it) ----
set "CONFIG=%~dp0..\config.bat"
if not exist "%CONFIG%" ( echo ERROR: config not found: %CONFIG% & echo Run setup.bat in the train_scripts folder once to create it. & exit /b 1 )
call "%CONFIG%"

REM ---- fixed paths / defaults (override with flags) ----
set "RENDER=%~dp0..\render_toml.py"
set "GEN_DIR=%~dp0_generated"
set "CACHE_ROOT=%MUSUBI_DIR:\=/%/cache"
set "VAE=%COMFY_MODELS%\vae\qwen_image_vae.safetensors"
set "TEXT_ENCODER=%COMFY_MODELS%\text_encoders\Qwen3-VL-4B-Instruct\model-00001-of-00002.safetensors"

set "NAME="
set "DATASET="
set "REPEATS=2"
set "RES=256x384"
set "DRYRUN=0"

REM ---- parse args ----
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--name"          ( set "NAME=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dataset"       ( set "DATASET=%~2" & shift & shift & goto parse )
if /i "%~1"=="--repeats"       ( set "REPEATS=%~2" & shift & shift & goto parse )
if /i "%~1"=="--res"           ( set "RES=%~2" & shift & shift & goto parse )
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
set "TOML=%GEN_DIR%\%NAME%_krea2.toml"

echo.
echo Rendering dataset TOML -^> %TOML%
python "%RENDER%" --arch krea2 --name "%NAME%" --out "%TOML%" --cache-root "%CACHE_ROOT%" --dataset "%DATASET%" --repeats %REPEATS% --res %RES%
if errorlevel 1 ( echo ERROR: TOML render failed & exit /b 1 )

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] TOML rendered. Would cache with:
    echo   VAE:          %VAE%
    echo   TEXT_ENCODER: %TEXT_ENCODER%
    echo   dataset:      %DATASET%  ^(repeats=%REPEATS%, res=%RES%^)
    echo [DRY RUN] Not launching cache.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set CUDA_VISIBLE_DEVICES=0
REM HF_HOME comes from config.bat

echo.
echo Caching VAE latents...
python src\musubi_tuner\krea2_cache_latents.py --dataset_config "%TOML%" --vae "%VAE%" --device cuda
if errorlevel 1 ( echo ERROR: latent caching failed & exit /b 1 )

echo.
echo Caching text encoder outputs (Qwen3-VL-4B)...
python src\musubi_tuner\krea2_cache_text_encoder_outputs.py --dataset_config "%TOML%" --text_encoder "%TEXT_ENCODER%" --batch_size 1 --device cuda
if errorlevel 1 ( echo ERROR: text encoder caching failed & exit /b 1 )

echo.
echo Cache complete: %TOML%
endlocal
