@echo off
setlocal enabledelayedexpansion
title Musubi FLUX.2 Klein 9B Cache (generic / parameter-driven)

REM =====================================================================
REM  Generic FLUX.2 Klein latent + text-encoder cache.
REM  Renders the dataset TOML from flags, then caches.
REM
REM  USAGE:
REM    klein_cache.bat --name courtney --dataset "D:/DATA/.../courtney/img" ^
REM       [--repeats 3] [--res 512,768] [--dry-run]
REM
REM  Run this ONCE per subject before klein_train.bat.
REM =====================================================================

REM ---- fixed paths / defaults ----
set "MUSUBI_DIR=D:\github\musubi-tuner"
set "RENDER=%~dp0..\render_toml.py"
set "GEN_DIR=%~dp0_generated"
set "CACHE_ROOT=D:/github/musubi-tuner/cache"
set "VAE=D:\ai\models\FLUX2\ae.safetensors"
set "TEXT_ENCODER=D:\ai\models\FLUX2\text_encoder\model-00001-of-00004.safetensors"
set "MODEL_VERSION=klein-base-9b"

set "NAME="
set "DATASET="
set "REPEATS=3"
set "RES=512x768"
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
if /i "%~1"=="--model-version" ( set "MODEL_VERSION=%~2" & shift & shift & goto parse )
if /i "%~1"=="--cache-root"    ( set "CACHE_ROOT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--dry-run"       ( set "DRYRUN=1" & shift & goto parse )
echo ERROR: unknown argument: %~1
exit /b 1
:parsed

if "%NAME%"==""    ( echo ERROR: --name is required & exit /b 1 )
if "%DATASET%"=="" ( echo ERROR: --dataset is required & exit /b 1 )

if not exist "%GEN_DIR%" mkdir "%GEN_DIR%"
set "TOML=%GEN_DIR%\%NAME%_klein.toml"

echo.
echo Rendering dataset TOML -^> %TOML%
python "%RENDER%" --arch klein --name "%NAME%" --out "%TOML%" --cache-root "%CACHE_ROOT%" --dataset "%DATASET%" --repeats %REPEATS% --res %RES%
if errorlevel 1 ( echo ERROR: TOML render failed & exit /b 1 )

if "%DRYRUN%"=="1" (
    echo.
    echo [DRY RUN] TOML rendered. Would cache with:
    echo   VAE:           %VAE%
    echo   TEXT_ENCODER:  %TEXT_ENCODER%
    echo   model_version: %MODEL_VERSION%
    echo   dataset:       %DATASET%  ^(repeats=%REPEATS%, res=%RES%^)
    echo [DRY RUN] Not launching cache.
    endlocal & exit /b 0
)

cd /d "%MUSUBI_DIR%"
set CUDA_VISIBLE_DEVICES=0

echo.
echo Caching VAE latents...
python src\musubi_tuner\flux_2_cache_latents.py --dataset_config "%TOML%" --vae "%VAE%" --model_version %MODEL_VERSION% --vae_dtype bfloat16 --device cuda
if errorlevel 1 ( echo ERROR: latent caching failed & exit /b 1 )

echo.
echo Caching text encoder outputs...
python src\musubi_tuner\flux_2_cache_text_encoder_outputs.py --dataset_config "%TOML%" --text_encoder "%TEXT_ENCODER%" --model_version %MODEL_VERSION% --fp8_text_encoder --device cuda --batch_size 1
if errorlevel 1 ( echo ERROR: text encoder caching failed & exit /b 1 )

echo.
echo Cache complete: %TOML%
endlocal
