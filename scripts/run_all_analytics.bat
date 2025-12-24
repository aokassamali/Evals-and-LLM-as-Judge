@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ============================
REM Config (edit if needed)
REM ============================

REM Gold file (your path)
set GOLD=data\scifact_oracle_dev_200.jsonl

REM Runs folder and pattern
set RUNS_DIR=runs
set RUN_GLOB=scifact_dev200_*_t0.0.jsonl

REM Reference run stem for disagreement slicing
REM If your GPT file is runs\scifact_dev200_gpt-5.2_t0.0.jsonl, this is correct:
set REF=scifact_dev200_gpt-5.2_t0.0

REM Outputs
set OUT_RELIABILITY=results\reliability_dev200.json
set OUT_DISAGREE_MD=results\disagreement_summary.md
set OUT_DISAGREE_SAMPLE=results\disagreements_sample.jsonl

echo === RUN ALL (Phase 4) ===
echo GOLD: %GOLD%
echo RUNS_DIR: %RUNS_DIR%
echo RUN_GLOB: %RUN_GLOB%
echo REF: %REF%
echo.

REM ============================
REM Sanity checks
REM ============================

if not exist "%GOLD%" (
  echo ERROR: Gold file not found: %GOLD%
  exit /b 1
)

if not exist "%RUNS_DIR%" (
  echo ERROR: Runs dir not found: %RUNS_DIR%
  exit /b 1
)

REM ============================
REM 1) Metrics summary
REM ============================

echo [1/3] Computing metrics tables...
python scripts\compute_multiple_model_metrics.py --gold "%GOLD%" --runs_dir "%RUNS_DIR%" --pattern "%RUN_GLOB%"
if errorlevel 1 goto :fail

REM ============================
REM 2) Reliability
REM ============================

echo.
echo [2/3] Computing reliability...
python scripts\compute_reliability.py --runs_dir "%RUNS_DIR%" --glob "%RUN_GLOB%" --out "%OUT_RELIABILITY%"
if errorlevel 1 goto :fail

REM ============================
REM 3) Disagreement slicing
REM ============================

echo.
echo [3/3] Disagreement slicing...
python scripts\analyze_disagreements.py --gold "%GOLD%" --runs_dir "%RUNS_DIR%" --glob "%RUN_GLOB%" --ref "%REF%" --out_md "%OUT_DISAGREE_MD%" --out_sample "%OUT_DISAGREE_SAMPLE%" --k_examples 5
if errorlevel 1 goto :fail

echo.
echo === DONE ===
echo Outputs:
echo   results\metrics_table.md
echo   results\metrics_table.csv
echo   results\metrics_json\*.json
echo   %OUT_RELIABILITY%
echo   %OUT_DISAGREE_MD%
echo   %OUT_DISAGREE_SAMPLE%
exit /b 0

:fail
echo.
echo === FAILED ===
echo One of the steps returned a non-zero exit code.
exit /b 1
