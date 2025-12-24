@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ============================
REM Config
REM ============================

set INPUT=data\scifact_oracle_dev_200.jsonl
set OUT_DIR=runs

REM Local models (edit to match your Ollama names)
set LOCAL_MODELS=llama3.1:8b qwen3:8b deepseek-r1:8b gemma3:4b

REM OpenAI model (optional)
set OPENAI_MODEL=gpt-5.2

REM Temperature
set TEMP=0.0

echo === RUN ALL EVALS ===
echo INPUT: %INPUT%
echo OUT_DIR: %OUT_DIR%
echo LOCAL_MODELS: %LOCAL_MODELS%
echo TEMP: %TEMP%
echo.

if not exist "%INPUT%" (
  echo ERROR: Input file not found: %INPUT%
  exit /b 1
)

REM ============================
REM 1) Local evals via run_multi
REM ============================

echo [1/2] Running local models...
python scripts\eval_many_models.py --models %LOCAL_MODELS% --input "%INPUT%" --out_dir "%OUT_DIR%" --temperature %TEMP%
if errorlevel 1 goto :fail

REM ============================
REM 2) OpenAI eval (optional)
REM Uncomment if you want to run GPT-5.2
REM ============================

REM echo.
REM echo [2/2] Running OpenAI model...
REM python scripts\eval_one_model.py --backend openai --model %OPENAI_MODEL% --input "%INPUT%" --output "%OUT_DIR%\scifact_dev200_%OPENAI_MODEL%_t%TEMP%.jsonl" --temperature %TEMP%
REM if errorlevel 1 goto :fail

echo.
echo === DONE (evals) ===
echo Check runs\ for new JSONL outputs.
exit /b 0

:fail
echo.
echo === FAILED ===
exit /b 1
