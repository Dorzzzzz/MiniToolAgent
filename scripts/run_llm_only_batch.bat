@echo off
setlocal

rem Serially run direct LLM-only baselines on Windows.
rem Edit the variables below to switch model config, log location, or datasets.

set "CONFIG=llm.yaml"
set "LOG_ROOT=qwen2.5_72b_logs"
set "RUN_NAME=llm_only_batch"
set "MAX_TOKENS=4096"
set "TEMPERATURE=0.0"
set "LIMIT="
set "ITEM_RETRIES=2"
set "RETRY_SLEEP=30"
set "RESUME_ARGS="
rem To force a full rerun, change the line above to: set "RESUME_ARGS=--no-resume"
set "DATASETS=aime25 hmmt"

pushd "%~dp0\.."

for %%D in (%DATASETS%) do (
    echo ============================================================
    echo Running LLM-only baseline on %%D
    echo Config:   %CONFIG%
    echo Log root: %LOG_ROOT%
    echo Run name: %RUN_NAME%
    echo ============================================================

    if "%LIMIT%"=="" (
        python .\scripts\run_llm_only_eval.py --dataset %%D --config "%CONFIG%" --log-root "%LOG_ROOT%" --run-name "%RUN_NAME%" --max-tokens %MAX_TOKENS% --temperature %TEMPERATURE% --item-retries %ITEM_RETRIES% --retry-sleep %RETRY_SLEEP% %RESUME_ARGS%
    ) else (
        python .\scripts\run_llm_only_eval.py --dataset %%D --config "%CONFIG%" --log-root "%LOG_ROOT%" --run-name "%RUN_NAME%" --max-tokens %MAX_TOKENS% --temperature %TEMPERATURE% --limit %LIMIT% --item-retries %ITEM_RETRIES% --retry-sleep %RETRY_SLEEP% %RESUME_ARGS%
    )

    if errorlevel 1 (
        echo.
        echo Failed while running %%D. Stopping batch.
        popd
        exit /b 1
    )
)

echo.
echo All LLM-only datasets finished.
echo Logs saved under: %LOG_ROOT%\%RUN_NAME%

popd
endlocal
