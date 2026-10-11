@echo off
REM ============================================================
REM  Double-click this file to start labeling.
REM
REM  Why this file exists:
REM    Telling someone "open cmd, cd to a folder, type a command"
REM    is three steps that have nothing to do with the actual task.
REM    Double-clicking is one step.
REM
REM  ASCII only, on purpose: cmd.exe reads .cmd files using the OEM
REM  codepage, so non-ASCII text here would be mangled. All Chinese
REM  output comes from Python instead.
REM
REM  Must keep CRLF line endings (cmd.exe batch label resolution).
REM ============================================================

setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul 2>nul

python tools\make_eval_set.py --wizard

echo.
pause
