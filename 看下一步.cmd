@echo off
REM ============================================================
REM  Double-click this file when you don't know what to do next.
REM
REM  It just shows the status panel and waits, so the window
REM  stays open long enough to read.
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

python tools\status.py

echo.
pause
