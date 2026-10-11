@echo off
REM ============================================================
REM  info-radar entry point
REM
REM  ASCII only, on purpose: cmd.exe reads .cmd files using the OEM
REM  codepage, so non-ASCII comments get mangled and can break
REM  parsing. All Chinese output comes from Python instead, where
REM  encoding is one setting rather than a minefield.
REM
REM  Usage:
REM    run.cmd status
REM    run.cmd doctor
REM    run.cmd selftest
REM    run.cmd probe
REM    run.cmd eval -Limit 3
REM    run.cmd ledger
REM ============================================================

setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul 2>nul

set "CMD=%~1"
if "%CMD%"=="" set "CMD=help"
shift

set "REST="
:collect
if "%~1"=="" goto collected
set "REST=%REST% %~1"
shift
goto collect
:collected

if /i "%CMD%"=="help"     goto :c_help
if /i "%CMD%"=="status"   goto :c_status
if /i "%CMD%"=="doctor"   goto :c_doctor
if /i "%CMD%"=="selftest" goto :c_selftest
if /i "%CMD%"=="probe"    goto :c_probe
if /i "%CMD%"=="eval"     goto :c_eval
if /i "%CMD%"=="sources"  goto :c_sources
if /i "%CMD%"=="ledger"   goto :c_ledger

echo.
echo   Unknown command: %CMD%
python tools\status.py --help
exit /b 2

:c_help
python tools\status.py --help
exit /b %ERRORLEVEL%

:c_status
python tools\status.py
exit /b %ERRORLEVEL%

:c_doctor
python tools\probe.py --doctor
exit /b %ERRORLEVEL%

:c_selftest
python tools\selftest.py
exit /b %ERRORLEVEL%

:c_probe
python tools\probe.py %REST%
exit /b %ERRORLEVEL%

:c_eval
python src\evaluate.py %REST%
exit /b %ERRORLEVEL%

:c_sources
python tools\probe.py --check-sources
exit /b %ERRORLEVEL%

:c_ledger
python tools\status.py --ledger
exit /b %ERRORLEVEL%
