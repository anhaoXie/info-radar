@echo off
REM ============================================================
REM  info-radar entry point
REM
REM  ASCII only, on purpose: cmd.exe reads .cmd files using the OEM
REM  codepage, so non-ASCII comments get mangled and can break
REM  parsing. All Chinese output comes from Python instead, where
REM  encoding is one setting rather than a minefield.
REM
REM  This file must keep CRLF line endings. cmd.exe fails to resolve
REM  batch labels in an LF-only file ("cannot find the batch label").
REM
REM  Usage:
REM    run.cmd status
REM    run.cmd doctor
REM    run.cmd selftest
REM    run.cmd mocktest
REM    run.cmd probe
REM    run.cmd collect
REM    run.cmd inbox
REM    run.cmd sample -N 20
REM    run.cmd labels
REM    run.cmd compile
REM    run.cmd pipeline
REM    run.cmd digest
REM    run.cmd review --apply <stamp>
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
:collect_args
if "%~1"=="" goto args_ready
set "REST=%REST% %~1"
shift
goto collect_args
:args_ready

if /i "%CMD%"=="help"     goto :c_help
if /i "%CMD%"=="status"   goto :c_status
if /i "%CMD%"=="doctor"   goto :c_doctor
if /i "%CMD%"=="selftest" goto :c_selftest
if /i "%CMD%"=="mocktest" goto :c_mocktest
if /i "%CMD%"=="probe"    goto :c_probe
if /i "%CMD%"=="collect"  goto :c_collect
if /i "%CMD%"=="inbox"    goto :c_inbox
if /i "%CMD%"=="sample"   goto :c_sample
if /i "%CMD%"=="labels"   goto :c_labels
if /i "%CMD%"=="compile"  goto :c_compile
if /i "%CMD%"=="pipeline" goto :c_pipeline
if /i "%CMD%"=="digest"   goto :c_digest
if /i "%CMD%"=="review"   goto :c_review
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

:c_mocktest
python tools\mocktest.py
exit /b %ERRORLEVEL%

:c_probe
python tools\probe.py %REST%
exit /b %ERRORLEVEL%

:c_collect
python src\collect.py %REST%
exit /b %ERRORLEVEL%

:c_inbox
python src\collect.py --inbox
exit /b %ERRORLEVEL%

:c_sample
python tools\make_eval_set.py --sample %REST%
exit /b %ERRORLEVEL%

:c_labels
python tools\make_eval_set.py --status
exit /b %ERRORLEVEL%

:c_compile
python tools\make_eval_set.py --compile %REST%
exit /b %ERRORLEVEL%

:c_pipeline
python src\pipeline.py %REST%
exit /b %ERRORLEVEL%

:c_digest
python src\archive.py --digest %REST%
exit /b %ERRORLEVEL%

:c_review
if not "%REST%"=="" goto :c_review_args
python src\archive.py --list
exit /b %ERRORLEVEL%
:c_review_args
python src\archive.py %REST%
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
