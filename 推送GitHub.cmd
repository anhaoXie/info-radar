@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
python "tools\push_github.py"
set RC=%ERRORLEVEL%
echo.
pause
exit /b %RC%
