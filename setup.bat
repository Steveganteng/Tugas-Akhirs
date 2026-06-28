@echo off
REM Setup otomatis Windows: klik-ganda atau jalankan dari Command Prompt.
REM Opsi reset: setup.bat --reset
cd /d "%~dp0"
python setup.py %*
echo.
pause
