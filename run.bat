@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" goto prepare
if not exist ".venv\.peleng-ready" goto prepare
goto launch

:prepare
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1

:launch
if "%~1"=="" goto launch_default
".venv\Scripts\python.exe" -m peleng %*
exit /b %ERRORLEVEL%

:launch_default
".venv\Scripts\python.exe" -m peleng --batch-ui --device cpu
exit /b %ERRORLEVEL%
