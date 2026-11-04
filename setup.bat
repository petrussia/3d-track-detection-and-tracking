@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1

if exist ".venv\Scripts\python.exe" goto check_venv

py -3.12 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 goto create_with_py
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 goto create_with_python
python3 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 goto create_with_python3

echo Python 3.12 was not found. Install 64-bit Python 3.12 from python.org.
echo Enable the Python launcher or Add Python to PATH, then run setup.bat again.
exit /b 1

:create_with_py
py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
goto check_venv

:create_with_python
python -m venv .venv
if errorlevel 1 exit /b 1
goto check_venv

:create_with_python3
python3 -m venv .venv
if errorlevel 1 exit /b 1

:check_venv
".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))"
if errorlevel 1 goto wrong_venv
if not exist "models\yolo11s.pt" goto missing_weights
".venv\Scripts\python.exe" -c "import hashlib,pathlib,sys; p=pathlib.Path('models/yolo11s.pt'); ok=hashlib.sha256(p.read_bytes()).hexdigest() == '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'; print('Model checksum OK' if ok else 'Model checksum mismatch. Extract the archive again.'); sys.exit(0 if ok else 1)"
if errorlevel 1 exit /b 1

if exist ".venv\.peleng-ready" del ".venv\.peleng-ready"
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_failed
".venv\Scripts\python.exe" -c "import cv2,numpy,torch,ultralytics,fastapi,uvicorn; print('Dependencies OK')"
if errorlevel 1 goto install_failed
if not exist outputs mkdir outputs
if errorlevel 1 exit /b 1
>".venv\.peleng-ready" echo Python 3.12 dependencies installed
if errorlevel 1 exit /b 1
echo Ready. Run run.bat --check-data, then run.bat.
exit /b 0

:wrong_venv
echo The existing .venv is not a working Python 3.12 environment.
echo Rename the .venv folder and run setup.bat again.
exit /b 1

:missing_weights
echo Missing models\yolo11s.pt. Extract the complete project archive first.
exit /b 1

:install_failed
echo Setup failed. Check the error above and your internet connection.
echo No ready marker was created. Run setup.bat again after fixing the problem.
exit /b 1
