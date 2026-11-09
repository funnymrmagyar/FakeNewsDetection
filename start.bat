@echo off
REM Backend (FastAPI, :8000) es frontend (Vite, :5173) inditasa a projekt gyokereben levo venv-vel.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [HIBA] Nincs venv a projekt gyokereben. Hozd letre:
    echo     py -m venv venv
    echo     venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

REM Hianyzo backend-csomagok telepitese (csak elso inditaskor tart tovabb)
venv\Scripts\python -c "import fastapi, uvicorn, multipart, timm, transformers, pandas" 2>nul
if errorlevel 1 (
    echo Hianyzo csomagok telepitese...
    venv\Scripts\python -m pip install -r requirements.txt
)

if not exist "frontend\node_modules" (
    echo Frontend fuggosegek telepitese...
    pushd frontend
    call npm install
    popd
)

start "Backend :8000" cmd /k "cd /d "%~dp0backend" && "%~dp0venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8000"
start "Frontend :5173" cmd /k "cd /d "%~dp0frontend" && npm run dev"
