@echo off
start cmd /k "cd backend && venv\Scripts\activate && python -m uvicorn main:app --reload --port 8000"
start cmd /k "cd frontend && npm run dev"