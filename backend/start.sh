#!/bin/bash
# 启动FastAPI应用
source venv/bin/activate
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000