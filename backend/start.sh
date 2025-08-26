#!/bin/bash
# 首次运行前请先安装依赖，推荐使用清华源加速easyocr安装
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 启动FastAPI应用
source venv/bin/activate
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000