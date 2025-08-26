from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from database.init_db import init_db

# 初始化数据库
init_db()

app = FastAPI(title="TaiheLabs API")

# 挂载前端静态文件
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# 导入 API 路由
from api import auth, portfolio, ocr

app.include_router(auth.auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(portfolio.portfolio_router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(ocr.ocr_router, prefix="/api/ocr", tags=["ocr"])
# app.include_router(user.router, prefix="/api/user", tags=["user"])

# 默认首页，返回登录页面
@app.get("/")
def root():
    return FileResponse("../frontend/index.html")

# 注册页路由
@app.get("/register")
def register_page():
    return FileResponse("../frontend/register.html")