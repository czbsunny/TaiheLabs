from fastapi import APIRouter, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from models.user import User
from database.init_db import get_db
from fastapi.responses import RedirectResponse

auth_router = APIRouter()


@auth_router.post("/register")
def register(username: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # 记录接收到的参数
    print(f"接收到注册请求: username={username}, email={email}")
    
    # 检查用户名是否已存在
    if db.query(User).filter(User.username == username).first():
        print("注册失败: 用户名已存在")
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 检查邮箱是否已存在
    if db.query(User).filter(User.email == email).first():
        print("注册失败: 邮箱已被注册")
        raise HTTPException(status_code=400, detail="邮箱已被注册")
    
    try:
        # 创建新用户（注意：实际应用中应使用密码哈希）
        user = User(username=username, email=email, password_hash=password)
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"注册成功: user_id={user.id}")
        return {"id": user.id, "username": user.username, "email": user.email}
    except Exception as e:
        db.rollback()
        print(f"注册过程发生错误: {str(e)}")
        raise HTTPException(status_code=400, detail=f"注册失败: {str(e)}")

@auth_router.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username, User.password_hash == password, User.is_deleted == 0).first()
    if not user:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    return {"message": "登录成功", "user_id": user.id, "username": user.username}

@auth_router.get("/logout")
def logout():
    # 在实际应用中，这里应该清除用户的会话信息
    # 由于当前是模拟环境，直接返回成功消息并跳转到登录页面
    print("用户退出登录")
    # 重定向到根路径，系统会自动处理到登录页面的跳转
    return RedirectResponse(url="/", status_code=302)

