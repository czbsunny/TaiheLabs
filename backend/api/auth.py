from fastapi import APIRouter, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from models.user import User
from database.init_db import get_db
from fastapi.responses import RedirectResponse

auth_router = APIRouter()


@auth_router.post("/register")
def register(username: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = User(username=username, email=email, password_hash=password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "username": user.username, "email": user.email}

@auth_router.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username, User.password_hash == password).first()
    if not user:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    else:
        response = RedirectResponse(url=f"/portfolios/{user.id}", status_code=303)
        return response

@auth_router.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username, User.password_hash == password, User.is_deleted == 0).first()
    if not user:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    return {"message": "登录成功", "user_id": user.id, "username": user.username}

