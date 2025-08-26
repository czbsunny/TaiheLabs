from sqlalchemy.orm import Session
from database.init_db import get_db, engine
from models.user import User
from datetime import datetime

# 测试数据库连接
try:
    # 获取数据库会话
    db_gen = get_db()
    db = next(db_gen)
    print("数据库连接成功!")
    
    # 测试插入用户
    try:
        # 创建一个测试用户（使用唯一的用户名和邮箱）
        test_username = f"test_user_{int(datetime.now().timestamp())}"
        test_email = f"{test_username}@example.com"
        
        # 检查用户是否存在
        existing_user = db.query(User).filter(User.username == test_username).first()
        if not existing_user:
            # 创建新用户
            new_user = User(
                username=test_username,
                email=test_email,
                password_hash="test_password"
            )
            db.add(new_user)
            db.commit()
            print(f"成功创建测试用户: {test_username}")
        else:
            print(f"用户已存在: {test_username}")
            
        # 查询所有用户
        users = db.query(User).all()
        print(f"数据库中共有 {len(users)} 个用户")
        
    except Exception as e:
        print(f"插入用户失败: {str(e)}")
        db.rollback()
    
finally:
    # 关闭数据库连接
    try:
        db.close()
    except:
        pass
    
    # 关闭生成器
    try:
        next(db_gen)
    except StopIteration:
        pass