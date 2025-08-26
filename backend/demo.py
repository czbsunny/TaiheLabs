from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.base import Base
from models.user import User

engine = create_engine("sqlite:///data/fund_db.sqlite3", echo=True)
Base.metadata.create_all(bind=engine)  # 初始化所有表
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# 创建用户
alice = User(username="alice", email="alice@example.com", password_hash="hashed123", nickname="小艾")
db.add(alice)
db.commit()
