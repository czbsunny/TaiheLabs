from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.base import Base

# MySQL数据库连接配置
# 格式: mysql+pymysql://username:password@host:port/database
# 注意：需要安装pymysql包: pip install pymysql
DATABASE_URL = "mysql+pymysql://root:Sunny520@localhost:3306/fund_db?charset=utf8mb4"

# 创建数据库引擎，MySQL不需要check_same_thread参数
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()