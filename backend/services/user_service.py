from sqlalchemy.orm import Session
from backend.models.user import User
from backend.models.portfolio import Portfolio
from backend.models.portfolio_detail import PortfolioDetail
from backend.utils.snowflake import snowflake

class UserService:
    def __init__(self, db: Session):
        self.db = db

    # 创建用户
    def create_user(self, username: str, email: str, password_hash: str, nickname: str = None):
        user = User(username=username, email=email, password_hash=password_hash, nickname=nickname)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    # 删除用户（软删除）
    def delete_user(self, user_id: int):
        user = self.db.query(User).filter(User.id == user_id, User.is_deleted == False).first()
        if user:
            user.is_deleted = True
            self.db.commit()
            return True
        return False

    # 查询用户信息
    def get_user(self, user_id: int):
        return self.db.query(User).filter(User.id == user_id, User.is_deleted == False).first()

    # 查询用户的所有组合及明细
    def get_user_portfolios(self, user_id: int):
        user = self.db.query(User).filter(User.id == user_id, User.is_deleted == False).first()
        if not user:
            return None

        portfolios = self.db.query(Portfolio).filter(
            Portfolio.user_id == user_id,
            Portfolio.is_deleted == False
        ).all()

        result = []
        for p in portfolios:
            details = self.db.query(PortfolioDetail).filter(
                PortfolioDetail.portfolio_id == p.id
            ).all()
            result.append({
                "portfolio": p,
                "details": details
            })

        return {
            "user": user,
            "portfolios": result
        }

    # 更新用户昵称
    def update_nickname(self, user_id: int, nickname: str):
        user = self.db.query(User).filter(User.id == user_id, User.is_deleted == False).first()
        if user:
            user.nickname = nickname
            self.db.commit()
            return user
        return None
