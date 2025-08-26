# backend/services/portfolio_service.py
from sqlalchemy.orm import Session
from models.portfolio import Portfolio
from models.portfolio_item import PortfolioItem
from utils.snowflake import snowflake

class PortfolioService:
    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def get_user_portfolios(db: Session, user_id: int):
        return db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
        
    # 创建组合
    def create_portfolio(self, user_id: int, name: str, risk_level: str = None):
        portfolio = Portfolio(user_id=user_id, name=name, risk_level=risk_level)
        self.db.add(portfolio)
        self.db.commit()
        self.db.refresh(portfolio)
        return portfolio

    # 删除组合（软删除）
    def delete_portfolio(self, portfolio_id: int):
        portfolio = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.is_deleted == False).first()
        if portfolio:
            portfolio.is_deleted = True
            self.db.commit()
            return True
        return False

    # 添加组合明细
    def add_item(self, portfolio_id: int, fund_code: str, weight: float):
        item = Portfolioitem(portfolio_id=portfolio_id, fund_code=fund_code, weight=weight)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item

    # 删除组合明细
    def delete_item(self, item_id: int):
        item = self.db.query(PortfolioItem).filter(PortfolioItem.id == item_id).first()
        if item:
            self.db.delete(item)
            self.db.commit()
            return True
        return False

    # 获取组合及明细
    def get_portfolio_items(self, portfolio_id: int):
        portfolio = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.is_deleted == False).first()
        if not portfolio:
            return None
        items = self.db.query(PortfolioItem).filter(PortfolioItem.portfolio_id == portfolio_id).all()
        return {
            "portfolio": portfolio,
            "items": items
        }

    # 更新组合明细权重
    def update_item_weight(self, item_id: int, new_weight: float):
        item = self.db.query(PortfolioItem).filter(PortfolioItem.id == item_id).first()
        if item:
            item.weight = new_weight
            self.db.commit()
            return item
        return None
