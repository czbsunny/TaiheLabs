# backend/services/portfolio_service.py
from sqlalchemy.orm import Session
from models.portfolio import Portfolio
from models.portfolio_item import PortfolioItem
from utils.snowflake import snowflake

class PortfolioService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_portfolios(self, user_id: int):
        # 获取用户的所有组合
        portfolios = self.db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
        
        # 为每个组合加载持仓明细
        for portfolio in portfolios:
            # 使用关系属性获取持仓明细，避免额外的查询
            portfolio.items = self.db.query(PortfolioItem).filter(PortfolioItem.portfolio_id == portfolio.id).all()
            
        return portfolios
        
    # 创建组合
    def create_portfolio(self, user_id: int, name: str, description: str = None):
        portfolio = Portfolio(user_id=user_id, name=name, description=description)
        self.db.add(portfolio)
        self.db.commit()
        self.db.refresh(portfolio)
        return portfolio

    # 删除组合（软删除）
    def delete_portfolio(self, portfolio_id: int):
        portfolio = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if portfolio:
            self.db.delete(portfolio)
            self.db.commit()
            return True
        return False

    # 添加组合明细
    def add_item(self, portfolio_id: int, symbol: str, quantity: float, cost: float):
        item = PortfolioItem(portfolio_id=portfolio_id, symbol=symbol, quantity=quantity, cost=cost)
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
        portfolio = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            return None
        items = self.db.query(PortfolioItem).filter(PortfolioItem.portfolio_id == portfolio_id).all()
        return {
            "portfolio": portfolio,
            "items": items
        }

    # 更新组合明细
    def update_item(self, item_id: int, quantity: float = None, cost: float = None):
        item = self.db.query(PortfolioItem).filter(PortfolioItem.id == item_id).first()
        if item:
            if quantity is not None:
                item.quantity = quantity
            if cost is not None:
                item.cost = cost
            self.db.commit()
            self.db.refresh(item)
            return item
        return None
