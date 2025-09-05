# backend/services/portfolio_service.py
import re
from sqlalchemy.orm import Session
from models.portfolio import Portfolio
from models.portfolio_item import PortfolioItem
from models.fund_nav_realtime import FundNavRealtime
from utils.snowflake import snowflake

class PortfolioService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_portfolios(self, user_id: int):
        # 获取用户的所有组合
        portfolios = self.db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
        
        # 为每个组合加载持仓明细和计算总值、总收益
        for portfolio in portfolios:
            # 使用关系属性获取持仓明细
            portfolio.items = self.db.query(PortfolioItem).filter(PortfolioItem.portfolio_id == portfolio.id).all()
            # 为了兼容前端模板，同时设置holdings属性
            portfolio.holdings = portfolio.items
            
            # 计算组合总值、总收益和日收益
            total_value = 0
            total_profit = 0
            daily_profit = 0
            
            for item in portfolio.items:
                # 尝试从FundNavRealtime表获取最新净值
                # 提取基金代码（假设symbol中包含6位数字的基金代码）
                fund_code_match = re.search(r'[0-9]{6}', item.symbol)
                fund_code = fund_code_match.group(0) if fund_code_match else None
                
                # 尝试获取实时净值
                current_nav = None
                if fund_code:
                    # 查询最新的实时净值
                    latest_nav = self.db.query(FundNavRealtime).filter(
                        FundNavRealtime.fund_code == fund_code
                    ).order_by(FundNavRealtime.datetime.desc()).first()
                    
                    if latest_nav:
                        current_nav = latest_nav.nav
                
                # 如果有实时净值，使用实时净值计算；否则使用成本价作为临时解决方案
                if current_nav:
                    item_value = item.quantity * current_nav
                    item_profit = item.quantity * (current_nav - item.cost)
                    # 对于日收益，暂时使用与总收益相同的值作为简化实现
                    # 在实际应用中，应该根据今日净值与昨日净值的差值计算
                    item_daily_profit = item_profit
                    daily_profit += item_daily_profit
                else:
                    item_value = item.quantity * item.cost
                    # 使用数据库中存储的持有收益（如果有）
                    item_profit = getattr(item, 'hold_profit', 0) or 0
                    item_daily_profit = 0
                
                # 为前端模板添加所需的属性，使其与模型字段匹配
                item.fund_name = item.name
                item.fund_code = item.symbol
                item.shares = item.quantity
                item.purchase_price = item.cost
                item.current_price = current_nav if current_nav else item.cost
                item.total_profit = item_profit
                item.daily_profit = item_daily_profit
                
                total_value += item_value
                total_profit += item_profit
            
            # 添加总值、总收益和日收益属性到组合对象
            portfolio.total_value = total_value
            portfolio.total_profit = total_profit
            portfolio.daily_profit = daily_profit
        
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
    def add_item(self, portfolio_id: int, symbol: str, quantity: float, cost: float, name: str = None, hold_amount: float = None, hold_profit: float = None):
        print(f"Adding new portfolio item: portfolio_id={portfolio_id}, symbol={symbol}, quantity={quantity}, cost={cost}, name={name}")
        item = PortfolioItem(
            portfolio_id=portfolio_id,
            symbol=symbol,
            name=name,
            quantity=quantity,
            cost=cost,
            hold_amount=hold_amount or 0,
            hold_profit=hold_profit or 0
        )
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        print(f"Successfully added portfolio item: id={item.id}, symbol={item.symbol}")
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
