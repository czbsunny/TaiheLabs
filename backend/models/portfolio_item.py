from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class PortfolioItem(Base):
    __tablename__ = "portfolio_items"
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol = Column(String(50), nullable=False)      # 基金或股票代码
    name = Column(String(100))                       # 基金名称
    quantity = Column(Float, nullable=False)         # 持有份额/股数
    cost = Column(Float, nullable=False)             # 单位成本价
    hold_amount = Column(Float, default=0)           # 持有金额
    hold_profit = Column(Float, default=0)           # 持有收益
    
    portfolio = relationship("Portfolio", back_populates="items")
