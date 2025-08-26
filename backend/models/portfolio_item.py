from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class PortfolioItem(Base):
    __tablename__ = "portfolio_items"
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol = Column(String(50), nullable=False)      # 基金或股票代码
    quantity = Column(Float, nullable=False)         # 持有份额/股数
    cost = Column(Float, nullable=False)             # 单位成本价
    
    portfolio = relationship("Portfolio", back_populates="items")
