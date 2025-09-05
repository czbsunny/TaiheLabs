# backend/models/fund_nav_history.py
from datetime import datetime
from sqlalchemy import Column, BigInteger, String, Float, Date, DateTime, func
from models.base import Base
from utils.snowflake import snowflake

class FundNavHistory(Base):
    __tablename__ = 'fund_nav_history'
    id = Column(BigInteger, primary_key=True, default=lambda: snowflake.get_id())
    fund_code = Column(String(20), nullable=False, index=True)
    nav = Column(Float, nullable=True)  # 单位净值
    accumulated_nav = Column(Float, nullable=True)  # 累计净值
    daily_growth_rate = Column(Float, nullable=True)  # 日增长率
    accumulated_return_rate = Column(Float, nullable=True)  # 累计收益率
    date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
