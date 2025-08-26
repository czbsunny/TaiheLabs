# backend/models/fund_nav_history.py
from datetime import datetime
from sqlalchemy import Column, BigInteger, String, Float, Date, DateTime, func
from models.base import Base
from utils.snowflake import snowflake

class FundNavHistory(Base):
    __tablename__ = 'fund_nav_history'
    id = Column(BigInteger, primary_key=True, default=lambda: snowflake.get_id())
    fund_code = Column(String(20), nullable=False)
    nav = Column(Float, nullable=False)
    date = Column(Date, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
