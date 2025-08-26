# backend/models/fund_nav_realtime.py
from datetime import datetime
from sqlalchemy import Column, BigInteger, String, Float, DateTime, func
from models.base import Base
from utils.snowflake import snowflake


class FundNavRealtime(Base):
    __tablename__ = 'fund_nav_realtime'
    id = Column(BigInteger, primary_key=True, default=lambda: snowflake.get_id())
    fund_code = Column(String(20), nullable=False)
    nav = Column(Float, nullable=False)
    datetime = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
