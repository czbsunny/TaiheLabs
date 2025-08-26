# backend/models/fund.py
from datetime import datetime
from sqlalchemy import Column, String, DateTime
from models.base import Base

class Fund(Base):
    __tablename__ = 'funds'
    fund_code = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    fund_type = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Fund(fund_code={self.fund_code}, name={self.name})>"
