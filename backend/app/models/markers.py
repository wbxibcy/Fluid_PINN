from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from datetime import datetime, timezone

from app.core.database import Base

class Marker(Base):
    __tablename__ = "markers"

    marker_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)  # 关联用户表
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    description = Column(String, nullable=True)