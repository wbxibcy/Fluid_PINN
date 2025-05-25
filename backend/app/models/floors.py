from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from datetime import datetime

from app.core.database import Base

class Floor(Base):
    __tablename__ = "floors"

    floor_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)  # 关联用户表
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    geojson_id = Column(Integer, ForeignKey("geojson.geojson_id"), nullable=True)  # 关联 GeoJSON 表
    marker_id = Column(Integer, ForeignKey("markers.marker_id"), nullable=True)  # 新增字段，关联 Marker 表