from sqlalchemy import Column, Integer, JSON, DateTime
from datetime import datetime

from app.core.database import Base

class GeoJSON(Base):
    __tablename__ = "geojson"

    geojson_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    geojson_data = Column(JSON, nullable=False)