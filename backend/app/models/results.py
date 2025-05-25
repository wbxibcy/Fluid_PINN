from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime
from datetime import datetime, timezone

from app.core.database import Base

class Result(Base):
    __tablename__ = "results"

    result_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    floor_id = Column(Integer, ForeignKey("floors.floor_id"), nullable=False)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    simulation_type = Column(String, nullable=False)
    gif_id = Column(Integer, ForeignKey("gifs.gif_id"), nullable=False)
    description = Column(String, nullable=False)