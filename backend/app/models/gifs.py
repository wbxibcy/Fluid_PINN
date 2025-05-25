from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from datetime import datetime, timezone

from app.core.database import Base

class GIF(Base):
    __tablename__ = "gifs"

    gif_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    gif_url = Column(String, nullable=False)