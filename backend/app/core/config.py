from pydantic_settings import BaseSettings
from typing import Optional

class Config(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    class Config:
        env_file = "db.env"
        env_file_encoding = 'utf-8'

# 创建配置实例
config = Config()
