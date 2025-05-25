from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from app.core.config import config

# 创建数据库引擎
SQLALCHEMY_DATABASE_URL = config.DATABASE_URL
engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# 用于定义 SQLAlchemy 模型的基类
Base = declarative_base()

# 获取数据库会话
async def get_db():
    async with async_session() as session:
        yield session

# Redis 配置
redis_client = redis.StrictRedis.from_url(
    config.REDIS_URL,
    decode_responses=True
)