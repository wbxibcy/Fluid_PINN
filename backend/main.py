import uvicorn
import argparse
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import auth, users, gifs, results, markers, geojson, floors, admin
from app.core.database import engine, Base, async_session
from scripts.db import insert_test_data  
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

def parse_args():
    parser = argparse.ArgumentParser(description="Start FastAPI application")
    parser.add_argument("--init-db", action="store_true", help="Initialize the database")
    return parser.parse_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    args = parse_args()
    if args.init_db:
        await initialize_database()
    yield

async def initialize_database():
    """
    删除数据库中的所有表，重新创建表并插入测试数据。
    """
    async with engine.begin() as conn:
        # 删除所有表
        print("Dropping all tables...")
        await conn.run_sync(Base.metadata.drop_all)

        # 创建所有表
        print("Creating all tables...")
        await conn.run_sync(Base.metadata.create_all)

    # 插入测试数据
    async with async_session() as session:
        await insert_test_data(session)

def register_routers(app: FastAPI):
    """注册所有路由模块"""
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(users.router, prefix="/users", tags=["Users"])
    app.include_router(gifs.router, prefix="/gifs", tags=["GIFs"])
    app.include_router(results.router, prefix="/results", tags=["Results"])
    app.include_router(markers.router, prefix="/markers", tags=["Markers"])
    app.include_router(geojson.router, prefix="/geojson", tags=["GeoJSON"])
    app.include_router(floors.router, prefix="/floors", tags=["Floors"])
    app.include_router(admin.router, prefix="/admin", tags=["Admin"])

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="./outputs"), name="static")

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头
)

register_routers(app)

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000)