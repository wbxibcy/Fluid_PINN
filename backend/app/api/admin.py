from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from app.models.user import User
from app.models.markers import Marker
from app.models.floors import Floor
from app.models.geojson import GeoJSON
from app.models.results import Result
from app.models.gifs import GIF
from app.core.database import get_db
from app.api.auth import get_current_user

router = APIRouter()

# 获取所有信息（管理员权限）
@router.get("/data")
async def get_all_data(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    管理员权限下，获取所有 marker、floor、geojson、result 和 gif 数据的巨型 JSON。
    """
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    users = await db.execute(select(User))
    markers = await db.execute(select(Marker))
    floors = await db.execute(select(Floor))
    geojsons = await db.execute(select(GeoJSON))
    results = await db.execute(select(Result))
    gifs = await db.execute(select(GIF))
    
    return {
        "users": users.scalars().all(),
        "markers": markers.scalars().all(),
        "floors": floors.scalars().all(),
        "geojsons": geojsons.scalars().all(),
        "results": results.scalars().all(),
        "gifs": gifs.scalars().all()
    }