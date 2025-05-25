from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from app.schemas.geojson import CreateGeoJSONRequest
from app.models.geojson import GeoJSON
from app.models.floors import Floor
from app.core.database import get_db
from app.api.auth import get_current_user

router = APIRouter()

# 获取单个 GeoJSON
@router.get("/{geojson_id}")
async def get_geojson(geojson_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    result = await db.execute(select(GeoJSON).where(GeoJSON.geojson_id == geojson_id))
    geojson = result.scalars().first()
    if not geojson:
        raise HTTPException(status_code=404, detail="GeoJSON not found")
    return {"geojson": geojson}

# 创建 GeoJSON
@router.post("/")
async def create_geojson(request: CreateGeoJSONRequest, db: Session = Depends(get_db)):
    new_geojson = GeoJSON(geojson_data=request.geojson_data)
    db.add(new_geojson)
    await db.commit()
    await db.refresh(new_geojson)
    return {"message": "GeoJSON created successfully", "geojson": new_geojson}

# 删除 GeoJSON
@router.delete("/{geojson_id}")
async def delete_geojson(geojson_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    # 查询 GeoJSON 是否存在
    result = await db.execute(select(GeoJSON).where(GeoJSON.geojson_id == geojson_id))
    geojson = result.scalars().first()
    if not geojson:
        raise HTTPException(status_code=404, detail="GeoJSON not found")

    # 将 Floor 表中引用该 geojson_id 的外键设为 NULL
    await db.execute(
        select(Floor).where(Floor.geojson_id == geojson_id)
    )
    await db.execute(
        Floor.__table__.update().where(Floor.geojson_id == geojson_id).values(geojson_id=None)
    )

    # 删除 GeoJSON
    await db.delete(geojson)
    await db.commit()
    return {"message": "GeoJSON deleted successfully"}