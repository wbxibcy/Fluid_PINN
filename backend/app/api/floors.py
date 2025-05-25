from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from app.schemas.floors import CreateFloorRequest  # 导入请求体模型
from app.models.floors import Floor
from app.core.database import get_db
from app.api.auth import get_current_user

router = APIRouter()


@router.get("/markers/{marker_id}/floors")
async def get_floors_by_marker(marker_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    获取当前用户的指定 marker 表 ID 的所有 floor 数据。
    """
    result = await db.execute(select(Floor).where(Floor.marker_id == marker_id, Floor.user_id == current_user["user"].user_id))
    floors = result.scalars().all()
    return {"floors": floors}


@router.post("/")
async def create_floor(request: CreateFloorRequest, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    为当前用户新增一个 floor 数据，关联指定的 marker 表 ID。
    """
    new_floor = Floor(
        user_id=current_user["user"].user_id,
        marker_id=request.marker_id,
        name=request.name,
        description=request.description,
        geojson_id=request.geojson_id
    )
    db.add(new_floor)
    await db.commit()
    await db.refresh(new_floor)
    return {"message": "Floor created successfully", "floor": new_floor}


@router.delete("/{floor_id}")
async def delete_user_floor(floor_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    删除当前用户的指定 ID 的 floor 数据。
    """
    result = await db.execute(select(Floor).where(Floor.floor_id == floor_id, Floor.user_id == current_user["user"].user_id))
    floor = result.scalars().first()
    if not floor:
        raise HTTPException(status_code=404, detail="Floor not found")
    await db.delete(floor)
    await db.commit()
    return {"message": "Floor deleted successfully"}