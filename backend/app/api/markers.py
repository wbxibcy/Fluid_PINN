from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from app.schemas.markers import CreateMarkerRequest  # 导入请求体模型
from app.models.markers import Marker
from app.core.database import get_db
from app.api.auth import get_current_user

router = APIRouter()

# 获取单个 Marker
@router.get("/")
async def get_user_markers(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    result = await db.execute(select(Marker).where(Marker.user_id == current_user["user"].user_id))
    markers = result.scalars().all()
    return {"markers": markers}

# 创建 Marker
@router.post("/")
async def create_marker(request: CreateMarkerRequest, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    new_marker = Marker(
        user_id=current_user["user"].user_id,
        latitude=request.latitude,
        longitude=request.longitude,
        description=request.description
    )
    db.add(new_marker)
    await db.commit()
    await db.refresh(new_marker)
    return {"message": "Marker created successfully", "marker": new_marker}

# 删除 Marker
@router.delete("/{marker_id}")
async def delete_user_marker(marker_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    删除当前登录用户的特定 ID 的 marker 数据。
    """
    result = await db.execute(select(Marker).where(Marker.marker_id == marker_id, Marker.user_id == current_user["user"].user_id))
    marker = result.scalars().first()
    if not marker:
        raise HTTPException(status_code=404, detail="Marker not found")
    await db.delete(marker)
    await db.commit()
    return {"message": "Marker deleted successfully"}