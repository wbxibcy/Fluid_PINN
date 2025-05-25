from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from app.schemas.gifs import CreateGIFRequest  # 导入请求体模型
from app.models.gifs import GIF
from app.core.database import get_db
from app.api.auth import get_current_user

router = APIRouter()

# 获取单个 GIF
@router.get("/{gif_id}")
async def get_gif(gif_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    result = await db.execute(select(GIF).where(GIF.gif_id == gif_id))
    gif = result.scalars().first()
    if not gif:
        raise HTTPException(status_code=404, detail="GIF not found")
    return {"gif": gif}

# 创建 GIF
@router.post("/")
async def create_gif(request: CreateGIFRequest, db: Session = Depends(get_db)):
    new_gif = GIF(gif_url=request.gif_url)
    db.add(new_gif)
    await db.commit()
    await db.refresh(new_gif)
    return {"message": "GIF created successfully", "gif": new_gif}

# 删除 GIF
@router.delete("/{gif_id}")
async def delete_gif(gif_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    result = await db.execute(select(GIF).where(GIF.gif_id == gif_id))
    gif = result.scalars().first()
    if not gif:
        raise HTTPException(status_code=404, detail="GIF not found")
    await db.delete(gif)
    await db.commit()
    return {"message": "GIF deleted successfully"}