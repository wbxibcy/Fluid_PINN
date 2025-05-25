from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from app.models.user import User
from app.core.database import get_db
from app.api.auth import get_password_hash, get_current_user

router = APIRouter()

# 获取用户信息
@router.get("/me")
async def get_user_info(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(select(User).where(User.user_id == current_user["user"].user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user_info = {
        "user_id": user.user_id,
        "user_name": user.user_name,
        "email": user.email,
        "role": user.role,
        "full_name": user.full_name
    }
    return {"user": user_info}

# 删除用户
@router.delete("/")
async def delete_user(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(select(User).where(User.user_id == current_user["user"].user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await db.delete(user)
    await db.commit()
    return {"message": "User deleted successfully"}
