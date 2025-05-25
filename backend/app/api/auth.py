from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone

from app.core.database import get_db
from app.models.user import User
from app.core.config import config
from app.services.redis_service import set_token, get_token
from app.schemas.auth import RegisterUserRequest, LoginUserRequest

router = APIRouter()

# 密码加密工具
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT 配置
SECRET_KEY = config.SECRET_KEY
ALGORITHM = config.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = config.ACCESS_TOKEN_EXPIRE_MINUTES

# OAuth2 配置
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# 创建访问令牌
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 验证密码
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# 哈希密码
def get_password_hash(password):
    return pwd_context.hash(password)

# 注册用户
@router.post("/register")
async def register_user(request: RegisterUserRequest, db: Session = Depends(get_db)):
    # 检查邮箱是否已注册
    print(request.user_name, request.email)
    result = await db.execute(select(User).where(User.email == request.email))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # 创建新用户
    hashed_password = get_password_hash(request.password)
    new_user = User(user_name=request.user_name, email=request.email, password_hash=hashed_password, role=request.role)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"message": "User registered successfully"}

# 用户登录
@router.post("/login")
async def login_user(request: LoginUserRequest, db: Session = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalars().first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.email, "role": user.role})  # 添加 role
    set_token(f"token:{user.email}", access_token, ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return {"access_token": access_token, "token_type": "bearer"}

# 验证用户
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        role = payload.get("role")  # 获取角色信息
        if email is None or role is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return {"user": user, "role": role}  # 返回用户和角色信息
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

