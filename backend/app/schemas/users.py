from pydantic import BaseModel

# 更新用户信息请求体
class UpdateUserRequest(BaseModel):
    full_name: str