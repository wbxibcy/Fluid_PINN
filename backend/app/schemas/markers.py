from pydantic import BaseModel

# 创建 Marker 请求体
class CreateMarkerRequest(BaseModel):
    latitude: float
    longitude: float
    description: str