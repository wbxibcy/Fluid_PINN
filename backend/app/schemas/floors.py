from pydantic import BaseModel

# 创建 Floor 请求体
class CreateFloorRequest(BaseModel):
    marker_id: int
    name: str
    description: str | None = None
    geojson_id: int | None = None