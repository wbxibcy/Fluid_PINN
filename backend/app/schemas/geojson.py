from pydantic import BaseModel

# 创建 GeoJSON 请求体
class CreateGeoJSONRequest(BaseModel):
    geojson_data: dict