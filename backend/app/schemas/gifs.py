from pydantic import BaseModel

# 创建 GIF 请求体
class CreateGIFRequest(BaseModel):
    gif_url: str