from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict

class Source(BaseModel):
    position: List[float]  # 确保 position 是一个浮点数列表
    strength: float

class CreateResultRequest(BaseModel):
    floor_id: int
    simulation_type: str  # "pinn" 或 "fvm"
    description: Optional[str] = None
    source: Source
    # coordinates: Optional[List[Tuple[float, float]]] = None  # 网格点坐标
    fvm_params: Optional[Dict[str, Optional[float]]] = None  # FVM 参数（如 D, u, v, steps）
    pinn_params: Optional[Dict[str, Optional[float]]] = None  # PINN 参数（如 CSV 地址和点坐标）