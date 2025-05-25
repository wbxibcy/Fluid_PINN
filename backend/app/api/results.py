from fastapi import APIRouter, HTTPException, Depends
from fastapi import File, UploadFile
from fastapi import Form
import json
import os
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from app.schemas.results import CreateResultRequest
from app.models.results import Result
from app.models.gifs import GIF
from app.models.floors import Floor
from app.models.geojson import GeoJSON
from app.core.database import get_db
from app.api.auth import get_current_user
from app.services.tasks import pinn_task, fvm_task

router = APIRouter()

@router.get("/floors/{floor_id}/results")
async def get_results_by_floor(floor_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    获取当前用户的指定 floor 表 ID 的所有 results 数据。
    """
    result = await db.execute(select(Result).where(Result.floor_id == floor_id, Result.user_id == current_user["user"].user_id))
    results = result.scalars().all()
    return {"results": results}


@router.post("/")
async def create_result(request: str = Form(...), csv_file: UploadFile = File(None),db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    创建新的模拟结果（支持 PINN 和 FVM）。
    """

    try:
        request_data = json.loads(request)
        print(request_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in 'request'")
    
    # 检查 Floor 是否存在
    floor_result = await db.execute(select(Floor).where(Floor.floor_id == request_data["floor_id"]))
    floor = floor_result.scalars().first()
    if not floor:
        raise HTTPException(status_code=404, detail="Floor not found")

    # 检查 GeoJSON 是否存在
    geojson_result = await db.execute(select(GeoJSON).where(GeoJSON.geojson_id == floor.geojson_id))
    geojson = geojson_result.scalars().first()
    if not geojson:
        raise HTTPException(status_code=404, detail="GeoJSON not found")
    
    # 保存上传的 CSV 文件
    csv_path = None
    if csv_file:
        upload_dir = "./data"
        os.makedirs(upload_dir, exist_ok=True)  # 确保目录存在
        csv_path = os.path.join(upload_dir, csv_file.filename)
        with open(csv_path, "wb") as f:
            f.write(await csv_file.read())
        print(f"CSV file saved at: {csv_path}")

    # 根据 simulation_type 提取参数
    if request_data["simulation_type"].lower() == "pinn":
        pinn_params = request_data.get("pinn_params", {})
        pinn_params["csv_path"] = csv_path

        task_data = {
            "geojson_data": geojson.geojson_data,
            "source": request_data["source"],
            "pinn_params": pinn_params
        }
        task = pinn_task.apply_async([task_data], queue="pinn")
    elif request_data["simulation_type"].lower() == "fvm":
        task_data = {
            "geojson_data": geojson.geojson_data,
            "source": request_data["source"],
            "fvm_params": request_data["fvm_params"]
        }
        task = fvm_task.apply_async([task_data], queue="fvm")
    else:
        raise HTTPException(status_code=400, detail="Invalid simulation type")

    # 等待任务完成并获取 GIF 文件路径
    try:
        gif_url = await asyncio.to_thread(task.get, timeout=None)
        print(gif_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation task failed: {e}")
    
    # 将本地路径转换为静态文件的 URL
    static_url = gif_url.replace("./outputs", "/static")

    # 将生成的 GIF 文件路径存储到 GIF 表中
    new_gif = GIF(gif_url=static_url)
    db.add(new_gif)
    await db.commit()
    await db.refresh(new_gif)

    # 将结果存储到 Result 表中
    new_result = Result(
        floor_id=request_data["floor_id"],
        user_id=current_user["user"].user_id,
        simulation_type=request_data["simulation_type"],
        description=request_data.get("description", ""),
        gif_id=new_gif.gif_id
    )
    db.add(new_result)
    await db.commit()
    await db.refresh(new_result)

    return {"message": "Result created successfully", "result": new_result}


@router.delete("/{result_id}")
async def delete_user_result(result_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """
    删除当前用户的指定 ID 的 results 数据。
    """
    result = await db.execute(select(Result).where(Result.result_id == result_id, Result.user_id == current_user["user"].user_id))
    result_item = result.scalars().first()
    if not result_item:
        raise HTTPException(status_code=404, detail="Result not found")
    await db.delete(result_item)
    await db.commit()
    return {"message": "Result deleted successfully"}