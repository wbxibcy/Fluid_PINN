
from celery import Celery
from celery import worker
from app.services.PINN import trainer as pinn_trainer
from app.services.FVM import generate_fvm_animation

celery_app = Celery(
    'tasks',
    broker='pyamqp://xx:xxpassword@localhost//',
    backend='rpc://'
)
celery_app.conf.task_default_queue = 'invalid'
celery_app.conf.task_time_limit = 7200
celery_app.conf.task_soft_time_limit = 7100
celery_app.conf.broker_transport_options = {
    'heartbeat': 0,
    'broker_heartbeat': 0
}

@celery_app.task(queue="pinn")
def pinn_task(task_data):
    """
    PINN 任务：基于 GeoJSON 数据运行 PINN 模型并生成结果 GIF。
    """
    print("Starting PINN task...")
    print(task_data)
    geojson_data = task_data["geojson_data"]
    source = task_data["source"]
    pinn_params = task_data["pinn_params"]

    # 从 pinn_params 中提取 CSV 地址和点坐标
    csv_path = pinn_params.get("csv_path")
    csv_coordinates = pinn_params.get("csv_coordinates")

    print(csv_coordinates)
    x_coords = None
    y_coords = None

    if csv_path:
        # 如果 csv_path 有值，必须要有合法的 csv_coordinates
        if not csv_coordinates or len(csv_coordinates) < 2:
            x_coords = None
            y_coords = None
            # raise ValueError("csv_path 已提供，但 csv_coordinates 缺失或格式错误")
        if csv_coordinates[0] is None or csv_coordinates[1] is None:
            x_coords = None
            y_coords = None
            # raise ValueError("csv_coordinates 不能包含 None")
        if csv_coordinates[0] and csv_coordinates[1]:
            x_coords = [float(csv_coordinates[0])]
            y_coords = [float(csv_coordinates[1])]
        
    try:
        # 调用 PINN 模型训练函数
        print(x_coords, y_coords)
        png_path = pinn_trainer(
        geojson_data=geojson_data,
        csv_path=csv_path,
        x_coords=x_coords,
        y_coords=y_coords,
        source_position=[float(x) for x in source.get("position")],
        source_strength=float(source.get("strength"))
)
        print(f"PINN task completed. PNG saved at: {png_path}")
        return png_path
    except Exception as e:
        print(f"Error in PINN task: {e}")
        raise e

@celery_app.task(queue="fvm")
@celery_app.task
def fvm_task(task_data):
    """
    FVM 任务：基于 GeoJSON 数据运行有限体积法模拟并生成动画 GIF。
    """
    print("Starting FVM task...")
    geojson_data = task_data["geojson_data"]
    source = task_data["source"]
    fvm_params = task_data["fvm_params"]

    # 从 fvm_params 中提取物理参数
    D = fvm_params.get("D")
    u = fvm_params.get("u")
    v = fvm_params.get("v")
    steps = fvm_params.get("steps")

    try:
        # 调用 FVM 模型动画生成函数
        gif_path = generate_fvm_animation(
            geojson_data=geojson_data,
            D=D,
            steps=steps,
            u=u,
            v=v,
            source_position=source.get("position"),
            source_strength=source.get("strength")
        )
        print(f"FVM task completed. GIF saved at: {gif_path}")
        return gif_path
    except Exception as e:
        print(f"Error in FVM task: {e}")
        raise e

# if __name__ == "__main__":
#     print(f"Project Root Directory: {ROOT_DIR}")
#     # 配置 Worker 参数
#     options = [
#         "worker",
#         "--loglevel=info",  # 日志级别
#         "--pool=prefork",   # 使用多进程模式
#     ]

#     print("Starting Celery Worker...")
#     celery_app.worker_main(argv=options)