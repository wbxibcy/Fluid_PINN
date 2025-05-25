from app.models.user import User
from app.models.markers import Marker
from app.models.floors import Floor
from app.models.geojson import GeoJSON
from app.models.gifs import GIF
from app.models.results import Result
from sqlalchemy.ext.asyncio import AsyncSession

async def insert_test_data(session: AsyncSession):
    """
    插入更多测试数据。
    """
    # 插入用户数据
    print("Inserting test users...")
    users = [
        User(user_name="test_user1", email="test_user1@example.com", password_hash="hashed_password1", role="user"),
        User(user_name="test_user2", email="test_user2@example.com", password_hash="hashed_password2", role="user"),
        User(user_name="admin_user", email="admin_user@example.com", password_hash="hashed_password3", role="admin"),
    ]
    session.add_all(users)
    await session.commit()

    # 插入 Marker 数据
    print("Inserting test markers...")
    markers = [
        Marker(latitude=37.7749, longitude=-122.4194, description="Marker 1 for User 1", user_id=1),
        Marker(latitude=34.0522, longitude=-118.2437, description="Marker 2 for User 1", user_id=1),
        Marker(latitude=40.7128, longitude=-74.0060, description="Marker 1 for User 2", user_id=2),
    ]
    session.add_all(markers)
    await session.commit()

    # 插入 Floor 数据
    print("Inserting test floors...")
    floors = [
        Floor(user_id=1, marker_id=1, name="Floor 1 for Marker 1", description="Description for Floor 1"),
        Floor(user_id=1, marker_id=2, name="Floor 2 for Marker 2", description="Description for Floor 2"),
        Floor(user_id=2, marker_id=3, name="Floor 1 for Marker 3", description="Description for Floor 3"),
    ]
    session.add_all(floors)
    await session.commit()

    # 插入 GeoJSON 数据
    print("Inserting test geojson...")
    geojsons = [
        GeoJSON(geojson_data={"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [37.7749, -122.4194]}, "properties": {}}]}),
        GeoJSON(geojson_data={"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [34.0522, -118.2437]}, "properties": {}}]}),
    ]
    session.add_all(geojsons)
    await session.commit()

    # 插入 GIF 数据
    print("Inserting test gifs...")
    gifs = [
        GIF(gif_id=1, gif_url="gif1.gif"),
        GIF(gif_id=2, gif_url="gif2.gif"),
        GIF(gif_id=3, gif_url="gif3.gif"),
    ]
    session.add_all(gifs)
    await session.commit()

    # 插入 Result 数据
    print("Inserting test results...")
    results = [
        Result(user_id=1, floor_id=1, simulation_type="FVM", gif_id=1, description="FVM simulation for Floor 1"),
        Result(user_id=1, floor_id=2, simulation_type="PINN", gif_id=2, description="PINN simulation for Floor 2"),
        Result(user_id=2, floor_id=3, simulation_type="PINN", gif_id=3, description="PINN simulation for Floor 3"),
    ]
    session.add_all(results)
    await session.commit()

    print("Test data inserted successfully.")