from app.core.database import redis_client

def set_token(key: str, value: str, expire: int):
    """设置 Redis 中的键值对"""
    print(key, value, expire)
    redis_client.setex(key, expire, value)

def get_token(key: str) -> str:
    """获取 Redis 中的值"""
    value = redis_client.get(key)
    print(value)
    return value

def delete_token(key: str):
    """删除 Redis 中的键"""
    redis_client.delete(key)