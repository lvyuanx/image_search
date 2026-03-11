# redis_client.py

import json
import redis
import redis.asyncio as aioredis
from typing import Any, Optional


class RedisClient:
    """
    工业级 Redis 客户端
    - 支持 async / sync
    - JSON 自动序列化
    - 分布式锁
    """

    _sync_client: Optional[redis.Redis] = None
    _async_client: Optional[aioredis.Redis] = None

    PREFIX = "img_search"

    # ==============================
    # 初始化
    # ==============================

    @classmethod
    def init(
        cls,
        host="127.0.0.1",
        port=6379,
        db=0,
        password=None,
        max_connections=50,
    ):

        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=False,
        )

        cls._sync_client = redis.Redis(connection_pool=pool)

        cls._async_client = aioredis.Redis(
            connection_pool=aioredis.ConnectionPool.from_url(
                f"redis://{host}:{port}/{db}",
                password=password,
                max_connections=max_connections,
                decode_responses=False,
            )
        )

    # ==============================
    # 获取客户端
    # ==============================

    @classmethod
    def sync(cls) -> redis.Redis:
        if not cls._sync_client:
            raise RuntimeError("RedisClient not initialized")
        return cls._sync_client

    @classmethod
    def async_client(cls) -> aioredis.Redis:
        if not cls._async_client:
            raise RuntimeError("RedisClient not initialized")
        return cls._async_client

    # ==============================
    # key 处理
    # ==============================

    @classmethod
    def key(cls, key: str) -> str:
        return f"{cls.PREFIX}:{key}"

    # ==============================
    # 同步方法
    # ==============================

    @classmethod
    def get(cls, key: str):
        return cls.sync().get(cls.key(key))

    @classmethod
    def set(cls, key: str, value, ex=None):
        return cls.sync().set(cls.key(key), value, ex=ex)

    @classmethod
    def delete(cls, key: str):
        return cls.sync().delete(cls.key(key))

    @classmethod
    def get_json(cls, key: str):
        v = cls.get(key)
        if not v:
            return None
        return json.loads(v)

    @classmethod
    def set_json(cls, key: str, value: Any, ex=None):
        cls.set(key, json.dumps(value), ex)

    # ==============================
    # 异步方法
    # ==============================

    @classmethod
    async def aget(cls, key: str):
        return await cls.async_client().get(cls.key(key))

    @classmethod
    async def aset(cls, key: str, value, ex=None):
        return await cls.async_client().set(cls.key(key), value, ex=ex)

    @classmethod
    async def adelete(cls, key: str):
        return await cls.async_client().delete(cls.key(key))

    @classmethod
    async def aget_json(cls, key: str):
        v = await cls.aget(key)
        if not v:
            return None
        return json.loads(v)

    @classmethod
    async def aset_json(cls, key: str, value: Any, ex=None):
        await cls.aset(key, json.dumps(value), ex)

    # ==============================
    # 分布式锁
    # ==============================

    @classmethod
    def lock(cls, name: str, timeout=30):
        return cls.sync().lock(cls.key(f"lock:{name}"), timeout=timeout)

    # ==============================
    # 批量删除
    # ==============================

    @classmethod
    def delete_pattern(cls, pattern: str):

        client = cls.sync()
        pattern = cls.key(pattern)

        cursor = 0
        while True:

            cursor, keys = client.scan(cursor, match=pattern, count=100)

            if keys:
                client.delete(*keys)

            if cursor == 0:
                break