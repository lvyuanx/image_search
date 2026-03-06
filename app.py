# app.py
import logging
import asyncio
import importlib
from typing import Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from common.utils import common_util
import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="以图搜图服务", description="基于 CLIP + FAISS 的可复用图像搜索引擎")

class Res(BaseModel):
    code: int = Field(0, description="状态码")
    msg: str = Field("成功", description="状态信息")
    data: Any = Field(None, description="数据")
    
    def ok(self, data = None):
        self.data = data
        return JSONResponse(content=self.model_dump(), status_code=200)
    
    def fail(self, msg: str = "失败", code: int = 1):
        self.code = code
        self.msg = msg
        return JSONResponse(content=self.model_dump(), status_code=200)


# 自动加载apps
def load_apps():
    for app_str in settings.INSERTAPPS:
        try:
            app_module = importlib.import_module(f"{app_str}.api")
            if hasattr(app_module, "router"):
                logger.info(f"Load app router: {app_str}")
                app.include_router(app_module.router, prefix="/api")
        except ModuleNotFoundError as e:
            pass
            

_uv_ready_event = asyncio.Event()

async def _check_uv_ready(timeout: float = 10.0):
        host = "127.0.0.1" if settings.HOST == "0.0.0.0" else settings.HOST
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout 

        while loop.time() < deadline:
            try:
                reader, writer = await asyncio.open_connection(host, settings.PORT)
                writer.close()
                await writer.wait_closed()
                _uv_ready_event.set()
                return
            except OSError:
                await asyncio.sleep(0.1)

        raise RuntimeError("Uvicorn 启动超时")

async def _run_uvicorn():
        import uvicorn

        config = uvicorn.Config(
            "app:app",
            host=settings.HOST,
            port=settings.PORT,
            log_level=settings.LOG_LEVEL.lower(),
            log_config=settings.LOGGING,
            workers=settings.WORKERS,
        )

        server = uvicorn.Server(config)

        # 异步启动 uvicorn
        await server.serve()


async def run():

    load_apps()
    
    app_task = common_util.create_task_safe(_run_uvicorn())
    common_util.create_task_safe(_check_uv_ready())

    await _uv_ready_event.wait()
    
    logging.basicConfig(**settings.LOGGING)
    
    HOST = "127.0.0.1" if settings.HOST == "0.0.0.0" else settings.HOST
    PORT = settings.PORT
    
    logger.warning(F"""
    
               
    Uvicorn running on http://{HOST}:{PORT} (Press CTRL+C to quit)

    Docs: http://{HOST}:{PORT}/docs


""")

    await asyncio.gather(app_task)



    