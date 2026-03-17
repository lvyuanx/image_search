# app.py
import asyncio
import importlib
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from common.cache.redis_client import RedisClient
from common.exceptions.business_exceptions import BusinessException
import settings
from common.orm import init_db
from common.utils import common_util
from image_search.image_search_engine import warm_up_image_search

logger = logging.getLogger(__name__)
warm_up_image_search()

# 模板和静态文件
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

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
    
    
@app.exception_handler(BusinessException)
async def api_error_handler(request: Request, exc: BusinessException):
    return JSONResponse(
        status_code=200,
        content=Res(code=exc.code, msg=exc.msg).model_dump()
    )    

# ----------------------------- orm -----------------------------
init_db(app)

# ----------------------------- redis -----------------------------
RedisClient.init(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD,
)

# ----------------------------- uvicorn -----------------------------

def load_app_routers():
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
            lifespan="on",
        )

        server = uvicorn.Server(config)

        # 异步启动 uvicorn
        await server.serve()


async def run():

    load_app_routers()
    
    app_task = common_util.create_task_safe(_run_uvicorn())
    common_util.create_task_safe(_check_uv_ready())

    await _uv_ready_event.wait()
    
    HOST = "127.0.0.1" if settings.HOST == "0.0.0.0" else settings.HOST
    PORT = settings.PORT
    
    logger.warning(F"""
    
               
    Uvicorn running on http://{HOST}:{PORT} (Press CTRL+C to quit)

    Docs: http://{HOST}:{PORT}/docs


""")

    await asyncio.gather(app_task)



    