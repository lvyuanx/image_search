# app.py
from typing import Any
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from image_search_engine import get_image_search_manager

app = FastAPI(title="以图搜图服务", description="基于 CLIP + FAISS 的可复用图像搜索引擎")

engine_manager = get_image_search_manager()

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


@app.get("/image/list", summary="获取图片列表")
async def image_list(
    group: str = Query("default", description="图片分组"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页数量"),
    keyword: str = Query(None, description="搜索关键词"),
    order: str = Query("desc", description="排序方式")
):
    res = engine_manager.list_gallery(
        group=group,
        page=page,
        page_size=page_size,
        keyword=keyword,
        order=order,
    )
    return Res().ok(res)

@app.post("/image/add")
async def image_add(file: UploadFile = File(..., description="图片"), group: str = Query("default", description="图片分组")):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return Res().fail("不支持的文件格式")
    res = engine_manager.add_images([(file.filename, file.file.read())], group)
    return Res().ok(res)