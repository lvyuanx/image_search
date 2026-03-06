import io
import mimetypes
from fastapi import File, Path, Query, UploadFile, APIRouter, UploadFile
from typing import Optional

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from app import Res
from common.utils import image_util
import settings
from PIL import Image, ImageDraw, ImageFont
from .image_search_engine import get_image_search_manager

engine_manager = get_image_search_manager()


router = APIRouter(tags=["图库"])

def generate_url(image: dict | list[dict] = None):
    pass


@router.get("/image", summary="获取图片列表")
async def image_list(
    group: str = Query("default", description="图片分组"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页数量"),
    keyword: str = Query(None, description="搜索关键词"),
    order: str = Query("desc", description="排序方式"),
):
    res = engine_manager.list_gallery(
        group=group,
        page=page,
        page_size=page_size,
        keyword=keyword,
        order=order,
    )
    return Res().ok(res)

@router.post("/image/search", summary="以图搜图")
def image_search(
    img_bytes: UploadFile = File(..., description="图片"),
    group: str = Query("default", description="分组"),
):
    res = engine_manager.search(img_bytes.file.read(), group=group)
    return Res().ok(res)


@router.post("/image", summary="添加图片")
async def image_add(
    file: UploadFile = File(..., description="图片"),
    group: str = Query("default", description="图片分组"),
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return Res().fail("不支持的文件格式")
    res = engine_manager.add_images([(file.filename, file.file.read())], group)
    return Res().ok(res)


@router.delete("/image", summary="删除图片")
async def image_delete(
    stored_name: str = Query(..., description="图片名称"),
    group: Optional[str] = Query(None, description="图片分组"),
):
    engine_manager.delete_image(stored_name, group)
    return Res().ok()


@router.get("/image/rebuild", summary="重建索引")
async def rebuild():
    engine_manager.rebuild_index()
    return Res().ok()


@router.get("/image/preview/{group}/{name}", summary="图片预览")
async def image_preview(
    name: str = Path(..., description="图片名称"),
    group: str = Path(..., description="图片分组"),
    w: int = Query(None, description="图片宽度"),
    h: int = Query(None, description="图片高度"),
    f: str = Query("contain", description="图片处理方式"),
):

    # 查询
    res = await run_in_threadpool(engine_manager.search_by_name_exact, name, group)

    if not res:
        return Res().fail("图片不存在")

    image_lib_dir = settings.BASE_DIR / "oss" / "media" / "groups"

    image_info = res[0]
    group = image_info["group"]
    stored_name = image_info["stored_name"]

    file_path = image_lib_dir / group / "gallery" / stored_name

    # 动态获取 mime
    media_type, _ = mimetypes.guess_type(stored_name)
    if not media_type:
        media_type = "image/jpeg"

    def process_image():

        image = Image.open(file_path)

        image = image.convert("RGB")

        image = image_util.process_image(image, w, h, f)

        image = image_util.compress_image(image, 75)

        image = image_util.add_watermark(image, "lvyuanxiang")

        buffer = io.BytesIO()

        image.save(buffer, "JPEG", quality=75)

        buffer.seek(0)

        return buffer

    buffer = await run_in_threadpool(process_image)

    return StreamingResponse(buffer, media_type=media_type)
       

    
    
    