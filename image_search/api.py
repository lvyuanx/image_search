import io
import mimetypes
import imghdr
from urllib.parse import urlencode, urlparse
from fastapi import Depends, File, Path, Query, UploadFile, APIRouter, UploadFile
from typing import Optional

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from app import Res
from auth.sign_depends import verify_sign_dependency
from auth.utils.sign_util import get_sign_util
from common.utils import file_util, image_util, time_util
import settings
from PIL import Image, ImageDraw, ImageFont
from .image_search_engine import GROUP_BACK, IMAGE_SEARCH_WORKSPACES, get_image_search_manager

engine_manager = get_image_search_manager()


router = APIRouter(tags=["图库"], dependencies=[Depends(verify_sign_dependency)])

async def generate_url(appid, images: dict | list[dict] = None):
    if images is None: return None
    if isinstance(images, dict):
        images = [images]
    
    signutil = await get_sign_util(appid=appid)
    for image in images:
        url = settings.IMAGE_PREVIEW_URL_TEMPLATE.format(group=image["group"], name=image["original_name"])
        parsed = urlparse(str(url))
        url_path = parsed.path  # 不包含 query
        sign_params = signutil.create_sign(params={
            "url": url_path,
        })
        del sign_params["url"]
        image["url"] = url + "?" + urlencode(sign_params)


@router.get("/image", summary="获取图片列表")
async def image_list(
    group: str = Query("default", description="图片分组"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页数量"),
    keyword: str = Query(None, description="搜索关键词"),
    order: str = Query("desc", description="排序方式"),
    sign: str = Query(None, description="签名"),
    appid: str = Query(None, description="应用ID"),
    timestamp: int = Query(None, description="时间戳"),
):
    res = engine_manager.list_gallery(
        group=group,
        page=page,
        page_size=page_size,
        keyword=keyword,
        order=order,
    )
    await generate_url(appid=appid, images=res["results"])
    return Res().ok(res)

@router.post("/image/search", summary="以图搜图")
async def image_search(
    file: UploadFile = File(..., description="图片"),
    md5: str = Query(None, description="图片 MD5"),
    group: str = Query("default", description="分组"),
    sign: str = Query(None, description="签名"),
    appid: str = Query(None, description="应用ID"),
    timestamp: int = Query(None, description="时间戳"),
):
    file_md5 = image_util.calc_file_md5(file)
    if md5 and md5 != file_md5:
        return Res.fail("图片 MD5 不一致")
    res = engine_manager.search(file.file.read(), group=group)
    await generate_url(appid=appid, images=res)
    return Res().ok(res)


@router.post("/image", summary="添加图片")
async def image_add(
    file: UploadFile = File(..., description="图片"),
    md5: str = Query(None, description="图片MD5"),
    group: str = Query("default", description="图片分组"),
    sign: str = Query(None, description="签名"),
    appid: str = Query(None, description="应用ID"),
    timestamp: int = Query(None, description="时间戳"),
):
    file_md5 = image_util.calc_file_md5(file)
    if md5 and md5 != file_md5:
        return Res().fail("图片 MD5 不一致")
    contents = await file.read()
    file_type = imghdr.what(None, h=contents)
    if file_type not in ["jpeg", "png", "jpg"]:
        return Res().fail("不支持的文件格式")
    res = engine_manager.add_images([(file.filename, contents)], group)
    return Res().ok(res)


@router.delete("/image", summary="删除图片")
async def image_delete(
    stored_name: str = Query(None, description="图片保存名称"),
    origin_name: str = Query(None, description="图片原始名称"),
    group: Optional[str] = Query(None, description="图片分组"),
    sign: str = Query(None, description="签名"),
    appid: str = Query(None, description="应用ID"),
    timestamp: int = Query(None, description="时间戳"),
):
    if not stored_name and not origin_name: 
        return Res().fail("stored_name 和 origin_name 不能都为空")
    engine_manager.delete_image(stored_name, origin_name, group)
    return Res().ok()


@router.get("/image/rebuild", summary="重建索引")
async def rebuild(
    sign: str = Query(None, description="签名"),
    appid: str = Query(None, description="应用ID"),
    timestamp: int = Query(None, description="时间戳"),
):
    engine_manager.rebuild_index()
    return Res().ok()


@router.get("/image/preview/{group}/{name}", summary="图片预览")
async def image_preview(
    name: str = Path(..., description="图片名称"),
    group: str = Path(..., description="图片分组"),
    w: int = Query(None, description="图片宽度"),
    h: int = Query(None, description="图片高度"),
    f: str = Query("contain", description="图片处理方式"),
    sign: str = Query(None, description="签名"),
    appid: str = Query(None, description="应用ID"),
    timestamp: int = Query(None, description="时间戳"),
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
       

    
    
@router.delete("/image/clear", summary="清空图片")
def clear(
    group: str = Query(..., description="图片分组"),
    sign: str = Query(None, description="签名"),
    appid: str = Query(None, description="应用ID"),
    timestamp: int = Query(None, description="时间戳"),
):
    groups = engine_manager.list_groups()
    if group not in groups:
        return Res().fail("分组不存在")
    
    group_path = IMAGE_SEARCH_WORKSPACES / group
    
    zip_name = f"{group}_{time_util.now_str('%Y%m%d%H%M%S')}.zip"
    file_util.zip_dir(str(group_path), GROUP_BACK / zip_name)
    
    group_data = IMAGE_SEARCH_WORKSPACES  / group / "data"
    file_util.clear_dir(group_data)
    group_deleted = IMAGE_SEARCH_WORKSPACES  / group / "deleted"
    file_util.clear_dir(group_deleted)
    group_gallery = IMAGE_SEARCH_WORKSPACES  / group / "gallery"
    file_util.clear_dir(group_gallery)
    
    engine_manager.rebuild_index(group)

    return Res().ok(zip_name)
