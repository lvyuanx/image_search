import hashlib
import os
from pathlib import Path as FsPath
from urllib.parse import urlencode, urlparse
from fastapi import Depends, File, Path, Query, UploadFile, APIRouter, UploadFile
from typing import Optional

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from app import Res
from auth.models import Site
from auth.sign_depends import verify_sign_dependency
from auth.utils.sign_util import get_sign_util
from common.exceptions.business_exceptions import BusinessException
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
        base_url = settings.IMAGE_PREVIEW_URL_TEMPLATE.format(group=image["group"], name=image["original_name"])
        parsed = urlparse(str(base_url))
        url_path = parsed.path  # 不包含 query

        # 缩略图（w=400）
        thumb_sign = signutil.create_sign(params={"url": url_path})
        del thumb_sign["url"]
        image["url"] = base_url + "?" + urlencode({**thumb_sign, "w": 400})

        # 大图（无损压缩 + 水印）
        full_sign = signutil.create_sign(params={"url": url_path})
        del full_sign["url"]
        image["full_url"] = base_url + "?" + urlencode({**full_sign, "lossless": 1})


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
    # 检查搜索次数配额
    site = await Site.filter(appid=appid, is_active=True).first()
    if not site:
        raise BusinessException(code=403, msg="无效的 appid")
    
    if site.search_quota != -1:
        if site.search_quota <= 0:
            raise BusinessException(code=403, msg="搜索次数已用尽，请联系管理员充值")
        # 扣减次数
        await Site.filter(appid=appid).update(search_quota=site.search_quota - 1)
    
    file_md5 = image_util.calc_file_md5(file)
    if md5 and md5 != file_md5:
        return Res.fail("图片 MD5 不一致")
    contents = await file.read()
    file_type = file.content_type.split("/")[-1].lower() if file.content_type else ""
    compressed = image_util.lossless_compress_bytes(contents, format_hint=file_type)
    res = engine_manager.search(compressed, group=group)
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
    file_type = file.content_type.split('/')[-1]
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
    lossless: int = Query(0, description="大图无损模式（1=启用）"),
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

    is_lossless = lossless == 1

    # 缓存：同一张图 + 参数命中则直接返回文件
    cache_dir = image_lib_dir / group / "preview_cache"
    os.makedirs(cache_dir, exist_ok=True)

    if is_lossless:
        def _cache_name():
            raw_key = f"{stored_name}|full|watermark:lvyuanxiang"
            digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
            return f"{FsPath(stored_name).stem}_{digest}.png"

        cache_path = cache_dir / _cache_name()
        media_type = "image/png"

        if not cache_path.exists():
            def process_and_cache_lossless():
                image = Image.open(file_path)
                image = image_util.add_watermark(image, "lvyuanxiang")
                tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
                image.save(tmp_path, "PNG", optimize=True)
                os.replace(tmp_path, cache_path)
                return cache_path

            cache_path = await run_in_threadpool(process_and_cache_lossless)
    else:
        def _cache_name():
            raw_key = f"{stored_name}|{w}|{h}|{f}|watermark:lvyuanxiang"
            digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
            return f"{FsPath(stored_name).stem}_{digest}.jpg"

        cache_path = cache_dir / _cache_name()
        media_type = "image/jpeg"

        if not cache_path.exists():
            def process_and_cache():
                image = Image.open(file_path)
                image = image.convert("RGB")
                image = image_util.process_image(image, w, h, f)
                image = image_util.add_watermark(image, "lvyuanxiang")
                tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
                image.save(tmp_path, "JPEG", quality=75)
                os.replace(tmp_path, cache_path)
                return cache_path

            cache_path = await run_in_threadpool(process_and_cache)

    return FileResponse(cache_path, media_type=media_type)
       

    
    
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
