from fastapi import APIRouter, Body
from auth.utils.sign_util import SignUtil
import settings
from .utils import auth_util
from .models import Site
from app import Res


router = APIRouter(tags=["权限管理"])


@router.post("/auth/site", summary="创建站点")
async def site(
    site_name: str = Body(..., description="站点名称"),
    remark: str = Body(None, description="站点备注"),
):
    appid = auth_util.generate_appid()
    secret_key = auth_util.generate_secret_key()
    
    site = Site(
        appid=appid,
        name=site_name,
        secret_key=secret_key,
        remark=remark,
        is_active=True,
    )
    
    await site.save()

    return Res().ok(data={
        "appid": appid,
        "secret_key": secret_key,
    })
    
    
@router.post("/auth/site/info", summary="获取站点信息")
async def site_info(
    appid: str = Body(..., description="应用ID"),
    secret_key: str = Body(..., description="应用密钥"),
):
    site = await Site.get_or_none(appid=appid)
    if not site or site.secret_key != secret_key:
        return Res.fail(msg="应用不存在或密钥错误")
    
    return Res().ok(data={
        "appid": site.appid,
        "name": site.name,
        "remark": site.remark,
        "is_active": site.is_active,
    })

if settings.DEBUG:
    @router.post("/auth/test/sign_create")
    async def test_sign_create(
        appid: str = Body(..., description="应用ID"),
        secret_key: str = Body(..., description="应用密钥"),
        data: dict = Body(..., description="数据"),
    ):
        sign_util = SignUtil(appid, secret_key)
        sign_data = sign_util.create_sign(data)
        return Res().ok(data=sign_data)

    @router.post("/auth/test/sign_verify")
    async def test_sign_verify(
        appid: str = Body(..., description="应用ID"),
        secret_key: str = Body(..., description="应用密钥"),
        data: dict = Body(..., description="数据"),
    ):
        sign_util = SignUtil(appid, secret_key)
        verify_result = sign_util.verify_sign(data)
        return Res().ok(data={"verify_result": verify_result})