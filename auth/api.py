from fastapi import APIRouter, Body, Depends, Query, Request
from auth.sign_depends import verify_sign_dependency
from auth.utils.sign_util import SignUtil
import secrets
import settings
from common.utils import time_util
from .utils import auth_util
from .models import Site, JdkCard
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
        return  Res().fail(msg="应用不存在或密钥错误")
    
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


@router.post("/auth/jdk/create", summary="生成 JDK 兑换卡")
async def create_jdk(
    quota: int = Body(10, description="包含的搜索次数"),
    count: int = Body(1, description="生成数量"),
    days: int = Body(None, description="有效期天数，不传则永久有效"),
):
    """生成 JDK 兑换卡"""
    jdk_list = []
    now = int(time_util.now_timestamp())
    expired_at = now + (days * 86400) if days else None

    for _ in range(count):
        # 生成 16 位随机兑换码
        code = secrets.token_hex(8).upper()
        jdk = await JdkCard.create(
            code=code,
            quota=quota,
            expired_at=expired_at,
        )
        jdk_list.append({
            "code": jdk.code,
            "quota": jdk.quota,
            "expired_at": jdk.expired_at,
        })

    return Res().ok(data=jdk_list)


@router.post("/auth/jdk/redeem", summary="兑换 JDK 次数", dependencies=[Depends(verify_sign_dependency)])
async def redeem_jdk(
    request: Request,
    code: str = Query(..., description="兑换码"),
):
    """使用 JDK 兑换搜索次数"""
    appid = request.state.appid 
    # 验证站点
    site = await Site.get_or_none(appid=appid)
    if not site or site.secret_key != request.state.secret_key:
        return  Res().fail(msg="应用不存在或密钥错误")

    # 查询兑换卡
    jdk = await JdkCard.get_or_none(code=code)
    if not jdk:
        return  Res().fail(msg="兑换码不存在")

    if jdk.is_used:
        return  Res().fail(msg="兑换码已使用")

    if jdk.expired_at and jdk.expired_at < int(time_util.now_timestamp()):
        return  Res().fail(msg="兑换码已过期")

    # 标记已使用
    now = int(time_util.now_timestamp())
    await JdkCard.filter(id=jdk.id).update(
        is_used=True,
        used_appid=appid,
        used_at=now,
    )

    # 更新站点配额
    current_quota = site.search_quota or 0
    await Site.filter(appid=appid).update(
        search_quota=current_quota + jdk.quota
    )

    return Res().ok(data={
        "quota": jdk.quota,
        "total_quota": current_quota + jdk.quota,
    })


@router.post("/auth/quota", summary="获取剩余搜索次数", dependencies=[Depends(verify_sign_dependency)])
async def get_quota(
    request: Request
):
    """获取站点的剩余搜索次数"""
    site = await Site.get_or_none(appid= request.state.appid)
    if not site or site.secret_key != request.state.secret_key:
        return  Res().fail(msg="应用不存在或密钥错误")
    
    return Res().ok(data={
        "search_quota": site.search_quota,
    })