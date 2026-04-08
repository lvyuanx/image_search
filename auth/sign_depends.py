from fastapi import HTTPException, Request
from typing import Dict
from urllib.parse import urlparse, parse_qsl
from auth.utils.sign_util import get_sign_util
from app import Res
from common.exceptions.business_exceptions import BusinessException

async def verify_sign_dependency(request: Request):
    """
    全局签名校验
    - query 参数 + body (json/form) 参与签名
    - URL path (不包含 query) 也参与签名
    """
    # 1️⃣ 获取 URL path
    parsed = urlparse(str(request.url))
    url_path = parsed.path  # 不包含 query

    # 2️⃣ 获取 query 参数
    query_params = dict(parse_qsl(parsed.query))

    # 3️⃣ 获取 body 参数
    body_params: Dict[str, str] = {}
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.json()
                if isinstance(body, dict):
                    body_params.update(body)
            # elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            #     form = await request.form()
            #     body_params.update(form)
        except Exception:
            pass

    # 4️⃣ 合并 query + body 参数
    params_for_sign = {}
    params_for_sign.update(query_params)
    params_for_sign.update(body_params)

    # 5️⃣ 检查 appid
    appid = params_for_sign.get("appid")
    if not appid:
        raise BusinessException(code=403, msg="appid 不能为空")

    # 6️⃣ 获取 SignUtil
    sign_util = await get_sign_util(appid)

    # 7️⃣ 将 URL path 添加到签名字段
    params_for_sign["url"] = url_path

    # 8️⃣ 校验签名
    if not sign_util.verify_sign(params_for_sign):
        raise BusinessException(code=403, msg="签名校验失败")

    # 9️⃣ 将 appid、sign_util、site 存储到 request 中，供后续处理器复用，避免重复查询
    request.state.appid = appid
    request.state.secret_key = sign_util.secret_key
    request.state.sign_util = sign_util
    request.state.site = sign_util.site