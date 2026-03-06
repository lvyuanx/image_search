import hashlib
import time
from typing import Dict, Optional

from fastapi import HTTPException

from common.utils import time_util
from auth.models import Site

class SignUtil:
    def __init__(self, appid: str, secret_key: str, timeout: int = 300):
        """
        :param appid: 接口 appid
        :param secret_key: 接口 secret_key
        :param timeout: 超时时间，单位秒，默认 5 分钟
        """
        self.appid = appid
        self.secret_key = secret_key
        self.timeout = timeout

    def create_sign(self, params: Dict[str, str], timestamp: Optional[int] = None) -> Dict[str, str]:
        """
        生成签名，并返回带 timestamp 的参数字典
        :param params: 请求参数
        :param timestamp: 可指定时间戳，否则使用当前时间
        :return: 新字典，带 timestamp 和 sign
        """
        if timestamp is None:
            timestamp = int(time_util.now_timestamp())

        # 添加 timestamp 到参数
        params_to_sign = dict(params)
        params_to_sign["timestamp"] = str(timestamp)
        params_to_sign["appid"] = self.appid

        # 排序参数
        sorted_items = sorted(params_to_sign.items())
        param_str = "&".join(f"{k}={v}" for k, v in sorted_items)

        # 拼接 secret_key
        raw_str = f"{param_str}&secret_key={self.secret_key}"

        # MD5 加密
        sign = hashlib.md5(raw_str.encode("utf-8")).hexdigest().upper()

        # 返回带 sign 的参数
        params_to_sign["sign"] = sign
        return params_to_sign

    def verify_sign(self, params: Dict[str, str]) -> bool:
        """
        验证签名
        :param params: 请求参数，必须包含 timestamp 和 sign
        :return: True/False
        """
        sign = params.get("sign")
        timestamp_str = params.get("timestamp")

        if not sign or not timestamp_str:
            return False

        try:
            timestamp = int(timestamp_str)
        except ValueError:
            return False

        # 检查超时
        now = int(time.time())
        if abs(now - timestamp) > self.timeout:
            return False

        # 重新生成 sign
        params_to_verify = dict(params)
        params_to_verify.pop("sign")

        sorted_items = sorted(params_to_verify.items())
        param_str = "&".join(f"{k}={v}" for k, v in sorted_items)
        raw_str = f"{param_str}&secret_key={self.secret_key}"
        expected_sign = hashlib.md5(raw_str.encode("utf-8")).hexdigest().upper()

        return expected_sign == sign.upper()


# 创建全局 SignUtil 工厂
async def get_sign_util(appid: str) -> SignUtil:
    """
    根据数据库 Site 表获取 appid 对应的 secret_key，并返回 SignUtil
    """
    site = await Site.filter(appid=appid, is_active=True).first()
    if not site:
        raise HTTPException(status_code=403, detail=f"未知或未启用的 appid: {appid}")

    return SignUtil(appid=site.appid, secret_key=site.secret_key)