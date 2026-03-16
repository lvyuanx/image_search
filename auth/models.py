from tortoise import Model, fields

from tortoise import fields
from tortoise.models import Model

from common.utils import time_util


class Site(Model):
    """
    站点 / 应用接入配置
    """

    id = fields.IntField(pk=True, description="站点ID")

    appid = fields.CharField(
        max_length=64,
        unique=True,
        index=True,
        description="应用ID"
    )

    secret_key = fields.CharField(
        max_length=64,
        description="应用密钥"
    )

    name = fields.CharField(
        max_length=64,
        description="站点名称"
    )

    is_active = fields.BooleanField(
        default=True,
        description="是否启用"
    )

    search_quota = fields.IntField(
        default=100,
        description="搜索次数配额，null 表示不限制，默认为 100"
    )

    remark = fields.CharField(
        max_length=255,
        null=True,
        description="备注"
    )

    updated_at = fields.BigIntField(
        default=lambda: int(time_util.now_timestamp()),
        description="更新时间"
    )


    class Meta:
        table = "t_site"
        table_description = "站点应用表"
        indexes = ("appid",)

    def __str__(self):
        return f"{self.name}({self.appid})"


class JdkCard(Model):
    """JDK 兑换卡"""

    id = fields.IntField(pk=True, description="ID")

    code = fields.CharField(
        max_length=32,
        unique=True,
        index=True,
        description="兑换码"
    )

    quota = fields.IntField(
        default=0,
        description="包含的搜索次数"
    )

    is_used = fields.BooleanField(
        default=False,
        description="是否已使用"
    )

    used_appid = fields.CharField(
        max_length=64,
        null=True,
        description="使用的应用ID"
    )

    used_at = fields.BigIntField(
        null=True,
        description="使用时间"
    )

    created_at = fields.BigIntField(
        default=lambda: int(time_util.now_timestamp()),
        description="创建时间"
    )

    expired_at = fields.BigIntField(
        null=True,
        description="过期时间"
    )

    class Meta:
        table = "t_jdk_card"
        table_description = "JDK 兑换卡表"
        indexes = ("code",)

    def __str__(self):
        return f"JDK({self.code}) - {self.quota}次"
