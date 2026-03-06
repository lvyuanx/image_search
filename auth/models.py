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
