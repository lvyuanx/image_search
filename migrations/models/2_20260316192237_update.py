from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `t_site` ADD `search_quota` INT COMMENT '搜索次数配额，null 表示不限制';
        ALTER TABLE `t_site` MODIFY COLUMN `remark` VARCHAR(255) COMMENT '备注';
        ALTER TABLE `t_site` MODIFY COLUMN `is_active` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1;
        ALTER TABLE `t_site` MODIFY COLUMN `appid` VARCHAR(64) NOT NULL COMMENT '应用ID';
        ALTER TABLE `t_site` MODIFY COLUMN `secret_key` VARCHAR(64) NOT NULL COMMENT '应用密钥';
        ALTER TABLE `t_site` MODIFY COLUMN `name` VARCHAR(64) NOT NULL COMMENT '站点名称';
        ALTER TABLE `t_site` MODIFY COLUMN `updated_at` BIGINT NOT NULL COMMENT '更新时间';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `t_site` DROP COLUMN `search_quota`;
        ALTER TABLE `t_site` MODIFY COLUMN `remark` VARCHAR(255) COMMENT '备注';
        ALTER TABLE `t_site` MODIFY COLUMN `is_active` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1;
        ALTER TABLE `t_site` MODIFY COLUMN `appid` VARCHAR(64) NOT NULL COMMENT '应用ID';
        ALTER TABLE `t_site` MODIFY COLUMN `secret_key` VARCHAR(64) NOT NULL COMMENT '应用密钥';
        ALTER TABLE `t_site` MODIFY COLUMN `name` VARCHAR(64) NOT NULL COMMENT '站点名称';
        ALTER TABLE `t_site` MODIFY COLUMN `updated_at` BIGINT NOT NULL COMMENT '更新时间';"""


MODELS_STATE = (
    "eJzdl1lvm0AQgP+KxVMqpS3GgKGKItnp5SqJpRxVpSRCCywYGRYHljRRlP/emcU2h7FjW2"
    "mc9mWN54CZj9mZ5VGKYpeG6YceTQJnJH1qPUqMRBQuapr9lkQmk0KOAk7sUJiSwsZOeUIc"
    "DlKPhCkFkUtTJwkmPIgZSFkWhiiMHTAMmF+IMhbcZtTisU/5iCaguLoBccBcek/T2d/J2P"
    "ICGrqVUAMXny3kFn+YCNmA8a/CEJ9mW04cZhErjCcPfBSzuXXAOEp9ymhCOMXb8yTD8DG6"
    "aZ6zjPJIC5M8xJKPSz2ShbyU7poMnJghP4gmFQn6+JT3SlvtqkZHVw0wEZHMJd2nPL0i99"
    "xREDi9kJ6EnnCSWwiMBbc7mqQY0gK8oxFJmumVXGoIIfA6whmwVQxnggJiUTgvRDEi91ZI"
    "mc+xwBVNW8HsZ+/s6HvvbA+s3mE2MRRzXuOnU5WS6xBsARK3xgYQp+b/JsC2LK8BEKyWAh"
    "S6KkB4Iqf5HqxC/HE+PG2GWHKpgbxkkOCVGzh8vxUGKb95m1hXUMSsMegoTW/DMry9k96v"
    "Otej42FfUIhT7ifiLuIGfWCMLdMblzY/CmzijH+TxLUWNLESL7NdVEVKVJcQRnzBCjPG/K"
    "ZD5DzgtGm4CPnK0cKtdGbz3GiRrrMucU1YZRtWjZoqXGuKcZ0Zhm5ItZdSMW99bFU99A7R"
    "QNLWYTXbqgtyTxdxPD+nrjAfGDM3/8vAqqAafK6T3M3wyhlv1nUbUW7Zd7dnWdTZ2iwrHV"
    "hX12jAurq0/6Kq2n5T6iSUW2P6sAnQqteOp5lU3b+a7eiwc1WivQ3C4ncDtjP73VOt9FRV"
    "xk5oevLboBqkFkyE4K4BbT+OQ0rYkpZa9qshtsHxbzGet4gaY11XPKSr6GL18ipej/EKpv"
    "3h8LhyjOgPLmpwL0/6X+C0JpiDUT6n59233CFI4oys2yzmZIPZVXd7foo1oJ5Se5Fq1tW2"
    "A3RdRYFrW2nDqnXl2Yw3DRNWz5Md9G/lBwcs+A65zlSKxW/qGphrSkd/zTFYvIiERiQZb9"
    "JKCo+tmslL0tdMuQvEHbpmbb/Cp1s2cTF3izR8fPQDf2llV/22quutWoh04GXMQaQtkvHR"
    "h+lJ+iAkke2Sw0Opub94Kla6LePq4WTUPHWjAjYVpdPpKnJHNzS129UMeV7Ji6pVJd0ffM"
    "OqrryhvMx3+rHy9Ad8RgBd"
)
