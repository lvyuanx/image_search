from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `t_site` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '站点ID',
    `appid` VARCHAR(64) NOT NULL UNIQUE COMMENT '应用ID',
    `secret_key` VARCHAR(64) NOT NULL COMMENT '应用密钥',
    `name` VARCHAR(64) NOT NULL COMMENT '站点名称',
    `is_active` BOOL NOT NULL COMMENT '是否启用',
    `remark` VARCHAR(255) COMMENT '备注',
    `updated_at` BIGINT NOT NULL COMMENT '更新时间',
    KEY `idx_t_site_appid_725fee` (`appid`)
) CHARACTER SET utf8mb4 COMMENT='站点应用表';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS `t_site`;"""


MODELS_STATE = (
    "eJzll21v2jAQgP8KyqdO2roQkhCmqhLpuo2pBall06S2spzECRaJQxOna9X1v8/nACHhpa"
    "RaW7Z9MXAv9vmxfXfcK1HskTDd75KEuiPlQ+NeYTgi4ktF87ah4MmkkIOAYyeUpriwcVKe"
    "YJcLqY/DlAiRR1I3oRNOYyakLAtDEMauMKQsKEQZo9cZQTwOCB+RRCguroSYMo/cknT2cz"
    "JGPiWhVwqVerC2lCN+N5GyHuOfpCGs5iA3DrOIFcaTOz6K2dyaMg7SgDCSYE5gep5kED5E"
    "N93nbEd5pIVJHuKCj0d8nIV8YbsOKmQKQv3BEJ0fDxFSagByYwZwRaip3H0AIbzTmnpbt1"
    "qmbgkTGeZc0n7Ily7A5I4ST3+oPEg95ji3kIwLqDckSSGkJbJHI5ysRrvgUuErAq/yndHc"
    "BHgmKAgXt+olEEf4FoWEBRyehmYYG4B+754dfeme7QmrN7BkLJ5B/jr6U5WW64B6QRkeVQ"
    "3CU/N/kG5TVbegK6zW0pW6Ml2xIif50y4T/no+6K8mvOBSoexRlzd+NUKaLuWKv4D2BrgA"
    "A2aO0vQ6XGS6d9r9UcV9dDKwJZw45UEiZ5ET2AI9JGh/vJBNQOBgd/wTJx5a0sRavM52WR"
    "VpUVWCGQ4kSNgx7G9ass4pJ6tKmZRvLGQcpTObxwqZcpm1sdcRo+qI0SAdXXw3NOsysyzT"
    "UiqHUjJvvG+UPcwWNoSkaYqx09Q9IfdNGcfjVfEC9iOK2tV/UR5LHHsfq5hfp1TmB1Avja"
    "/k/MRE/kygixu6NehSSjf1LTK6qa9N6KAq5/OUuAnhaEzu6tAue+1y7VTKacFwXFMkBB0b"
    "u4FfftYAP7PfceSlPK6rkH07vrobyGmKRBWiNyu423EcEszWpPFFvwp/Rzg+1wHMMk+tAz"
    "BNzQf0milHP7//2x3ABuD2YHBS6mvs3rBC/tupfSy6SnkgwihvHOYZvziGhEQ4Gde5+4XH"
    "k27/lOML5JuO2hYn4JIteb/Af6Js4gEYhFc07jYN1rYuZb/HW5g/deeVAz9jLiBt4IyP9q"
    "ft5kGII8fDh4fKEx6EL4qAaTgqjD4UAcPXazU7HU1rtdqa2jItQ2+3DUuddz3Lqk3tj937"
    "DO+hdHz5A3nVdv/hNxu15TQ="
)
