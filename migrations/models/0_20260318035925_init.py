from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `aerich` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `version` VARCHAR(255) NOT NULL,
    `app` VARCHAR(100) NOT NULL,
    `content` JSON NOT NULL
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `t_jdk_card` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT 'ID',
    `code` VARCHAR(32) NOT NULL UNIQUE COMMENT '兑换码',
    `quota` INT NOT NULL COMMENT '包含的搜索次数' DEFAULT 0,
    `is_used` BOOL NOT NULL COMMENT '是否已使用' DEFAULT 0,
    `used_appid` VARCHAR(64) COMMENT '使用的应用ID',
    `used_at` BIGINT COMMENT '使用时间',
    `created_at` BIGINT NOT NULL COMMENT '创建时间',
    `expired_at` BIGINT COMMENT '过期时间',
    KEY `idx_t_jdk_card_code_bcf79c` (`code`)
) CHARACTER SET utf8mb4 COMMENT='JDK 兑换卡表';
CREATE TABLE IF NOT EXISTS `t_site` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '站点ID',
    `appid` VARCHAR(64) NOT NULL UNIQUE COMMENT '应用ID',
    `secret_key` VARCHAR(64) NOT NULL COMMENT '应用密钥',
    `name` VARCHAR(64) NOT NULL COMMENT '站点名称',
    `is_active` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
    `search_quota` INT NOT NULL COMMENT '搜索次数配额，null 表示不限制，默认为 100' DEFAULT 100,
    `remark` VARCHAR(255) COMMENT '备注',
    `updated_at` BIGINT NOT NULL COMMENT '更新时间',
    KEY `idx_t_site_appid_725fee` (`appid`)
) CHARACTER SET utf8mb4 COMMENT='站点应用表';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """


MODELS_STATE = (
    "eJztmW1vm0gQx7+Kxaue1PZ4XOBUVYrTuzZpm0h9OJ2URmiBxabG4MDSS1Tlu98MGPNgsM"
    "FNbLe6N9SZnYHdH7P/2aHfhXnksiB5fsJi35kKf4y+CyGdM/jRGHk6EuhiUdrRwKkdZK60"
    "9LETHlOHg9WjQcLA5LLEif0F96MQrGEaBGiMHHD0w0lpSkP/JmUWjyaMT1kMA1fXYPZDl9"
    "2ypPhzMbM8nwVubaq+i8/O7Ba/W2S2s5D/lTni02zLiYJ0HpbOizs+jcKVtx9ytE5YyGLK"
    "Gd6exylOH2e3XGexonympUs+xUqMyzyaBryy3J4MnChEfjCbJFvgBJ/yTJZUXTUUohrgks"
    "1kZdHv8+WVa88DMwIXn4T7bJxymntkGEtu31ic4JTW4J1OadxOrxLSQAgTbyIsgG1iWBhK"
    "iGXiPBDFOb21AhZOOCa4rGkbmP198uH0zcmHJ+D1G64mgmTOc/xiOSTnYwi2BIlbYwDEpf"
    "vPCVASxR4AwasTYDZWBwhP5Czfg3WI5x8vL9ohVkIaID+HsMAr13f401HgJ/z6OLFuoIir"
    "xknPk+QmqMJ78v7knybX03eX44xClPBJnN0lu8EYGKNkerPK5keDTZ3ZvzR2rbWRSI66fN"
    "eH5vK8aaEhnWSscMW4vmUROXdnp3CTtvpSDG0sMNz66s4sp/DbVmSE81dvR19STdKkLylR"
    "iAy/FQK/DYMYQuPldDgLvUrSFbxUlwnXv0ppEs5eNfkcpjRlWAdIauH/MJq6K716GumGKP"
    "WkWVNYRe4hsIrcqa84VJfXmzTidEAirvy35+JDaanYSlMRNbiqsg00iaECWVVy4LcrA19i"
    "y8ha08V95mxlbydWmrCWDT6OooDRsGOTl1ENujaEPRbeDqEEfET2MsQErq4HWFVPB4uuyW"
    "tSObiOjS8v39Xq2PjsUyNrP78f/wnHhSyZwcnnrB02MrOgQrQJarcs1KN2Eofl1v/xg4FQ"
    "RVvks8ZMNbf0Vt6aVhC1h1YQtVMrcKgNdMtRbOxPOgWjErSTZDwWZKJ5kNem5qmDJMKUZU"
    "XRZVEhhqbqumaIK61YH9okGuOz15jKNfjrue3EDNEMpl6P259WCy+8NHSQ+IimfPp8eZR7"
    "EdC57dKXL4VWLZclG/Pdoz/La2G3Cz/e4bXU4w6+HwzP0QG5LnnHDP5IOpWPeQFaa1My+5"
    "YeJSl8tvYnIE/UNeEq2ma1BnR0KFX30e+jegRRKJ6RJAJXU1JdsHuE9W1f8tr46/QvVVTH"
    "0ssMPrX82IHlIbuZozudJAyKHrdm7G4I0HrUgb+7CfX9q9kOSrJKteMgnP07gG3hf3iqNU"
    "1VRVRC0+vbGz4yVej6oCL431rQbusWy7g99osridjULqryETaKCaOxM7WGfvJohu3vNC2J"
    "rd8+2r90FDXeNEy4ep7o4ONH+cEBE16B47XKMPlNool47FZI7ggW5uIRw6Yq+igUH32YTy"
    "cxm9N4NkRmyoiDd/GaKeKB2mE9834P/wGVLtydOsh63JF3kIR4+AVQs8X/GxmhvZG5/w8H"
    "TvWb"
)
