

import importlib
from pathlib import Path

# 用于标记无默认值
NONE = object()

# 尝试导入本地 config.py
try:
    local_config = importlib.import_module("config")
except ModuleNotFoundError:
    local_config = None

def merge_config(name, default=NONE):
    """
    从 config.py 获取配置，如果没有则使用 default
    如果 default 为 NONE 且 config.py 没有该配置，则报错
    """
    # 1️⃣ 优先从 config.py 获取
    if local_config and hasattr(local_config, name):
        return getattr(local_config, name)
    
    # 2️⃣ 再使用 default
    if default is not NONE:
        return default

    # 3️⃣ 都没有则报错
    raise ValueError(f"Config '{name}' not found in config.py and no default provided")


BASE_DIR = Path(__file__).resolve().parent
DEBUG = merge_config("DEBUG", True)

INSERTAPPS = [
    "image_search",
    "auth",
]


HOST = "0.0.0.0"
PORT = 8001

IMAGE_PREVIEW_URL_TEMPLATE = merge_config("IMAGE_PREVIEW_URL_TEMPLATE", "http://127.0.0.1:8701/api/image/preview/{group}/{name}")

WORKERS = 1

LOG_DIR = BASE_DIR / "logs"
LOGGING_BACK_COUNT = 10
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # 详细日志（用于文件，保持无颜色）
        "detailed": {
            "format": (
                "[%(asctime)s]"
                "[%(threadName)s:%(thread)d]"
                "[task_id:%(name)s]"
                "[%(filename)s:%(lineno)d]"
                "[%(funcName)s]"
                "[%(levelname)s] - %(message)s"
            )
        },
        # ---- 彩色 simple（console 专用）------
        "simple": {
            "()": "colorlog.ColoredFormatter",
            "format": (
                "%(log_color)s"
                "[%(asctime)s]"
                "[%(levelname)s]"
                "[%(filename)s:%(lineno)d]"
                "[%(funcName)s]"
                " - "
                "%(message)s"
                "%(reset)s"
            ),
            # 主颜色：levelname 和 message
            "log_colors": {
                "DEBUG":    "light_black",              # 深灰（存在感最低）
                "INFO":     "light_white",               # 清晰蓝
                "WARNING":  "light_yellow",              # 橙黄
                "ERROR":    "fg_196",              # 亮红
                "CRITICAL": "fg_15;bold",   # 白字红底
            },
        },
    },
    "handlers": {
        "all": {
            "level": "DEBUG",
            "class": "common.logging.multiprocess_time_handler.MultiprocessTimeHandler",
            "file_path": LOG_DIR,
            "suffix": "%Y-%m-%d-all",
            "formatter": "detailed",
            "backup_count": LOGGING_BACK_COUNT,
            "encoding": "utf-8",
        },
        "project": {
            "level": "DEBUG",
            "class": "common.logging.multiprocess_time_handler.MultiprocessTimeHandler",
            "file_path": LOG_DIR,
            "suffix": "%Y-%m-%d-project",
            "formatter": "detailed",
            "backup_count": LOGGING_BACK_COUNT,
            "encoding": "utf-8",
        },
        "error": {
            "level": "ERROR",
            "class": "common.logging.multiprocess_time_handler.MultiprocessTimeHandler",
            "file_path": LOG_DIR,
            "suffix": "%Y-%m-%d-error",
            "formatter": "detailed",
            "backup_count": LOGGING_BACK_COUNT,
            "encoding": "utf-8",
        },
        # ---- 控制台使用彩色 simple ----
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "loggers": {
        # root logger
        "": {
            "handlers": ["all", "console", "error"],
            "level": LOG_LEVEL,
            "propagate": True,
        },
        "project": {
            "handlers": ["project"],
            "level": LOG_LEVEL,
            "propagate": True,
        },
        "django.db.backends": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
        },
    },
}


# ----------------- 数据库配置 -----------------
DB_USERNAME = merge_config("DB_USERNAME", "root")
DB_PASSWORD = merge_config("DB_PASSWORD", "123456")
DB_HOST = merge_config("DB_HOST", "127.0.0.1")
DB_PORT = int(merge_config("DB_PORT", 3306))
DB_DATABASE = merge_config("DB_DATABASE", "image_search")
DB_CHARSET = merge_config("DB_CHARSET", "utf8mb4")
DB_TIMEZONE = merge_config("DB_TIMEZONE", "Asia/Shanghai")
DB_MAXSIZE = int(merge_config("DB_MAXSIZE", 20))
DB_MINSIZE = int(merge_config("DB_MINSIZE", 1))
DB_GENERATE_SCHEMAS = merge_config("DB_GENERATE_SCHEMAS", True)

# ----------------- 缓存配置 -----------------
REDIS_HOST = merge_config("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(merge_config("REDIS_PORT", 6379))
REDIS_PASSWORD = merge_config("REDIS_PASSWORD", "")
REDIS_DB = int(merge_config("REDIS_DB", 0))

# ----------------- 后台登录 -----------------
ADMIN_USERNAME = merge_config("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = merge_config("ADMIN_PASSWORD", "admin123")
SESSION_SECRET_KEY = merge_config("SESSION_SECRET_KEY", "change-me")




