

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEBUG = True

INSERTAPPS = [
    "image_search",
]


HOST = "0.0.0.0"
PORT = 8000
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
