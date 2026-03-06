import asyncio
import logging

logger = logging.getLogger(__name__)


def create_task_safe(coro, *, name=None):
    """
    创建安全的 asyncio task
    自动捕获异常并打印日志
    """
    task = asyncio.create_task(coro, name=name)

    def _done_callback(t: asyncio.Task):
        try:
            exc = t.exception()
            if exc:
                logger.exception("Async task failed", exc_info=exc)
        except asyncio.CancelledError:
            logger.info("Async task cancelled")

    task.add_done_callback(_done_callback)

    return task