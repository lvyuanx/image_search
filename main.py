import asyncio
import argparse
import logging.config
import sys

from dotenv import load_dotenv

import settings

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main():
    
    logging.config.dictConfig(settings.LOGGING)
    
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="Project management commands",
    )

    sub = parser.add_subparsers(dest="command")

    # runserver
    sub.add_parser("run", help="Run FastAPI application")

    # makemigrations
    sub.add_parser("makemigrations", help="Generate database migrations")

    # migrate
    sub.add_parser("migrate", help="Apply database migrations")

    args = parser.parse_args()

    load_dotenv()

    if args.command == "run":
        from app import run
        asyncio.run(run())

    elif args.command == "makemigrations":
        from common.orm import makemigrations
        makemigrations()

    elif args.command == "migrate":
        from common.orm import migrate
        migrate()



if __name__ == "__main__":
    main()
