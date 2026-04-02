#!/usr/bin/env python3
"""Build image_search image and export an offline bundle.

Outputs:
  build/<TAG>/<project>.tar
  build/<TAG>/docker-compose.yaml
  build/<TAG>/.env   (IMAGE_PROJECT, IMAGE_SITE, IMAGE_TAG, HOST_DATA, LOG_PATH, IMAGE_APP_PATH, IMAGE_PORT, IMAGE_DOMAIN, SERVER_NAME)
  build/<TAG>/app_src.tar.gz  (source bundle by default)
  build/<TAG>/config.py
  build/<TAG>/oss/**
  build/<TAG>/init.py
  build/<TAG>/register_nginx.py

Features:
- -s/--site to set site key (writes to .env)
- -p/--project to set project name
- --domain to generate SERVER_NAME=<site>-<project>.<domain>
- --app-path to set host code dir, default /data/<project>/<site>/app
- --port to set host port, default 27001
- --no-src to skip source bundle
- Can be called from any directory (uses absolute paths)
"""
import argparse
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT_DIR / "docker" / "Dockerfile"
COMPOSE_FILE = ROOT_DIR / "docker" / "docker-compose.yaml"
SOURCE_CONFIG = ROOT_DIR / "config.py"
SOURCE_OSS_MEDIA = ROOT_DIR / "oss" / "media"
SOURCE_OSS_STATIC = ROOT_DIR / "oss" / "static"
INIT_TEMPLATE = ROOT_DIR / "run" / "linuxsvr" / "init.py"
NGINX_SCRIPT = ROOT_DIR / "run" / "linuxsvr" / "register_nginx.py"
DEFAULT_PROJECT = "image_search"
DEFAULT_BUILD_DIR = Path("/data/build")
DEFAULT_DATA_ROOT = "/data"
DEFAULT_IMAGE_PORT = "27001"
DEFAULT_APP_DIRNAME = "app"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "logs",
    "build",
}
EXCLUDE_FILES = {
    ".DS_Store",
}


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image_search image and offline bundle")
    parser.add_argument("-s", "--site", "-site", required=True, help="site key, e.g. dev")
    parser.add_argument("-p", "--project", "-project", default=DEFAULT_PROJECT, help=f"project name, default {DEFAULT_PROJECT}")
    parser.add_argument("-d", "--domain", "-domain", default=None, help="root domain, e.g. lvyx.cc")
    parser.add_argument("-t", "--tag", default=None, help="image tag / output dir name, default timestamp")
    parser.add_argument("--no-cache", action="store_true", help="build with --no-cache")
    parser.add_argument("-o", "--output", default=str(DEFAULT_BUILD_DIR), help="output root dir, default /data/build")
    parser.add_argument("--data-dir", default=None, help="host data root, default /data/<project>")
    parser.add_argument("--app-path", default=None, help="host code dir, default /data/<project>/<site>/app")
    parser.add_argument("--port", default=DEFAULT_IMAGE_PORT, help="host port, default 27001")
    parser.add_argument("--no-src", action="store_true", help="skip app_src.tar.gz")
    return parser.parse_args()


def validate_site(site: str) -> str:
    if "/" in site or "\\" in site:
        raise ValueError("site cannot contain path separators")
    return site


def validate_project(project: str) -> str:
    if "/" in project or "\\" in project:
        raise ValueError("project cannot contain path separators")
    return project


def normalize_domain(domain: str) -> str:
    return domain.strip().lstrip(".")


def resolve_host_data(project: str, data_dir: Optional[str]) -> str:
    if data_dir:
        return data_dir
    return f"{DEFAULT_DATA_ROOT}/{project}"


def resolve_app_path(host_data: str, site: str, app_path: Optional[str]) -> str:
    if app_path:
        return app_path
    return f"{host_data}/{site}/{DEFAULT_APP_DIRNAME}"


def build_server_name(site: str, project: str, domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    normalized = normalize_domain(domain)
    slug = f"{site}-{project}".replace("_", "-")
    return f"{slug}.{normalized}"


def write_env(env_path: Path, tag: str, project: str, site: str, data_dir: str, app_path: str, image_port: str, domain: Optional[str], server_name: Optional[str]) -> None:
    log_path = f"{data_dir}/{site}/logs"
    env_content = (
        f"IMAGE_TAG={tag}\n"
        f"IMAGE_PROJECT={project}\n"
        f"IMAGE_SITE={site}\n"
        f"HOST_DATA={data_dir}\n"
        f"LOG_PATH={log_path}\n"
        f"IMAGE_APP_PATH={app_path}\n"
        f"IMAGE_PORT={image_port}\n"
    )
    if domain:
        env_content += f"IMAGE_DOMAIN={normalize_domain(domain)}\n"
    if server_name:
        env_content += f"SERVER_NAME={server_name}\n"
    env_path.write_text(env_content, encoding="utf-8")


def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.name in EXCLUDE_FILES:
        return True
    return False


def build_source_bundle(dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    with tarfile.open(dst, "w:gz") as tar:
        for path in ROOT_DIR.rglob("*"):
            if should_exclude(path):
                continue
            if path.is_dir():
                continue
            arcname = path.relative_to(ROOT_DIR)
            tar.add(path, arcname=str(arcname))


def main() -> None:
    args = parse_args()
    site = validate_site(args.site)
    project = validate_project(args.project)
    tag = args.tag or datetime.now().strftime("%Y%m%d%H%M%S")

    host_data = resolve_host_data(project, args.data_dir)
    app_path = resolve_app_path(host_data, site, args.app_path)
    server_name = build_server_name(site, project, args.domain)

    build_cmd = ["docker", "build", "-f", str(DOCKERFILE), "-t", f"{project}:{tag}"]
    if args.no_cache:
        build_cmd.append("--no-cache")
    build_cmd.append(str(ROOT_DIR))
    run(build_cmd)

    base_dir = Path(args.output).expanduser()
    if not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    target_dir = base_dir / tag
    target_dir.mkdir(parents=True, exist_ok=True)

    tar_path = target_dir / f"{project}.tar"
    save_cmd = ["docker", "save", "-o", str(tar_path), f"{project}:{tag}"]
    run(save_cmd)

    shutil.copy2(COMPOSE_FILE, target_dir / "docker-compose.yaml")

    if SOURCE_CONFIG.exists():
        shutil.copy2(SOURCE_CONFIG, target_dir / "config.py")
    else:
        print(f"Warning: {SOURCE_CONFIG} not found, skipped config.py")

    oss_media_dest = target_dir / "oss" / "media"
    if SOURCE_OSS_MEDIA.exists():
        oss_media_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE_OSS_MEDIA, oss_media_dest)
    else:
        print(f"Warning: {SOURCE_OSS_MEDIA} not found, skipped oss/media")

    oss_static_dest = target_dir / "oss" / "static"
    if SOURCE_OSS_STATIC.exists():
        oss_static_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE_OSS_STATIC, oss_static_dest)
    else:
        print(f"Warning: {SOURCE_OSS_STATIC} not found, skipped oss/static")

    write_env(target_dir / ".env", tag, project, site, host_data, app_path, args.port, args.domain, server_name)

    if not args.no_src:
        build_source_bundle(target_dir / "app_src.tar.gz")

    if INIT_TEMPLATE.exists():
        shutil.copy2(INIT_TEMPLATE, target_dir / "init.py")
    else:
        print(f"Warning: {INIT_TEMPLATE} not found, skipped init.py")

    if NGINX_SCRIPT.exists():
        shutil.copy2(NGINX_SCRIPT, target_dir / "register_nginx.py")
    else:
        print(f"Warning: {NGINX_SCRIPT} not found, skipped register_nginx.py")

    print(
        "Done\n"
        f"Image: {project}:{tag}\n"
        f"Output: {target_dir}\n"
        "Contains: <project>.tar, docker-compose.yaml, .env, config.py, init.py, register_nginx.py, oss/**"
        + (", app_src.tar.gz" if not args.no_src else "")
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
