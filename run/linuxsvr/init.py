#!/usr/bin/env python3
"""Init deploy: prepare data dirs, load image, start container.

Run inside the bundle directory (contains .env / docker-compose.yaml / <project>.tar / config.py).
"""
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict

REQUIRED_ENV = ["IMAGE_TAG", "IMAGE_SITE"]
DEFAULT_PROJECT = "image_search"
DEFAULT_DATA_ROOT = "/data"
DEFAULT_APP_DIRNAME = "app"
DEFAULT_IMAGE_PORT = "27001"


def parse_env(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f".env not found: {path}")
    env: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    for key in REQUIRED_ENV:
        if key not in env:
            raise ValueError(f".env missing {key}")
    return env


def run(cmd):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def validate_site(site: str) -> str:
    if "/" in site or "\\" in site:
        raise ValueError("IMAGE_SITE cannot contain path separators")
    return site


def validate_project(project: str) -> str:
    if "/" in project or "\\" in project:
        raise ValueError("IMAGE_PROJECT cannot contain path separators")
    return project


def resolve_host_data(env: Dict[str, str], project: str) -> Path:
    host_data = env.get("HOST_DATA")
    if host_data:
        return Path(host_data)
    return Path(DEFAULT_DATA_ROOT) / project


def resolve_app_path(env: Dict[str, str], host_data: Path, site: str) -> Path:
    app_path = env.get("IMAGE_APP_PATH")
    if app_path:
        return Path(app_path)
    return host_data / site / DEFAULT_APP_DIRNAME


def ensure_paths(host_data: Path, site: str) -> Path:
    base = host_data / site
    logs = base / "logs"
    oss = base / "oss"
    base.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    oss.mkdir(parents=True, exist_ok=True)
    return base


def copy_oss_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        try:
            next(dst.iterdir())
            return
        except StopIteration:
            shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_config(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"source config.py not found: {src}")
    shutil.copy2(src, dst)


def ensure_image(image: str, tar_path: Path) -> None:
    inspect = subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if inspect.returncode == 0:
        return
    if not tar_path.exists():
        raise FileNotFoundError(f"image not found and tar missing: {tar_path}")
    run(["docker", "load", "-i", str(tar_path)])


def container_exists(name: str) -> bool:
    proc = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return bool(proc.stdout.strip())


def resolve_tar_file(here: Path, project: str) -> Path:
    primary = here / f"{project}.tar"
    if primary.exists():
        return primary
    legacy = here / "image_search.tar"
    if legacy.exists():
        return legacy
    return primary


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    base = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(base)):
            raise RuntimeError("tar path traversal detected")
    tar.extractall(path)


def ensure_app_source(here: Path, app_path: Path) -> None:
    if app_path.exists():
        try:
            next(app_path.iterdir())
            return
        except StopIteration:
            shutil.rmtree(app_path)
    app_path.mkdir(parents=True, exist_ok=True)
    src_tar = here / "app_src.tar.gz"
    if src_tar.exists():
        with tarfile.open(src_tar, "r:gz") as tar:
            safe_extract(tar, app_path)
        return
    src_dir = here / "app"
    if src_dir.exists():
        shutil.copytree(src_dir, app_path)
        return
    print("Warning: app_src.tar.gz or app dir not found, skipped code init")


def main() -> None:
    here = Path(__file__).resolve().parent
    env_file = here / ".env"
    compose_file = here / "docker-compose.yaml"
    config_src = here / "config.py"

    env = parse_env(env_file)
    project = validate_project(env.get("IMAGE_PROJECT", DEFAULT_PROJECT))
    site = validate_site(env["IMAGE_SITE"])
    image_tag = env["IMAGE_TAG"]
    image_port = env.get("IMAGE_PORT", DEFAULT_IMAGE_PORT)

    host_data = resolve_host_data(env, project)
    app_path = resolve_app_path(env, host_data, site)

    image = f"{project}:{image_tag}"
    container_name = f"{project}_{image_port}"
    tar_file = resolve_tar_file(here, project)

    ensure_image(image, tar_file)

    if container_exists(container_name):
        subprocess.run([
            "docker", "compose",
            "--env-file", str(env_file),
            "-f", str(compose_file),
            "down",
        ], check=False)

    data_base = ensure_paths(host_data, site)

    ensure_app_source(here, app_path)

    copy_config(config_src, app_path / "config.py")

    oss_src = here / "oss"
    oss_dst = data_base / "oss"
    copy_oss_dir(oss_src, oss_dst)

    run([
        "docker", "compose",
        "--env-file", str(env_file),
        "-f", str(compose_file),
        "up", "-d",
    ])

    print("Done: data=%s, project=%s, site=%s, image=%s" % (data_base, project, site, image))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed: {exc}", file=sys.stderr)
        sys.exit(1)
