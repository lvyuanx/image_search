#!/usr/bin/env python3
"""Generate nginx site config.

Usage (run inside bundle dir):
  python register_nginx.py --listen 80
  python register_nginx.py --server-name dev-image_search.lvyx.cc --listen 80
  python register_nginx.py --server-name dev-image_search.lvyx.cc --https

Reads .env for IMAGE_PROJECT / IMAGE_SITE / HOST_DATA / IMAGE_DOMAIN / SERVER_NAME / IMAGE_PORT.
- listen port default 80
- optional SSL 443 config (default on, use --no-https to disable)
- proxy to container <project>_<IMAGE_PORT>:8001
- static from HOST_DATA/<site>/oss
"""
import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

REQUIRED_ENV = ["IMAGE_SITE"]
DEFAULT_PROJECT = "image_search"
DEFAULT_DATA_ROOT = "/data"
DEFAULT_IMAGE_PORT = "27001"
DEFAULT_SSL_CERT = "/etc/nginx/ssl/fullchain.pem"
DEFAULT_SSL_KEY = "/etc/nginx/ssl/privkey.pem"
NGINX_CONF_DIR = Path("/home/applications/ng_container/nginx/conf.d")

LOCATION_BLOCK = """    location /static/ {{
        alias {host_data}/{site}/oss/static/;
        autoindex off;
        expires 30d;
        access_log off;
        add_header Cache-Control \"public\";
    }}

    location /media/ {{
        alias {host_data}/{site}/oss/media/;
        autoindex off;
        expires 30d;
        access_log off;
        add_header Cache-Control \"public\";
    }}

    location / {{
        proxy_pass http://{container_name}:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60;
        proxy_send_timeout 60;
        proxy_read_timeout 60;
    }}
"""

HTTP_TEMPLATE = """server {{
    listen {listen_port};
    listen [::]:{listen_port};

    server_name {server_name};

{locations}
}}
"""

HTTPS_TEMPLATE = """server {{
    listen {listen_port} ssl;
    listen [::]:{listen_port} ssl;

    server_name {server_name};

    ssl_certificate     {ssl_cert};
    ssl_certificate_key {ssl_key};

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

{locations}
}}
"""


def parse_env(path: Path) -> Dict[str, str]:
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


def validate_site(site: str) -> str:
    if "/" in site or "\\" in site:
        raise ValueError("IMAGE_SITE cannot contain path separators")
    return site


def validate_project(project: str) -> str:
    if "/" in project or "\\" in project:
        raise ValueError("IMAGE_PROJECT cannot contain path separators")
    return project


def normalize_domain(domain: str) -> str:
    return domain.strip().lstrip(".")


def resolve_host_data(env: Dict[str, str], project: str) -> Path:
    host_data = env.get("HOST_DATA")
    if host_data:
        return Path(host_data)
    return Path(DEFAULT_DATA_ROOT) / project


def build_server_name(site: str, project: str, server_name: Optional[str], root_domain: Optional[str]) -> str:
    if server_name:
        return server_name
    if root_domain:
        slug = f"{site}-{project}".replace("_", "-")
        return f"{slug}.{normalize_domain(root_domain)}"
    return "_"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate nginx site config")
    parser.add_argument("-p", "--listen", type=int, default=80, help="nginx listen port")
    parser.add_argument("--https", dest="https", action="store_true", default=True, help="include SSL 443 config")
    parser.add_argument("--no-https", dest="https", action="store_false", help="only generate HTTP config")
    parser.add_argument("--server-name", "--domain", dest="server_name", default=None, help="server_name override")
    parser.add_argument("--root-domain", default=None, help="root domain for <site>-<project>.<domain>")
    parser.add_argument("--ssl-cert", default=None, help=f"SSL fullchain path, default {DEFAULT_SSL_CERT}")
    parser.add_argument("--ssl-key", default=None, help=f"SSL key path, default {DEFAULT_SSL_KEY}")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    env_file = here / ".env"
    if not env_file.exists():
        raise FileNotFoundError(f".env not found: {env_file}")

    env = parse_env(env_file)
    project = validate_project(env.get("IMAGE_PROJECT", DEFAULT_PROJECT))
    site = validate_site(env["IMAGE_SITE"])
    host_data = resolve_host_data(env, project)
    image_port = env.get("IMAGE_PORT", DEFAULT_IMAGE_PORT)

    root_domain = args.root_domain or env.get("IMAGE_DOMAIN")
    server_name = build_server_name(site, project, args.server_name or env.get("SERVER_NAME"), root_domain)

    container_name = f"{project}_{image_port}"

    locations = LOCATION_BLOCK.format(site=site, host_data=host_data, container_name=container_name)

    blocks = [HTTP_TEMPLATE.format(listen_port=args.listen, server_name=server_name, locations=locations)]
    if args.https:
        ssl_cert = args.ssl_cert or env.get("SSL_CERT") or DEFAULT_SSL_CERT
        ssl_key = args.ssl_key or env.get("SSL_KEY") or DEFAULT_SSL_KEY
        blocks.append(HTTPS_TEMPLATE.format(listen_port=443, server_name=server_name, ssl_cert=ssl_cert, ssl_key=ssl_key, locations=locations))

    conf_text = "\n\n".join(blocks)

    out_path = here / f"{project}_{site}.conf"
    out_path.write_text(conf_text, encoding="utf-8")
    print(f"Generated {out_path}")

    if NGINX_CONF_DIR.exists():
        dest = NGINX_CONF_DIR / out_path.name
        if dest.exists():
            print(f"Exists, skipped {dest}")
        else:
            try:
                shutil.copy2(out_path, dest)
                print(f"Copied to: {dest}")
            except PermissionError:
                print(f"Permission denied: {dest}", file=sys.stderr)
    else:
        print(f"nginx conf dir not found: {NGINX_CONF_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
