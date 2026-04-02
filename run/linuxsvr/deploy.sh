#!/usr/bin/env bash
# Build + deploy + generate nginx config
# Usage:
#   bash deploy.sh -s dev -p image_search -d lvyx.cc
#   bash deploy.sh          # interactive
#   bash deploy.sh ... --no-cache --no-nginx

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- args ----------
SITE=""
PROJECT=""
DOMAIN=""
HTTPS="true"
LISTEN="80"
NGINX_RELOAD_DIR="/home/applications/ng_container"
HOST_DATA=""
NO_CACHE=""
NO_NGINX=0
NO_MIGRATE=0
IMAGE_PORT=""
APP_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--site)       SITE="$2";             shift 2 ;;
        -p|--project)    PROJECT="$2";           shift 2 ;;
        -d|--domain)     DOMAIN="$2";            shift 2 ;;
        --https)         HTTPS="true";           shift ;;
        --no-https)      HTTPS="false";          shift ;;
        --listen)        LISTEN="$2";            shift 2 ;;
        --nginx-dir)     NGINX_RELOAD_DIR="$2";  shift 2 ;;
        --host-data)     HOST_DATA="$2";         shift 2 ;;
        --port)          IMAGE_PORT="$2";        shift 2 ;;
        --app-path)      APP_PATH="$2";          shift 2 ;;
        --no-cache)      NO_CACHE="--no-cache";  shift ;;
        --no-nginx)      NO_NGINX=1;             shift ;;
        --no-migrate)    NO_MIGRATE=1;           shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
 done

prompt() {
    local var_name="$1" prompt_text="$2" default="${3:-}"
    local value
    if [ -n "$default" ]; then
        read -r -p "$prompt_text [$default]: " value
        echo "${value:-$default}"
    else
        while true; do
            read -r -p "$prompt_text: " value
            [ -n "$value" ] && break
            echo "  required, please retry" >&2
        done
        echo "$value"
    fi
}

[ -z "$SITE" ]    && SITE=$(prompt SITE    "site (e.g. dev/prod)")
[ -z "$PROJECT" ] && PROJECT=$(prompt PROJECT "project" "image_search")
[ -z "$DOMAIN" ]  && DOMAIN=$(prompt DOMAIN  "root domain (blank to skip)" "__skip__")
[ "$DOMAIN" = "__skip__" ] && DOMAIN=""

[ -z "$HOST_DATA" ] && HOST_DATA="/data/$PROJECT"

echo ""
echo "========================================"
echo " site=$SITE  project=$PROJECT  domain=${DOMAIN:-<none>}"
echo "========================================"

# ---------- 1) build ----------
echo ""
echo "[1/5] Build image..."
BUILD_ARGS=(-s "$SITE" -p "$PROJECT")
[ -n "$DOMAIN" ] && BUILD_ARGS+=(-d "$DOMAIN")
[ -n "$NO_CACHE" ] && BUILD_ARGS+=("$NO_CACHE")
[ -n "$HOST_DATA" ] && BUILD_ARGS+=("--data-dir" "$HOST_DATA")
[ -n "$IMAGE_PORT" ] && BUILD_ARGS+=("--port" "$IMAGE_PORT")
[ -n "$APP_PATH" ] && BUILD_ARGS+=("--app-path" "$APP_PATH")

python3 "$SCRIPT_DIR/build_image.py" "${BUILD_ARGS[@]}"

# ---------- latest bundle ----------
BUILD_DIR="/data/build"
LATEST_DIR=$(ls -td "$BUILD_DIR"/[0-9]* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No build output: $BUILD_DIR" >&2
    exit 1
fi
echo "Bundle: $LATEST_DIR"

# ---------- 2) init deploy ----------
echo ""
echo "[2/5] Init deploy..."
python3 "$LATEST_DIR/init.py"

ENV_FILE="$LATEST_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    IMAGE_PORT=$(grep '^IMAGE_PORT=' "$ENV_FILE" | head -1 | cut -d= -f2)
fi
IMAGE_PORT="${IMAGE_PORT:-27001}"
CONTAINER_NAME="${PROJECT}_${IMAGE_PORT}"

# ---------- 3) migrate ----------
echo ""
if [ "$NO_MIGRATE" -eq 0 ]; then
    echo "[3/5] Run migrate..."
    docker exec "$CONTAINER_NAME" python main.py migrate
else
    echo "[3/5] Skip migrate (--no-migrate)"
fi

# ---------- 4) nginx config ----------
echo ""
echo "[4/5] Generate nginx config..."
NGINX_ARGS=("--listen" "$LISTEN")
[ "$HTTPS" = "true" ] && NGINX_ARGS+=("--https") || NGINX_ARGS+=("--no-https")
python3 "$LATEST_DIR/register_nginx.py" "${NGINX_ARGS[@]}"

if [ "$NO_NGINX" -eq 0 ]; then
    if [ -f "$NGINX_RELOAD_DIR/reload.sh" ]; then
        echo "Reload nginx..."
        bash "$NGINX_RELOAD_DIR/reload.sh"
    else
        echo "Reload script not found: $NGINX_RELOAD_DIR/reload.sh" >&2
    fi
else
    echo "Skip nginx reload (--no-nginx)"
fi

# ---------- 5) save build.json ----------
SAVED_CONFIG="$HOST_DATA/$SITE/build.json"
python3 - "$SAVED_CONFIG" "$SITE" "$PROJECT" "$DOMAIN" "$HTTPS" "$LISTEN" "$NGINX_RELOAD_DIR" "$HOST_DATA" "$IMAGE_PORT" "$APP_PATH" <<'PYEOF'
import sys, json
path, site, project, domain, https, listen, nginx_dir, host_data, image_port, app_path = sys.argv[1:]

data = {
    "site": site,
    "project": project,
    "domain": domain,
    "https": https == "true",
    "listen": int(listen),
    "nginx_reload_dir": nginx_dir,
    "host_data": host_data,
    "image_port": image_port,
    "app_path": app_path,
}
json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print("Saved config: {}".format(path))
PYEOF

echo ""
echo "======== Deploy done ========"
