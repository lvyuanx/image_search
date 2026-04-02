#!/usr/bin/env bash
# Update existing container: build new image -> replace container -> (optional) migrate
# Usage:
#   bash update.sh -s dev -p image_search       # manual
#   bash update.sh                              # read /data/<project>/<site>/build.json
#   bash update.sh ... --no-cache --no-nginx

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- args ----------
SITE=""
PROJECT=""
DOMAIN=""
HOST_DATA=""
NGINX_RELOAD_DIR=""
NO_CACHE=""
NO_NGINX=0
NO_MIGRATE=0
IMAGE_PORT=""
APP_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--site)      SITE="$2";            shift 2 ;;
        -p|--project)   PROJECT="$2";         shift 2 ;;
        -d|--domain)    DOMAIN="$2";          shift 2 ;;
        --host-data)    HOST_DATA="$2";       shift 2 ;;
        --nginx-dir)    NGINX_RELOAD_DIR="$2"; shift 2 ;;
        --port)         IMAGE_PORT="$2";      shift 2 ;;
        --app-path)     APP_PATH="$2";        shift 2 ;;
        --no-cache)     NO_CACHE="--no-cache"; shift ;;
        --no-nginx)     NO_NGINX=1;           shift ;;
        --no-migrate)   NO_MIGRATE=1;         shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
 done

# ---------- load build.json if present ----------
load_saved_config() {
    local saved="$1"
    if [ ! -f "$saved" ]; then
        return 1
    fi
    local val
    val=$(python3 -c "import json; d=json.load(open('$saved')); print(d.get('$2',''))" 2>/dev/null) && echo "$val"
}

# prompt if missing
if [ -z "$SITE" ] || [ -z "$PROJECT" ]; then
    echo "Missing -s/-p, prompt for build.json location..."
    [ -z "$PROJECT" ] && read -r -p "project [image_search]: " PROJECT && PROJECT="${PROJECT:-image_search}"
    [ -z "$SITE" ]    && { read -r -p "site: " SITE; [ -z "$SITE" ] && { echo "site required" >&2; exit 1; }; }
fi

[ -z "$HOST_DATA" ] && HOST_DATA="/data/$PROJECT"
SAVED_CONFIG="$HOST_DATA/$SITE/build.json"

if [ -f "$SAVED_CONFIG" ]; then
    echo "Loaded config: $SAVED_CONFIG"
    read_saved() { python3 -c "import json; d=json.load(open('$SAVED_CONFIG')); print(d.get('$1',''))"; }
    [ -z "$DOMAIN" ]           && DOMAIN=$(read_saved domain)
    [ -z "$NGINX_RELOAD_DIR" ] && NGINX_RELOAD_DIR=$(read_saved nginx_reload_dir)
    [ -z "$HOST_DATA" ]        && HOST_DATA=$(read_saved host_data)
    [ -z "$IMAGE_PORT" ]       && IMAGE_PORT=$(read_saved image_port)
    [ -z "$APP_PATH" ]         && APP_PATH=$(read_saved app_path)
else
    echo "No saved config at $SAVED_CONFIG, using defaults" >&2
fi

[ -z "$NGINX_RELOAD_DIR" ] && NGINX_RELOAD_DIR="/home/applications/ng_container"

echo ""
echo "========================================"
echo " Update: site=$SITE  project=$PROJECT  domain=${DOMAIN:-<none>}"
echo "========================================"

# ---------- 1) build ----------
echo ""
echo "[1/3] Build image..."
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

# ---------- 2) replace container ----------
echo ""
echo "[2/3] Replace container..."
python3 "$LATEST_DIR/init.py"

ENV_FILE="$LATEST_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    IMAGE_PORT=$(grep '^IMAGE_PORT=' "$ENV_FILE" | head -1 | cut -d= -f2)
fi
IMAGE_PORT="${IMAGE_PORT:-27001}"
CONTAINER_NAME="${PROJECT}_${IMAGE_PORT}"

# ---------- 3) migrate ----------
if [ "$NO_MIGRATE" -eq 0 ]; then
    echo ""
    echo "[3/3] Run migrate..."
    docker exec "$CONTAINER_NAME" python main.py migrate
else
    echo ""
    echo "[3/3] Skip migrate (--no-migrate)"
fi

# ---------- nginx reload ----------
if [ "$NO_NGINX" -eq 0 ] && [ -f "$NGINX_RELOAD_DIR/reload.sh" ]; then
    echo ""
    echo "Reload nginx..."
    bash "$NGINX_RELOAD_DIR/reload.sh"
fi

echo ""
echo "======== Update done ========"
