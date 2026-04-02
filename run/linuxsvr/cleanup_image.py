#!/usr/bin/env python3
"""Force remove image: remove dependent containers first.

Examples:
  python cleanup_image.py image_search:latest
  python cleanup_image.py image_search:202403 -v  # also remove anonymous volumes
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def get_containers(image: str):
    cmd = ["docker", "ps", "-a", "-q", "--filter", f"ancestor={image}"]
    print("$", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Force remove image and dependent containers")
    parser.add_argument("image", help="image:tag, e.g. image_search:latest")
    parser.add_argument("-v", "--prune-volumes", action="store_true", help="also remove anonymous volumes")
    args = parser.parse_args()

    container_ids = get_containers(args.image)
    if container_ids:
        rm_cmd = ["docker", "rm", "-f"]
        if args.prune_volumes:
            rm_cmd.append("-v")
        rm_cmd += container_ids
        run(rm_cmd)
    else:
        print("No containers use this image")

    run(["docker", "rmi", "-f", args.image])
    print("Image removed")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
