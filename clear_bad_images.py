# clean_bad_images.py
import os
import shutil
from PIL import Image, UnidentifiedImageError

GALLERY_DIR = "gallery"
DELETED_DIR = "deleted"

os.makedirs(DELETED_DIR, exist_ok=True)

valid_exts = ('.jpg', '.jpeg', '.png')

bad_count = 0

for fname in os.listdir(GALLERY_DIR):
    if not fname.lower().endswith(valid_exts):
        continue
    path = os.path.join(GALLERY_DIR, fname)
    try:
        # 尝试打开图片
        img = Image.open(path)
        img.verify()  # 只验证图片，不加载
    except (UnidentifiedImageError, OSError) as e:
        print(f"[WARN] 无法打开或损坏的图片: {fname}, 移动到 deleted 文件夹")
        shutil.move(path, os.path.join(DELETED_DIR, fname))
        bad_count += 1

print(f"[INFO] 清理完成，共处理 {bad_count} 张损坏图片。")
