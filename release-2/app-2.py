import os
import io
import json
import shutil
import numpy as np
from PIL import Image
from datetime import datetime
from typing import List, Optional

import torch
import clip
import faiss
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

app = FastAPI(title="Image Search API")

# =============================
#  路径与初始化
# =============================
GALLERY_DIR = "gallery"
DELETED_DIR = "deleted"
CHUNKS_DIR = "chunks"

os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(DELETED_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

MAX_FEATURES_PER_CHUNK = 2000  # 每个分块存多少特征，可根据显存调整

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# =============================
#  特征提取函数
# =============================
def extract_feature(image: Image.Image) -> np.ndarray:
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(image)
    feature /= feature.norm(dim=-1, keepdim=True)
    return feature.cpu().numpy().astype("float32")


# =============================
#  分块文件操作函数
# =============================
def get_chunk_path(index: int):
    return os.path.join(CHUNKS_DIR, f"features_part_{index}.npy")

def get_info_path(index: int):
    return os.path.join(CHUNKS_DIR, f"filenames_part_{index}.json")

def get_current_chunk_index():
    existing = [
        int(f.split("_")[2].split(".")[0])
        for f in os.listdir(CHUNKS_DIR)
        if f.startswith("features_part_")
    ]
    return max(existing) if existing else 0

def append_to_chunk(new_features, new_files_info):
    idx = get_current_chunk_index()
    feature_path = get_chunk_path(idx)
    info_path = get_info_path(idx)

    if os.path.exists(feature_path):
        features = np.load(feature_path)
        with open(info_path, "r", encoding="utf-8") as f:
            infos = json.load(f)
    else:
        features = np.empty((0, new_features.shape[1]), dtype="float32")
        infos = []

    # 当前块容量满则切新块
    if len(features) + len(new_features) > MAX_FEATURES_PER_CHUNK:
        idx += 1
        feature_path = get_chunk_path(idx)
        info_path = get_info_path(idx)
        features = np.empty((0, new_features.shape[1]), dtype="float32")
        infos = []

    features = np.vstack([features, new_features])
    infos.extend(new_files_info)

    np.save(feature_path, features)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)


# =============================
#  上传接口（支持批量）
# =============================
@app.post("/upload_gallery")
async def upload_gallery(files: List[UploadFile] = File(...)):
    new_features, new_files_info = [], []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            feature = extract_feature(image)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_name = f"{timestamp}_{file.filename}"
            save_path = os.path.join(GALLERY_DIR, new_name)
            image.save(save_path)

            new_features.append(feature)
            new_files_info.append({
                "file": new_name,
                "original_name": file.filename,
                "upload_time": timestamp
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    new_features = np.vstack(new_features)
    append_to_chunk(new_features, new_files_info)

    return {"message": f"Uploaded {len(files)} images successfully."}


# =============================
#  删除接口（软删除）
# =============================
@app.delete("/delete_gallery")
async def delete_gallery(filename: str):
    src = os.path.join(GALLERY_DIR, filename)
    dst = os.path.join(DELETED_DIR, filename)

    if not os.path.exists(src):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    shutil.move(src, dst)
    return {"message": f"File moved to deleted folder: {filename}"}


# =============================
#  搜索接口（以图搜图）
# =============================
@app.post("/search_gallery")
async def search_gallery(file: UploadFile = File(...), top_k: int = 5):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    query_feature = extract_feature(image)

    all_features, all_infos = [], []
    for f in os.listdir(CHUNKS_DIR):
        if f.startswith("features_part_") and f.endswith(".npy"):
            idx = int(f.split("_")[2].split(".")[0])
            features = np.load(get_chunk_path(idx))
            with open(get_info_path(idx), "r", encoding="utf-8") as j:
                infos = json.load(j)
            all_features.append(features)
            all_infos.extend(infos)

    if not all_features:
        return {"message": "No indexed features found."}

    all_features = np.vstack(all_features)
    index = faiss.IndexFlatIP(all_features.shape[1])
    index.add(all_features)

    scores, indices = index.search(query_feature, top_k)
    results = [all_infos[i] for i in indices[0]]

    return {"results": results, "scores": scores.tolist()}


# =============================
#  新增：分页查询接口
# =============================
@app.get("/list_gallery")
async def list_gallery(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: Optional[str] = Query(None, description="模糊匹配文件名")
):
    """分页列出图库中的图片信息"""
    all_infos = []
    # 从每个分块加载文件信息
    for f in sorted(os.listdir(CHUNKS_DIR)):
        if f.startswith("filenames_part_") and f.endswith(".json"):
            with open(os.path.join(CHUNKS_DIR, f), "r", encoding="utf-8") as fp:
                try:
                    infos = json.load(fp)
                    all_infos.extend(infos)
                except Exception as e:
                    print(f"[WARN] 跳过损坏的文件 {f}: {e}")

    # 按上传时间降序排列（如果有 upload_time 字段）
    all_infos.sort(key=lambda x: x.get("upload_time", ""), reverse=True)

    # 过滤关键字
    if keyword:
        keyword = keyword.lower()
        all_infos = [
            info for info in all_infos
            if keyword in info.get("original_name", "").lower()
            or keyword in info.get("file", "").lower()
        ]

    total = len(all_infos)
    start = (page - 1) * page_size
    end = start + page_size
    results = all_infos[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": results,
    }


# =============================
#  重建索引接口
# =============================
@app.post("/rebuild_index")
async def rebuild_index():
    all_files = os.listdir(GALLERY_DIR)
    if not all_files:
        return {"message": "No images found in gallery."}

    # 清空旧索引
    for f in os.listdir(CHUNKS_DIR):
        os.remove(os.path.join(CHUNKS_DIR, f))

    new_features, new_files_info = [], []
    for file in all_files:
        try:
            path = os.path.join(GALLERY_DIR, file)
            image = Image.open(path).convert("RGB")
            feature = extract_feature(image)
            new_features.append(feature)
            new_files_info.append({"file": file, "original_name": file})
        except Exception as e:
            print(f"Skip {file}: {e}")

        if len(new_features) >= MAX_FEATURES_PER_CHUNK:
            append_to_chunk(np.vstack(new_features), new_files_info)
            new_features, new_files_info = [], []

    if new_features:
        append_to_chunk(np.vstack(new_features), new_files_info)

    return {"message": "Index rebuilt successfully."}


# =============================
#  状态接口
# =============================
@app.get("/status")
async def status():
    chunks = [
        f for f in os.listdir(CHUNKS_DIR) if f.startswith("features_part_")
    ]
    gallery_count = len(os.listdir(GALLERY_DIR))
    deleted_count = len(os.listdir(DELETED_DIR))
    return {
        "gallery_count": gallery_count,
        "deleted_count": deleted_count,
        "chunks": chunks,
    }
