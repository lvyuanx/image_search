# app.py
import os
import io
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List

# =======================
# 1. 初始化 CLIP 模型
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"[INFO] Using device: {device}")

# =======================
# 2. 基本路径配置
# =======================
GALLERY_DIR = "gallery"
FEATURES_PATH = "data/features.npy"
FILENAMES_PATH = "data/filenames.txt"

os.makedirs("data", exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)

# =======================
# 3. 工具函数
# =======================
def extract_feature(img: Image.Image) -> np.ndarray:
    """提取单张图片特征"""
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(image)
    feature /= feature.norm(dim=-1, keepdim=True)
    return feature.cpu().numpy().astype("float32")

def build_index():
    """从 gallery 文件夹构建图库索引"""
    features, filenames = [], []
    for fname in os.listdir(GALLERY_DIR):
        if fname.lower().endswith(('jpg', 'jpeg', 'png')):
            path = os.path.join(GALLERY_DIR, fname)
            try:
                img = Image.open(path).convert("RGB")
                features.append(extract_feature(img))
                filenames.append(fname)
            except Exception as e:
                print(f"[WARN] 跳过 {fname}: {e}")

    if not features:
        raise RuntimeError("图库为空，请在 gallery/ 中放入图片。")

    features = np.vstack(features)
    np.save(FEATURES_PATH, features)
    with open(FILENAMES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(filenames))

    print(f"[INFO] 已建立索引，共 {len(filenames)} 张图片。")
    return features, filenames

def load_index():
    """加载现有索引"""
    if not (os.path.exists(FEATURES_PATH) and os.path.exists(FILENAMES_PATH)):
        return None, None, None

    features = np.load(FEATURES_PATH)
    with open(FILENAMES_PATH, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f.readlines()]
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    return features, filenames, index

# =======================
# 4. 初始化索引
# =======================
features, filenames, index = load_index()
if features is None:
    features, filenames = np.empty((0, 512), dtype="float32"), []
    index = faiss.IndexFlatIP(512)
    print("[INFO] 空索引已创建。")
else:
    print(f"[INFO] Gallery indexed: {len(filenames)} images.")

# =======================
# 5. FastAPI 服务
# =======================
app = FastAPI(title="以图搜图服务", description="支持图库初始化、上传、搜索的 CLIP+FAISS 示例")

@app.post("/init_gallery")
async def init_gallery():
    """初始化图库（重新构建索引）"""
    global features, filenames, index
    try:
        features, filenames = build_index()
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)
        return {"message": f"图库初始化成功，共 {len(filenames)} 张图片。"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/upload_gallery")
async def upload_gallery(files: List[UploadFile] = File(...)):
    """批量上传图片到图库，并更新索引"""
    global features, filenames, index
    uploaded = []

    for file in files:
        try:
            img_bytes = await file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            save_path = os.path.join(GALLERY_DIR, file.filename)
            img.save(save_path)
            # 提取特征并更新索引
            feat = extract_feature(img)
            index.add(feat)
            features = np.vstack([features, feat]) if features.size else feat
            filenames.append(file.filename)
            uploaded.append(file.filename)
        except Exception as e:
            print(f"[WARN] 上传失败 {file.filename}: {e}")

    # 保存更新后的特征
    np.save(FEATURES_PATH, features)
    with open(FILENAMES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(filenames))

    return {"message": f"上传成功 {len(uploaded)} 张图片", "uploaded": uploaded}

@app.post("/search")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    """上传图片进行以图搜图"""
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        query_vec = extract_feature(img)
        D, I = index.search(query_vec, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({
                "filename": filenames[idx],
                "score": float(score),
                "path": f"/gallery/{filenames[idx]}"
            })
        return {"results": results}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "以图搜图服务已启动！可使用 /init_gallery、/upload_gallery、/search 接口。"}
