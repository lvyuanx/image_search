# app.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from image_search_engine import ImageSearchEngine

engine = ImageSearchEngine()
app = FastAPI(title="以图搜图服务", description="基于 CLIP + FAISS 的可复用图像搜索引擎")

@app.post("/init_gallery")
def init_gallery():
    count = engine.rebuild_gallery()
    return {"status": "ok", "count": count}

@app.post("/upload_gallery")
async def upload_gallery(files: list[UploadFile] = File(...)):
    file_data = [(f.filename, await f.read()) for f in files]
    added = engine.add_images(file_data)
    return {"status": "ok", "new_files": added}

@app.post("/search_image")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    img_bytes = await file.read()
    results = engine.search_image(img_bytes, top_k)
    return {"results": results}

@app.get("/search_name")
def search_name(q: str = Query(...), limit: int = 10):
    return {"results": engine.search_name(q, limit)}

@app.get("/list_gallery")
def list_gallery(page: int = 1, page_size: int = 20, keyword: str | None = None):
    return engine.list_gallery(page, page_size, keyword)

@app.post("/delete_image")
def delete_image(stored_name: str = Query(...)):
    return engine.delete_image(stored_name)

@app.get("/")
def root():
    return {"message": "以图搜图引擎已启动。"}


