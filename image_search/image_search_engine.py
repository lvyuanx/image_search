import heapq
import io
import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import clip
import faiss
import numpy as np
import torch
from PIL import Image

import settings


IMAGE_SEARCH_WORKSPACES = settings.BASE_DIR / "oss" / "media" / "groups"

GROUP_BACK = settings.BASE_DIR / "oss" / "media" / "back"

# 批量特征提取的 chunk 大小，防止 GPU OOM
_FEATURE_BATCH_SIZE = 32


# =========================================================
# 读写锁（允许多读单写）
# =========================================================

class _RWLock:
    """简单的读写锁：多读单写，写优先。"""

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()

    class _ReadCtx:
        def __init__(self, lock): self._lock = lock
        def __enter__(self): self._lock.acquire_read(); return self
        def __exit__(self, *_): self._lock.release_read()

    class _WriteCtx:
        def __init__(self, lock): self._lock = lock
        def __enter__(self): self._lock.acquire_write(); return self
        def __exit__(self, *_): self._lock.release_write()

    def read(self): return self._ReadCtx(self)
    def write(self): return self._WriteCtx(self)


# =========================================================
# CLIP 全局单例
# =========================================================

torch.backends.cudnn.benchmark = True


class ClipModelSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "ClipModelSingleton":
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            instance = object.__new__(cls)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[ImageSearch] Loading CLIP model on {device} ...")
            model, preprocess = clip.load("ViT-B/32", device=device)
            model.eval()
            instance.device = device
            instance.model = model
            instance.preprocess = preprocess
            cls._instance = instance
            print("[ImageSearch] CLIP model loaded.")
            return instance

    def __new__(cls):
        return cls.get()


# =========================================================
# 单个分组引擎
# =========================================================


class ImageSearchEngine:

    INDEX_FILE = "index.faiss"
    META_FILE = "meta.json"

    def __init__(self, gallery_dir: str, data_dir: str, deleted_dir: str):
        self.gallery_dir = str(gallery_dir)
        self.data_dir = str(data_dir)
        self.deleted_dir = str(deleted_dir)

        os.makedirs(self.gallery_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.deleted_dir, exist_ok=True)

        self._rwlock = _RWLock()

        clip_model = ClipModelSingleton.get()
        self.device = clip_model.device
        self.model = clip_model.model
        self.preprocess = clip_model.preprocess

        self.index_path = os.path.join(self.data_dir, self.INDEX_FILE)
        self.meta_path = os.path.join(self.data_dir, self.META_FILE)

        self.index = None  # 懒加载
        # meta 结构：List[{"id": int, "stored_name": str, "original_name": str, "upload_time": str}]
        self._meta: List[dict] = []
        # 辅助字典，O(1) 查找
        self._id_to_meta: Dict[int, dict] = {}
        self._stored_to_id: Dict[str, int] = {}
        self._original_to_ids: Dict[str, List[int]] = {}
        # 下一个可用 ID（单调递增，不复用）
        self._next_id: int = 0

        self._load_meta()

    # =====================================================
    # FAISS 懒加载（IndexIDMap 包装，支持按 ID 删除）
    # =====================================================

    def _ensure_index(self):
        if self.index is not None:
            return
        if os.path.exists(self.index_path):
            loaded = faiss.read_index(self.index_path)
            # 旧格式：IndexFlatIP，迁移为 IndexIDMap
            if not isinstance(loaded, faiss.IndexIDMap):
                print(f"[ImageSearch] migrating index to IndexIDMap: {self.index_path}")
                id_map = faiss.IndexIDMap(faiss.IndexFlatIP(512))
                if loaded.ntotal > 0:
                    vecs = faiss.rev_swig_ptr(loaded.get_xb(), loaded.ntotal * 512).reshape(loaded.ntotal, 512).copy()
                    ids = np.array([m["id"] for m in self._meta[:loaded.ntotal]], dtype=np.int64)
                    id_map.add_with_ids(vecs, ids)
                self.index = id_map
                self._save_index_unsafe()
            else:
                self.index = loaded
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(512))
            self._save_index_unsafe()

    def _save_index_unsafe(self):
        """调用方须持写锁。"""
        if self.index is not None:
            tmp = self.index_path + f".{os.getpid()}.tmp"
            faiss.write_index(self.index, tmp)
            os.replace(tmp, self.index_path)

    # =====================================================
    # meta 持久化（原子写入，紧凑 JSON）
    # =====================================================

    def _load_meta(self):
        if not os.path.exists(self.meta_path):
            return
        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容旧格式（直接是 list）
        if isinstance(data, list):
            for i, item in enumerate(data):
                item.setdefault("id", i)
            self._meta = data
            self._next_id = len(data)
        else:
            self._meta = data.get("items", [])
            self._next_id = data.get("next_id", 0)
        self._rebuild_aux_dicts()

    def _rebuild_aux_dicts(self):
        self._id_to_meta.clear()
        self._stored_to_id.clear()
        self._original_to_ids.clear()
        for item in self._meta:
            fid = item["id"]
            self._id_to_meta[fid] = item
            self._stored_to_id[item["stored_name"]] = fid
            self._original_to_ids.setdefault(item["original_name"], []).append(fid)

    def _save_meta_unsafe(self):
        """调用方须持写锁。原子写入。"""
        payload = json.dumps(
            {"next_id": self._next_id, "items": self._meta},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        tmp = self.meta_path + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, self.meta_path)

    # =====================================================
    # 特征提取（支持批处理）
    # =====================================================

    def _extract_features_batch(self, images: List[Image.Image]) -> np.ndarray:
        """批量提取特征，返回 (N, 512) float32 数组。"""
        all_features = []
        for i in range(0, len(images), _FEATURE_BATCH_SIZE):
            chunk = images[i: i + _FEATURE_BATCH_SIZE]
            tensors = torch.stack([self.preprocess(img) for img in chunk]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu().numpy().astype("float32"))
        return np.vstack(all_features)

    def _extract_feature(self, img: Image.Image) -> np.ndarray:
        return self._extract_features_batch([img])

    # =====================================================
    # 新增图片
    # =====================================================

    def add_images(self, files: List[tuple]) -> int:
        if not files:
            return 0

        # 1. 解码并写盘（锁外，纯 I/O）
        prepared = []
        for filename, data in files:
            new_name = f"{uuid.uuid4().hex[:8]}_{filename}"
            path = os.path.join(self.gallery_dir, new_name)
            with open(path, "wb") as f:
                f.write(data)
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:
                print(f"[ImageSearch] skip bad image: {filename} ({e})")
                os.remove(path)
                continue
            prepared.append((new_name, filename, img))

        if not prepared:
            return 0

        # 2. GPU 批量推理（锁外）
        images = [item[2] for item in prepared]
        features = self._extract_features_batch(images)  # (N, 512)

        # 3. 写入索引（持写锁，尽量短）
        with self._rwlock.write():
            self._ensure_index()

            new_ids = list(range(self._next_id, self._next_id + len(prepared)))
            self._next_id += len(prepared)

            ids_array = np.array(new_ids, dtype=np.int64)
            self.index.add_with_ids(features, ids_array)
            self._save_index_unsafe()

            now = datetime.now().strftime("%Y%m%d%H%M%S")
            for (stored_name, original_name, _), fid in zip(prepared, new_ids):
                item = {
                    "id": fid,
                    "stored_name": stored_name,
                    "original_name": original_name,
                    "upload_time": now,
                }
                self._meta.append(item)
                self._id_to_meta[fid] = item
                self._stored_to_id[stored_name] = fid
                self._original_to_ids.setdefault(original_name, []).append(fid)

            self._save_meta_unsafe()

        return len(prepared)

    # =====================================================
    # 搜索（持读锁）
    # =====================================================

    def search_image(self, img_bytes: bytes, top_k: int = 5) -> List[dict]:
        with self._rwlock.read():
            if not self._meta:
                return []

        # 特征提取在锁外
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        query_vec = self._extract_feature(img)

        with self._rwlock.read():
            self._ensure_index()
            real_top_k = min(top_k, len(self._meta))
            D, I = self.index.search(query_vec, real_top_k)

            results = []
            for fid, score in zip(I[0], D[0]):
                if fid < 0:
                    continue
                item = self._id_to_meta.get(fid)
                if item is None:
                    continue
                results.append({
                    "stored_name": item["stored_name"],
                    "original_name": item["original_name"],
                    "upload_time": item["upload_time"],
                    "score": float(score),
                })

        return results

    # =====================================================
    # 删除（持写锁）
    # =====================================================

    def delete_image(self, stored_name: str = None, origin_name: str = None) -> bool:
        if stored_name is None and origin_name is None:
            return False

        with self._rwlock.write():
            self._ensure_index()

            # O(1) 查找目标 ID
            fid = None
            if stored_name:
                fid = self._stored_to_id.get(stored_name)
            elif origin_name:
                ids = self._original_to_ids.get(origin_name, [])
                fid = ids[0] if ids else None

            if fid is None:
                return False

            item = self._id_to_meta.get(fid)
            if item is None:
                return False

            actual_stored = item["stored_name"]
            actual_original = item["original_name"]

            # 移动文件
            src = os.path.join(self.gallery_dir, actual_stored)
            if os.path.exists(src):
                shutil.move(src, os.path.join(self.deleted_dir, actual_stored))

            # 从 FAISS 删除（IndexIDMap 支持）
            self.index.remove_ids(np.array([fid], dtype=np.int64))
            self._save_index_unsafe()

            # 更新内存结构
            self._meta = [m for m in self._meta if m["id"] != fid]
            del self._id_to_meta[fid]
            del self._stored_to_id[actual_stored]
            ids_list = self._original_to_ids.get(actual_original, [])
            if fid in ids_list:
                ids_list.remove(fid)
            if not ids_list:
                self._original_to_ids.pop(actual_original, None)

            self._save_meta_unsafe()

        return True

    # =====================================================
    # 分页（持读锁）
    # =====================================================

    def list_gallery(
        self,
        page: int = 1,
        page_size: int = 20,
        keyword: Optional[str] = None,
        order: str = "desc",
    ) -> dict:
        with self._rwlock.read():
            items = self._meta if not keyword else [
                m for m in self._meta
                if keyword.lower() in m["original_name"].lower()
            ]
            items = sorted(items, key=lambda x: x["upload_time"], reverse=(order == "desc"))
            total = len(items)
            start = (page - 1) * page_size
            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "results": items[start: start + page_size],
            }

    # =====================================================
    # 重建索引（锁外推理，持写锁替换）
    # =====================================================

    def rebuild_index(self) -> int:
        files = [
            f for f in os.listdir(self.gallery_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]

        if not files:
            with self._rwlock.write():
                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(512))
                self._meta.clear()
                self._next_id = 0
                self._rebuild_aux_dicts()
                self._save_index_unsafe()
                self._save_meta_unsafe()
            return 0

        print(f"[ImageSearch] rebuilding index ({len(files)} images) ...")

        # 锁外推理
        images, valid_files = [], []
        for filename in files:
            path = os.path.join(self.gallery_dir, filename)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_files.append((filename, path))
            except Exception as e:
                print(f"[ImageSearch] skip bad image: {filename} ({e})")

        if not images:
            return 0

        features = self._extract_features_batch(images)

        with self._rwlock.write():
            new_index = faiss.IndexIDMap(faiss.IndexFlatIP(512))
            new_meta = []
            ids = list(range(len(valid_files)))
            new_index.add_with_ids(features, np.array(ids, dtype=np.int64))

            for fid, (filename, path) in zip(ids, valid_files):
                new_meta.append({
                    "id": fid,
                    "stored_name": filename,
                    "original_name": filename,
                    "upload_time": datetime.fromtimestamp(
                        os.path.getmtime(path)
                    ).strftime("%Y%m%d%H%M%S"),
                })

            self.index = new_index
            self._meta = new_meta
            self._next_id = len(valid_files)
            self._rebuild_aux_dicts()
            self._save_index_unsafe()
            self._save_meta_unsafe()

        print(f"[ImageSearch] rebuild finished ({len(new_meta)} images)")
        return len(new_meta)

    # =====================================================
    # 按名称精确搜索（O(1) 字典查找）
    # =====================================================

    def search_by_name_exact(self, name: str) -> List[dict]:
        with self._rwlock.read():
            ids = self._original_to_ids.get(name, [])
            return [dict(self._id_to_meta[fid]) for fid in ids if fid in self._id_to_meta]


# =========================================================
# 分组管理器
# =========================================================


class ImageSearchManager:

    def __init__(self, base_workspace: str):
        self.base_workspace = str(base_workspace)
        self.groups_dir = os.path.join(self.base_workspace, "groups")
        os.makedirs(self.groups_dir, exist_ok=True)

        self._engines: Dict[str, ImageSearchEngine] = {}
        self._groups: set = set(os.listdir(self.groups_dir))
        self._lock = threading.Lock()

    def _get_group_paths(self, group: str):
        group_dir = os.path.join(self.groups_dir, group)
        return {
            "gallery": os.path.join(group_dir, "gallery"),
            "data": os.path.join(group_dir, "data"),
            "deleted": os.path.join(group_dir, "deleted"),
        }

    def get_engine(self, group: str = "default") -> ImageSearchEngine:
        # 快速路径：无需持锁
        engine = self._engines.get(group)
        if engine is not None:
            return engine

        # 慢路径：初始化引擎（初始化在锁外完成，避免阻塞）
        paths = self._get_group_paths(group)
        new_engine = ImageSearchEngine(
            gallery_dir=paths["gallery"],
            data_dir=paths["data"],
            deleted_dir=paths["deleted"],
        )
        with self._lock:
            # 双重检查，防止并发时重复初始化
            if group not in self._engines:
                self._engines[group] = new_engine
                self._groups.add(group)
            return self._engines[group]

    def list_groups(self) -> List[str]:
        return list(self._groups)

    def add_images(self, files: List[tuple], group: str = "default") -> int:
        return self.get_engine(group).add_images(files)

    def delete_image(self, stored_name: str = None, origin_name: str = None, group: Optional[str] = None):
        if not stored_name and not origin_name:
            return False
        if group:
            return self.get_engine(group).delete_image(stored_name, origin_name)

        for g in self.list_groups():
            if self.get_engine(g).delete_image(stored_name, origin_name):
                return True
        return False

    def search(self, img_bytes: bytes, group: Optional[str] = None, top_k: int = 10) -> List[dict]:
        if group:
            results = self.get_engine(group).search_image(img_bytes, top_k)
            for r in results:
                r["group"] = group
            return results

        groups = self.list_groups()
        # 各 group 并行搜索
        all_results = []
        with ThreadPoolExecutor(max_workers=min(len(groups), 8)) as pool:
            futures = {pool.submit(self.get_engine(g).search_image, img_bytes, top_k): g for g in groups}
            for future in as_completed(futures):
                g = futures[future]
                try:
                    for r in future.result():
                        r["group"] = g
                        all_results.append(r)
                except Exception as e:
                    print(f"[ImageSearch] search error in group {g}: {e}")

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def list_gallery(
        self,
        group: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        keyword: Optional[str] = None,
        order: str = "desc",
    ) -> dict:
        if group:
            result = self.get_engine(group).list_gallery(page=page, page_size=page_size, keyword=keyword, order=order)
            for item in result["results"]:
                item["group"] = group
            return result

        groups = self.list_groups()
        reverse = (order == "desc")

        # 各 group 并行取全部条目，使用堆归并避免全量 sort
        def _get_all(g):
            engine = self.get_engine(g)
            with engine._rwlock.read():
                items = engine._meta if not keyword else [
                    m for m in engine._meta
                    if keyword.lower() in m["original_name"].lower()
                ]
                # 每个 group 内部先排序（数据量小时快）
                return sorted(items, key=lambda x: x["upload_time"], reverse=reverse), g

        sorted_per_group = []
        with ThreadPoolExecutor(max_workers=min(len(groups), 8)) as pool:
            for items, g in pool.map(_get_all, groups):
                for item in items:
                    item["group"] = g
                sorted_per_group.append(items)

        # 堆归并 k 路有序列表
        key_fn = lambda x: x["upload_time"]
        if reverse:
            merged = heapq.merge(*sorted_per_group, key=key_fn, reverse=True)
        else:
            merged = heapq.merge(*sorted_per_group, key=key_fn)

        start = (page - 1) * page_size
        # 跳过前 start 条，取 page_size 条
        result_items = []
        for i, item in enumerate(merged):
            if i < start:
                continue
            result_items.append(item)
            if len(result_items) == page_size:
                break

        # 总数需要遍历，用并行结果的 sum
        total = sum(len(items) for items in sorted_per_group)

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "results": result_items,
        }

    def rebuild_index(self, group: Optional[str] = None) -> dict:
        if group:
            return {group: self.get_engine(group).rebuild_index()}
        groups = self.list_groups()
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(groups), 4)) as pool:
            futures = {pool.submit(self.get_engine(g).rebuild_index): g for g in groups}
            for future in as_completed(futures):
                g = futures[future]
                try:
                    results[g] = future.result()
                except Exception as e:
                    print(f"[ImageSearch] rebuild error in group {g}: {e}")
                    results[g] = -1
        return results

    def search_by_name_exact(self, name: str, group: Optional[str] = None) -> List[dict]:
        if group:
            items = self.get_engine(group).search_by_name_exact(name)
            for item in items:
                item["group"] = group
            return items

        groups = self.list_groups()
        results = []
        with ThreadPoolExecutor(max_workers=min(len(groups), 8)) as pool:
            futures = {pool.submit(self.get_engine(g).search_by_name_exact, name): g for g in groups}
            for future in as_completed(futures):
                g = futures[future]
                try:
                    for item in future.result():
                        item["group"] = g
                        results.append(item)
                except Exception as e:
                    print(f"[ImageSearch] search_by_name error in group {g}: {e}")
        return results


# =========================================================
# 单例入口
# =========================================================

BASE_DIR = settings.BASE_DIR
_workspace = BASE_DIR / "oss" / "media"

_manager: ImageSearchManager = None
_manager_lock = threading.Lock()


def get_image_search_manager() -> ImageSearchManager:
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is not None:
            return _manager
        ClipModelSingleton.get()
        _manager = ImageSearchManager(base_workspace=_workspace)
        return _manager


# =========================================================
# 预热
# =========================================================

def warm_up_image_search():
    clip_model = ClipModelSingleton.get()
    dummy = Image.new("RGB", (224, 224), (255, 255, 255))
    tensor = clip_model.preprocess(dummy).unsqueeze(0).to(clip_model.device)
    with torch.no_grad():
        _ = clip_model.model.encode_image(tensor)
    print("[ImageSearch] Warm up finished.")
