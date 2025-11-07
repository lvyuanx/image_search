# 🖼️ Image Search Tool（以图搜图工具）
## 📘 项目简介

Image Search Tool 是一个基于本地图片特征向量检索的轻量级以图搜图系统。
通过提取图片特征（如 OpenCV 或深度模型），并进行相似度计算，实现“以图找图”的功能。
适用于图片去重、相似图检测、图库管理等场景。

##✨ 功能特性

✅ 以图搜图：上传一张图片即可搜索相似图片

✅ 批量索引：支持批量导入图片目录并生成索引

✅ 相似度评分：结果带相似度分值

✅ 自动容错：跳过损坏或非图像文件

✅ 图库清理：自动删除坏图

✅ 本地运行，无需外部服务

## 🧩 项目结构
```
image_search/
├── gallery/                # 图片图库
├── deleted/                # 存放坏图
├── app.py                  # 主程序
├── feature_extractor.py    # 特征提取逻辑
├── search_engine.py        # 检索逻辑
├── clean_bad_images.py     # 清理坏图脚本
├── requirements.txt        # 依赖
└── README.md               # 说明文档
```

## ⚙️ 环境依赖

Python 3.9+

安装依赖：
```bash
pip install pillow numpy opencv-python scikit-learn tqdm
```

如果使用深度学习模型（可选）：
```bash
pip install torch torchvision
```

## 🧠 运行示例
```bash
uvicorn app:app 
```

## 📜 License

本项目仅供学习与研究使用，禁止用于商业用途。

## ✍️ 作者信息

作者：lvyuanxiang

日期：2025-11-07