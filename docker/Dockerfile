# syntax=docker/dockerfile:1.6

# ----------------------
# 基础镜像
# ----------------------
FROM python:3.11-slim

# ----------------------
# 设置环境变量
# ----------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VIRTUALENVS_CREATE=false \
    DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# ----------------------
# 配置 apt 国内源并安装系统依赖
# ----------------------
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free" > /etc/apt/sources.list \
 && echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list \
 && echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      curl \
      git \
 && rm -rf /var/lib/apt/lists/*

# ----------------------
# 设置工作目录
# ----------------------
WORKDIR /app

# ----------------------
# 复制依赖文件
# ----------------------
COPY requirements.txt .

# ----------------------
# 安装 Python 依赖和 CPU 版本 torch
# ----------------------
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu

# ----------------------
# 复制项目代码
# ----------------------
# COPY . .

# ----------------------
# 默认启动命令
# ----------------------
CMD ["python", "main.py", "run"]