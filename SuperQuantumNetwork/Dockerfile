# 量子共生网络交易系统 Dockerfile
# 基于Python 3.9的基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建输出目录
RUN mkdir -p /app/output

# 暴露API端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "quantum_symbiotic_network.interfaces.api"] 