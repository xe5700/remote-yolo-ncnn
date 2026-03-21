# syntax=docker/dockerfile:1
FROM  docker.1ms.run/ultralytics/ultralytics:latest-arm64
RUN sed -i 's@//ports.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list.d/ubuntu.sources && pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
#禁用apt-get交互
ENV DEBIAN_FRONTEND=noninteractive
RUN <<EOF
apt-get update
apt-get install -y software-properties-common sudo
echo 安装mesa相关包，适配ncnn框架vulkan加速
apt-get install -y mesa-utils mesa-vulkan-drivers
echo 安装工具方便调试
apt-get install -y vulkan-tools neofetch
echo 安装ncnn相关包
pip install ncnn pnnx
echo 清理apt-get数据
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

RUN <<EOF
echo 适配postmarketos等基于musl的发行版的驱动目录结构
ln -s /usr/lib/aarch64-linux-gnu/libvulkan_freedreno.so /usr/lib
ln -s /usr/lib/aarch64-linux-gnu/libvulkan_*.so /usr/lib
EOF

WORKDIR /models
