#!/bin/bash
echo 开始编译基础镜像
docker build --platform linux/arm64 -t xe5700/ultralytics_ncnn_vulkan:latest base
echo 开始编译应用镜像
docker build --platform linux/arm64 -t xe5700/ultralytics_ncnn_vulkan_server:latest -f server/Dockerfile .