# CodeProject.AI Server - YOLO26n NCNN 推理服务器

为 **Frigate NVR** 和 **边缘设备** 打造的高性能对象检测 API 服务器，基于 YOLO26n 模型 + NCNN 框架 + Ultralytics 推理引擎。

## 📋 目录

- [功能特性](#-功能特性)
- [系统要求](#-系统要求)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [API 文档](#-api-文档)
- [Docker 部署](#-docker-部署)
- [Frigate 集成](#-frigate-集成)
- [性能优化](#-性能优化)
- [故障排查](#-故障排查)

## ✨ 功能特性

- 🚀 **高性能推理** - 基于 NCNN 框架，支持 Vulkan GPU 加速
- 🎯 **YOLO26n 模型** - 轻量级高精度目标检测模型
- 🔌 **Frigate 兼容** - 完美适配 Frigate NVR 的 MQTT 检测接口
- 🐳 **容器化部署** - 提供完整的 Docker 镜像和构建脚本
- ⚙️ **灵活配置** - 支持 CPU/Vulkan 设备切换，端口自定义
- 📊 **标准 API** - CodeProject.AI 兼容接口，易于集成

## 🛠 系统要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Linux 发行版 (默认镜像为ARM 64位系统构建，X86不建议使用。) |
| Python | 3.12 |
| GPU 驱动 | Mesa Vulkan 驱动（可选，用于 GPU 加速） |
| 内存 | 最低 2GB，推荐 4GB+ |
| 存储 | 最低 500MB（含模型文件） |

## 📁 项目结构

```
.
├── base/                    # 基础镜像构建
│   └── Dockerfile
├── server/                  # 服务器代码
│   ├── config/
│   │   └── config.toml     # 配置文件
│   ├── Dockerfile          # 服务器镜像
│   ├── main.py             # 测试脚本
│   └── server.py           # 主服务器代码
├── build.sh                # 构建脚本
├── gen_models.sh           # 模型生成脚本
├── config.toml             # 根目录配置
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r server/requirements.txt
```

### 2. 配置服务器

编辑 [server/config/config.toml](file:///mnt/g/projects/remote_yolo_ncnn/server/config/config.toml)：

```toml
[config]
device = "vulkan:0"      # 可选：cpu, vulkan:0, vulkan:1
server_port = 8080

[model]
model_name = "YOLO26n_384"
model_path = "models/yolo26n_ncnn_model_384"
```

### 3. 启动服务器

```bash
cd server
python3 server.py
```

### 4. 验证服务

```bash
curl http://localhost:8080/health
```

## ⚙️ 配置说明

### 配置文件 [config.toml](file:///mnt/g/projects/remote_yolo_ncnn/server/config/config.toml)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| [device](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) | string | `cpu` | 推理设备：`cpu`, `vulkan:0`, `vulkan:1` |
| [server_port](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) | int | `8080` | 服务器监听端口 |
| [model_name](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) | string | `YOLO26n_384` | 模型名称标识 |
| [model_path](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) | string | `models/yolo26n_ncnn_model_384` | 模型文件路径 |

### 环境变量

| 变量 | 说明 |
|------|------|
| `MESAGL_VERSION_OVERRIDE` | 覆盖 Mesa GL 版本（解决某些 GPU 兼容性问题） |

## 📡 API 文档

### 根端点

```http
GET /
```

**响应示例：**
```json
{
  "name": "CodeProject.AI Server",
  "version": "1.0.0",
  "description": "Object Detection API Server",
  "endpoints": {
    "detection": "/v1/vision/detection",
    "status": "/v1/vision/detection/status",
    "list": "/v1/vision/detection/list"
  }
}
```

### 健康检查

```http
GET /health
```

**响应示例：**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-13T10:00:00",
  "uptime": "running"
}
```

### 对象检测

```http
POST /v1/vision/detection
Content-Type: multipart/form-data
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | file | 是 | 待检测的图片文件 |
| `min_confidence` | float | 否 | 最小置信度，默认 0.4 |

**响应示例：**
```json
{
  "success": true,
  "message": "Inference completed successfully",
  "predictions": [
    {
      "x_min": 150.5,
      "y_min": 200.3,
      "x_max": 350.8,
      "y_max": 450.2,
      "confidence": 0.92,
      "label": "person"
    }
  ],
  "count": 1,
  "inferenceMs": 45,
  "processMs": 52,
  "moduleId": "YOLO26n",
  "moduleName": "Object Detection",
  "executionProvider": "Vulkan",
  "canUseGPU": true
}
```

## 🐳 Docker 部署


### 使用build.sh构建镜像
./build.sh
会输出两个镜像 xe5700/ultralytics_ncnn_vulkan:latest xe5700/ultralytics_ncnn_vulkan_server:latest

### 手动构建镜像

```bash
# 构建基础镜像
docker build -t xe5700/ultralytics_ncnn_vulkan:latest -f base/Dockerfile .
# 构建服务器镜像
docker build -t xe5700/ultralytics_ncnn_vulkan_server:latest -f server/Dockerfile .
```

### 运行容器

```bash
docker run -d \
  --name codeproject_ai \
  --device /dev/dri:/dev/dri \
  -p 8080:8080 \
  -v ./models:/server/models \
  -v ./config:/server/config \
  xe5700/ultralytics_ncnn_vulkan_server:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  server:
    image: xe5700/ultralytics_ncnn_vulkan_server:latest
    ports:
      - "8080:8080"
    volumes:
      - ./models:/server/models
      - ./server/config:/server/config
    devices:
      # 对应 --device /dev/dri/renderD128
      - /dev/dri/renderD128:/dev/dri/renderD128
    volumes:
      # 对应 -v /usr/share/vulkan/icd.d/freedreno_icd.aarch64.json:...
      - /usr/share/vulkan/icd.d/freedreno_icd.aarch64.json:/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json
      # 对应 -v $(pwd)/config:/server/config
      # 假设此文件在当前目录下的 config 文件夹
      - ./config:/server/config
    logging:
     driver: "json-file"   #默认的文件日志驱动
     options:
      max-size: "16m"
      max-file: "3"
    environment:
      # 适配骁龙平台gpu 可能和我一样使用GPU会推理出错 
      #- VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json
      
    restart: unless-stopped
```

## 🎥 Frigate 集成

### Frigate 配置示例

```yaml
# config.yml
detectors:
  codeproject_ai:
    type: http
    url: http://codeproject_ai:8080/v1/vision/detection

cameras:
  front_door:
    detect:
      enabled: true
      width: 1920
      height: 1080
      fps: 5
    objects:
      track:
        - person
        - car
        - dog
    detection:
      enabled: true
```

### 注意事项

1. **网络连通性**：确保 Frigate 容器能访问推理服务器
2. **模型匹配**：Frigate 的追踪对象需与模型支持的类别匹配
3. **性能调优**：根据边缘设备性能调整 `fps` 和分辨率

## ⚡ 性能优化

### Vulkan GPU 加速

```toml
# config.toml
[config]
device = "vulkan:0"  # 使用第一块 Vulkan GPU
```

### CPU 加速 支持

```toml
# config.toml
[config]
device = "cpu"  # 使用处理器进行推理
```

### 日志控制

服务器默认屏蔽了 `werkzeug` 和 `ultralytics` 的日志输出，如需调试可修改 [server.py](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py) 中的日志级别。

## 🔧 故障排查

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| Vulkan 初始化失败 | 检查 `mesa-vulkan-drivers` 是否安装 |
| 模型加载失败 | 确认 [model_path](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) 路径正确 |
| 端口占用 | 修改 [server_port](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) 配置 |
| GPU 不被识别 | 检查 `/dev/dri` 设备权限 |

### 日志查看

```bash
# 查看容器日志
docker logs codeproject_ai

# 实时跟踪日志
docker logs -f codeproject_ai
```

## 📱 postmarketOS 设备支持

### 一加 6 (OnePlus 6/fajita) + postmarketOS 测试结果
#### 使用ultralytics官方的bus.jpg进行测试
| 项目 | 配置 | 结果 |
|------|------|------|
| **设备** | OnePlus 6 (Snapdragon 845) | ⚠️ 部分兼容 |
| **系统** | postmarketOS (musl libc) | ✅ 支持 |
| **模型** | YOLO26n_384 (320x320 FP32) | ✅ 可用 |
| **Vulkan 推理** | `device="vulkan:0"` | ❌ 数据错误 |
| **CPU 推理** | `device="cpu"` | ✅ 正常 |
| **推理延迟** | Frigate 服务器端 | ~100ms |

### Vulkan 失败原因分析 (postmarketOS 环境)

1. **freedreno/turnip 驱动支持有限** - 开源 Vulkan 驱动仍在开发中
2. **musl libc 兼容性** - 部分 Vulkan 运行时库可能不完整
3. **内核驱动版本** - msm DRM 驱动功能受限

一加 6 的 Adreno 630 GPU 在 postmarketOS 上存在以下限制：

## 🖥️ 桌面平台对比测试

### Linux 测试结果

| 项目 | 配置 | 结果 |
|------|------|------|
| **处理器** | AMD Ryzen 7 5900X | ✅ 完全兼容 |
| **显卡** | AMD Radeon RX 6900XT | ✅ 完全兼容 |
| **系统** | Arch Linux | ✅ 支持 |
| **模型** | YOLO26n_384 (320x320 FP32) | ✅ 可用 |
| **Vulkan 推理** | `device="vulkan:0"` | ✅ 正常 |
| **CPU 推理** | `device="cpu"` | ✅ 正常 |
| **推理延迟** | 接口端提示 | ~6ms |


## 📄 许可证

本项目基于 AGPL 3.0 许可证开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

- 文档：https://www.codeproject.com/AI/docs/
- 问题反馈：GitHub Issues

---

**版本**: 1.0.0  
**最后更新**: 2026-03-21
**维护者**: xe5700