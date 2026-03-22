# CodeProject.AI Server - YOLO26n NCNN Inference Server

High-performance object detection API server built for **Frigate NVR** and **Edge Devices**, based on YOLO26n model + NCNN framework + Ultralytics inference engine.

## 📋 Table of Contents

- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [Frigate Integration](#-frigate-integration)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- 🚀 **High-Performance Inference** - Based on NCNN framework, supports Vulkan GPU acceleration
- 🎯 **YOLO26n Model** - Lightweight high-precision object detection model
- 🔌 **Frigate Compatible** - Perfectly adapts to Frigate NVR's CodeProject.AI interface
- 🐳 **Containerized Deployment** - Complete Docker image and build scripts provided
- ⚙️ **Flexible Configuration** - Supports CPU/Vulkan device switching, customizable ports
- 📊 **Standard API** - CodeProject.AI compatible interface, easy to integrate

## 🛠 System Requirements

| Component | Requirements |
|-----------|--------------|
| Operating System | Linux Distribution (Default image built for ARM 64-bit, X86 not recommended) |
| Python | 3.12 |
| GPU Driver | Mesa Vulkan Driver (Optional, for GPU acceleration) |
| Memory | Minimum 2GB, Recommended 4GB+ |
| Storage | Minimum 500MB (including model files) |

## 📁 Project Structure

```
.
├── base/                    # Base image build
│   └── Dockerfile
├── server/                  # Server code
│   ├── config/
│   │   └── config.toml     # Configuration file
│   ├── Dockerfile          # Server image
│   ├── main.py             # Test script
│   └── server.py           # Main server code
├── build.sh                # Build script
├── gen_models.sh           # Model generation script
├── config.toml             # Root directory configuration
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r server/requirements.txt
```

### 2. Configure Server

Edit [server/config/config.toml](file:///mnt/g/projects/remote_yolo_ncnn/server/config/config.toml):

```toml
[config]
device = "vulkan:0"      # Options: cpu, vulkan:0, vulkan:1
server_port = 8080

[model]
model_name = "YOLO26n_384"
model_path = "models/yolo26n_ncnn_model_384"
```

### 3. Start Server

```bash
cd server
python3 server.py
```

### 4. Verify Service

```bash
curl http://localhost:8080/health
```

## ⚙️ Configuration

### Configuration File [config.toml](file:///mnt/g/projects/remote_yolo_ncnn/server/config/config.toml)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| [device](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) | string | `cpu` | Inference device: `cpu`, `vulkan:0`, `vulkan:1` |
| [server_port](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) | int | `8080` | Server listening port |
| [model_name](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) | string | `YOLO26n_384` | Model name identifier |
| [model_path](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) | string | `models/yolo26n_ncnn_model_384` | Model file path |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MESAGL_VERSION_OVERRIDE` | Override Mesa GL version (resolve certain GPU compatibility issues) |

## 📡 API Documentation

### Root Endpoint

```http
GET /
```

**Response Example:**
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

### Health Check

```http
GET /health
```

**Response Example:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-13T10:00:00",
  "uptime": "running"
}
```

### Object Detection

```http
POST /v1/vision/detection
Content-Type: multipart/form-data
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | Yes | Image file to be detected |
| `min_confidence` | float | No | Minimum confidence, default 0.4 |

**Response Example:**
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

## 🐳 Docker Deployment

### Build Image Using build.sh

```bash
./build.sh
```

Outputs two images: `xe5700/ultralytics_ncnn_vulkan:latest` and `xe5700/ultralytics_ncnn_vulkan_server:latest`

### Manual Image Build

```bash
# Build base image
docker build -t xe5700/ultralytics_ncnn_vulkan:latest -f base/Dockerfile .
# Build server image
docker build -t xe5700/ultralytics_ncnn_vulkan_server:latest -f server/Dockerfile .
```

### Run Container

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
      # Corresponds to --device /dev/dri/renderD128
      - /dev/dri/renderD128:/dev/dri/renderD128
    volumes:
      # Corresponds to -v /usr/share/vulkan/icd.d/freedreno_icd.aarch64.json:...
      - /usr/share/vulkan/icd.d/freedreno_icd.aarch64.json:/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json
      # Corresponds to -v $(pwd)/config:/server/config
      - ./config:/server/config
    logging:
     driver: "json-file"   # Default file logging driver
     options:
      max-size: "16m"
      max-file: "3"
    environment:
      # Adapt for Snapdragon platform GPU, may encounter inference errors like mine
      #- VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json
      
    restart: unless-stopped
```

## 🎥 Frigate Integration

### Frigate Configuration Example

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

### Notes

1. **Network Connectivity**: Ensure Frigate container can access the inference server
2. **Model Matching**: Frigate's tracked objects must match model-supported categories
3. **Performance Tuning**: Adjust `fps` and resolution based on edge device performance

## ⚡ Performance Optimization

### Vulkan GPU Acceleration

```toml
# config.toml
[config]
device = "vulkan:0"  # Use first Vulkan GPU
```

### CPU Acceleration Support

```toml
# config.toml
[config]
device = "cpu"  # Use processor for inference
```

### Log Control

Server defaults to suppressing `werkzeug` and `ultralytics` log output. Modify log level in [server.py](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py) for debugging.

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Vulkan initialization failed | Check if `mesa-vulkan-drivers` is installed |
| Model loading failed | Verify [model_path](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) is correct |
| Port occupied | Modify [server_port](file:///mnt/g/projects/remote_yolo_ncnn/server/server.py#L0-L0) configuration |
| GPU not recognized | Check `/dev/dri` device permissions |

### Log Viewing

```bash
# View container logs
docker logs codeproject_ai

# Follow logs in real-time
docker logs -f codeproject_ai
```

## 📱 postmarketOS Device Support

### OnePlus 6 (fajita) + postmarketOS Test Results

#### Tested with Ultralytics official bus.jpg

| Item | Configuration | Result |
|------|---------------|--------|
| **Device** | OnePlus 6 (Snapdragon 845) | ⚠️ Partially Compatible |
| **System** | postmarketOS (musl libc) | ✅ Supported |
| **Model** | YOLO26n_384 (320x320 FP32) | ✅ Available |
| **Vulkan Inference** | `device="vulkan:0"` | ❌ Data Error |
| **CPU Inference** | `device="cpu"` | ✅ Normal |
| **Inference Latency** | Frigate Server Side | ~100ms |

### Vulkan Failure Analysis (postmarketOS Environment)

1. **Limited freedreno/turnip Driver Support** - Open-source Vulkan driver still in development
2. **musl libc Compatibility** - Some Vulkan runtime libraries may be incomplete
3. **Kernel Driver Version** - msm DRM driver functionality limited

OnePlus 6's Adreno 630 GPU has the following limitations on postmarketOS.

## 🖥️ Desktop Platform Comparison Tests

### Linux Test Results

| Item | Configuration | Result |
|------|---------------|--------|
| **Processor** | AMD Ryzen 7 5900X | ✅ Fully Compatible |
| **Graphics Card** | AMD Radeon RX 6900XT | ✅ Fully Compatible |
| **System** | Arch Linux | ✅ Supported |
| **Model** | YOLO26n_384 (320x320 FP32) | ✅ Available |
| **Vulkan Inference** | `device="vulkan:0"` | ✅ Normal |
| **CPU Inference** | `device="cpu"` | ✅ Normal |
| **Inference Latency** | API Endpoint | ~6ms |

## 📄 License

This project is open-sourced under the AGPL 3.0 License.

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📞 Support

- Documentation: https://www.codeproject.com/AI/docs/
- Issue Reporting: GitHub Issues

---

**Version**: 1.0.0  
**Last Updated**: 2026-03-21  
**Maintainer**: xe5700
