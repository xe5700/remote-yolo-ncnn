#!/usr/bin/env python3
"""
CodeProject.AI Server - 空壳实现
基于Flask的AI检测API服务器
"""

import sys
import traceback

from flask import Flask, request, jsonify
import time
import uuid
import os
from datetime import datetime
from ultralytics import YOLO
from werkzeug.datastructures import FileStorage
from PIL import Image
import toml  # [修复] 添加 toml 导入
class Config:
    device: str
    server_port: int
    model_name: str
    model_path: str

    def __init__(self):
        self.device = "cpu"
        self.server_port = 8080
        self.model_name = "YOLO26n_384"
        self.model_path = "models/yolo26n_ncnn_model_384"

    def load(self, config_file):
        """Load configuration from TOML file."""
        with open(config_file, "r") as f:
            config = toml.load(f)

        self.device = config['config']["device"]
        self.server_port = config['config']["server_port"]
        self.model_name = config["model"]["model_name"]
        self.model_path = config["model"]["model_path"]

    def save(self, config_file):
        """Save configuration to TOML file."""
        with open(config_file, "w") as f:
            toml.dump(
                {
                    "config": {"device": self.device, "server_port": self.server_port},
                    "model": {
                        "model_name": self.model_name,
                        "model_path": self.model_path,
                    },
                },
                f,
            )

config: 'Config'
app = Flask(__name__)
model : YOLO
# 模块配置
MODULE_ID = str(uuid.uuid4())
MODULE_NAME = "ObjectDetection-YOLO"
VERSION = "1.0.0"

# 支持的检测标签 (COCO数据集80类)
COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# 自定义模型配置
CUSTOM_MODELS = {
    "ipcam-animal": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "bear",
        "deer",
        "rabbit",
        "raccoon",
        "fox",
        "skunk",
        "squirrel",
        "pig",
    ],
    "ipcam-dark": ["Bicycle", "Bus", "Car", "Cat", "Dog", "Motorcycle", "Person"],
    "ipcam-general": ["person", "vehicle"],
    "ipcam-combined": [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "bear",
        "deer",
        "rabbit",
        "raccoon",
        "fox",
        "skunk",
        "squirrel",
        "pig",
    ],
}


def create_base_response():
    """创建基础响应结构"""
    use_gpu = False
    excution_provider = "CPU"
    if "vulkan" in config.device:
        use_gpu = True
        excution_provider = "GPU"
    return {
        "success": True,
        "message": "Inference completed successfully",
        "predictions": [],
        "count": 0,
        "inferenceMs": 0,
        "processMs": 0,
        "moduleId": MODULE_ID,
        "moduleName": config.model_name,
        "command": "detect",
        "executionProvider": excution_provider,
        "canUseGPU": use_gpu,
        "analysisRoundTripMs": 0,
    }


def create_mock_predictions(num_predictions=3):
    """创建模拟的预测结果"""
    import random

    predictions = []
    for i in range(num_predictions):
        label = random.choice(COCO_LABELS)
        predictions.append(
            {
                "x_min": random.randint(50, 300),
                "y_min": random.randint(50, 300),
                "x_max": random.randint(301, 600),
                "y_max": random.randint(301, 600),
                "confidence": round(random.uniform(0.5, 0.99), 2),
                "label": label,
            }
        )
    return predictions


# ==================== 核心API端点 ====================


@app.route("/v1/vision/detection", methods=["POST"])
def object_detection():
    """
    标准对象检测端点
    使用YOLO26模型检测COCO数据集中的80种对象
    """
    start_time = time.time()
    response = create_base_response()

    try:
        # # 检查是否有图片文件
        # if "image" not in request.files:
        #     response["success"] = False
        #     response["message"] = "No image file provided"
        #     response["error"] = "Missing 'image' parameter in request"
        #     return jsonify(response), 400

        if len(request.files) != 1:
            response["success"] = False
            response["message"] = "Multiple image files provided"
            return jsonify(response), 400
        # 获取最小置信度参数
        min_confidence = float(request.form.get("min_confidence", 0.4))
        predict_files = []
        for file in request.files:
            file: FileStorage = request.files[file]
            # 导入为PIL图片
            img = Image.open(file.stream)
            predict_files.append(img)

        data = model.predict(predict_files[0], device=config.device, conf=min_confidence)
        # 获取推理时间
        # inference_time = out[0]["speed"]["inference"]
        # # 模拟推理时间 (实际应用中这里会调用YOLO模型)
        # inference_start = time.time()
        # time.sleep(0.05)  # 模拟推理延迟
        # inference_time = int((time.time() - inference_start) * 1000)

        for item in data:
            # item = data[i]
            response["processMs"] = int(
                round(
                    item.speed["preprocess"]  + item.speed["postprocess"],
                    10,
                )
            )
            predictions = []
            response["inferenceMs"] = int(round(item.speed["inference"]))
            for box in item.boxes:
                xyxy = box.xyxy.float().tolist()
                
                # print(xyxy)
                # print(xyxy[0])
                x_min = xyxy[0][0]
                y_min = xyxy[0][1]
                x_max = xyxy[0][2]
                y_max = xyxy[0][3]
                confidence = box.conf[0].float().item()
                predictions.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "confidence": confidence,
                    "label": item.names[box.cls.int().item()],
                })
                response["predictions"] = predictions
                response["count"] = len(predictions)

                response["analysisRoundTripMs"] = (
                    response["processMs"] + response["inferenceMs"]
                )
                response["message"] = f"Found {len(predictions)} objects in the image"
        # # 过滤低置信度结果
        # predictions = [p for p in predictions if p["confidence"] >= min_confidence]

        # # 填充响应
        # response["predictions"] = predictions
        # response["count"] = len(predictions)
        # response["inferenceMs"] = inference_time
        # response["processMs"] = int((time.time() - start_time) * 1000)
        # response["analysisRoundTripMs"] = response["processMs"]
        # response["message"] = f"Found {len(predictions)} objects in the image"

    except Exception as e:
        response["success"] = False
        response["message"] = "Error during inference"
        response["error"] = str(e)
        traceback.print_exc(file=sys.stdout)
        return jsonify(response), 500

    return jsonify(response)


# ==================== 管理API端点 ====================


@app.route("/v1/vision/detection/status", methods=["GET"])
def status():
    """获取模块状态"""
    use_gpu = False
    if "vulkan" in config.device:
        use_gpu = True
    return jsonify(
        {
            "success": True,
            "message": "Module is running",
            "moduleId": MODULE_ID,
            "moduleName": MODULE_NAME,
            "version": VERSION,
            "status": "Healthy",
            "canUseGPU": use_gpu,
            "executionProvider": config.device,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/v1/vision/detection/list", methods=["GET"])
def list_models():
    """列出可用的模型"""
    return jsonify(
        {
            "success": True,
            "message": "Available models",
            "models": {
                "standard": {
                    "name": config.model_name,
                    "labels": COCO_LABELS,
                    "labelCount": len(COCO_LABELS),
                },
                "custom": [],
            },
            "moduleId": MODULE_ID,
            "moduleName": MODULE_NAME,
        }
    )


# @app.route('/v1/vision/custom/list', methods=['GET'])
# def list_custom_models():
#     """列出所有自定义模型"""
#     models_info = []
#     for model_name, labels in CUSTOM_MODELS.items():
#         models_info.append({
#             "name": model_name,
#             "labelCount": len(labels),
#             "labels": labels
#         })

#     return jsonify({
#         "success": True,
#         "message": f"Found {len(CUSTOM_MODELS)} custom models",
#         "models": models_info,
#         "moduleId": MODULE_ID,
#         "moduleName": MODULE_NAME
#     })


# ==================== 其他端点 ====================


@app.route("/", methods=["GET"])
def index():
    """API根端点"""
    return jsonify(
        {
            "name": "CodeProject.AI Server",
            "version": VERSION,
            "description": "Object Detection API Server (Mock Implementation)",
            "endpoints": {
                "detection": "/v1/vision/detection",
                # "custom": "/v1/vision/custom/<model_name>",
                "status": "/v1/vision/detection/status",
                "list": "/v1/vision/detection/list",
                "custom_list": "/v1/vision/custom/list",
            },
            "documentation": "https://www.codeproject.com/AI/docs/",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """健康检查端点"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": "running",
        }
    )


# [config]
# device="cpu"
# server_port=8080
# [model]
# model_name="YOLO26n_384"
# #可选fp16 int8 量化模型
# model_path="models/yolo26n_ncnn_model_384"
# 加载config.toml，载入配置。

def main():
    global config,model
    # 检测配置是否存在
    config = Config()
    cfg_path=os.path.join("config","config.toml")
    if os.path.exists():
        config.load(cfg_path)
        print(f"Loaded configuration from config.toml: {config}")
    else:
        config.save(cfg_path)
        print(f"No configuration file found. Using default configuration: {config}")
    print("=" * 60)
    print("CodeProject.AI Server - Mock Implementation")
    print("=" * 60)
    print(f"Module ID: {MODULE_ID}")
    print(f"Module Name: {MODULE_NAME}")
    print(f"Version: {VERSION}")
    print(f"Server: http://localhost:32168")
    print("=" * 60)
    model = YOLO(config.model_path, task="detect")
    # out=model.predict(list("./models/bus.jpg","n01440764_tench.JPEG"), device="vulkan:0", save=False)
    # 在容器环境中运行时，监听所有网络接口
    app.run(host="0.0.0.0", port=config.server_port, debug=False, threaded=False)


if __name__ == "__main__":
    main()