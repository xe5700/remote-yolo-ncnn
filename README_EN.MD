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