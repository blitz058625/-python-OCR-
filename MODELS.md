# Models Documentation for Parking Violation Detection System

## Introduction
This document provides comprehensive instructions on how to download and configure the models required for the parking violation detection system.

## YOLOv8 Vehicle Detection Models
1. **Download the YOLOv8 Models**:
   - Visit the [YOLOv8 Releases page](https://github.com/ultralytics/yolov8/releases) on GitHub.
   - Download the desired model weights (.pt files) specific to vehicle detection. Recommended models include:
     - YOLOv8n (Nano)
     - YOLOv8s (Small)
     - YOLOv8m (Medium)

2. **Installation**:
   Make sure you have the necessary dependencies:
   ```bash
   pip install -U opencv-python
   pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   ```

3. **Configure the Model**:
   - Create a configuration file (e.g., `yolo_config.yaml`) to set parameters like input size and confidence threshold.
   - Place the downloaded weights in an accessible directory.

## HyperLPR3 License Plate Recognition Models
1. **Download the HyperLPR3 Models**:
   - You can find the HyperLPR3 models in the [HyperLPR GitHub repository](https://github.com/shinkai213/HyperLPR).
   - Look for the pre-trained models under the releases section.

2. **Installation**:
   Ensure you have the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure HyperLPR**:
   - Set up the model configuration file as specified in the HyperLPR documentation.
   - Place the model weights in the designated folder for HyperLPR.

## Other Dependencies
- Ensure you have the following installed:
  - NumPy
  - OpenCV
  - TensorFlow
  - PyTorch

## Conclusion
Make sure to verify all installations and downloads. For any issues during the setup, refer to the respective GitHub issues page for each model. 
