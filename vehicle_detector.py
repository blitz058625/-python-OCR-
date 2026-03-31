#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车辆检测模块 - 集成YOLOv8模型
负责检测图片/视频帧中的车辆
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class YOLOv8VehicleDetector:
    """YOLOv8车辆检测器"""

    def __init__(self, model_path: str = None, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45, device: str = 'cpu',
                 min_box_area: int = 0, suppress_part_detections: bool = True):
        """
        初始化YOLOv8检测器

        Args:
            model_path: 模型文件路径 (.pt文件)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            device: 运行设备 ('cpu' 或 'cuda')
            min_box_area: 最小框面积（像素），过小的如后视镜可过滤；0=不启用
            suppress_part_detections: 是否抑制重叠的小框（如后视镜被识别成车）
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.min_box_area = min_box_area
        self.suppress_part_detections = suppress_part_detections
        self.model = None

        # 车辆类别 (COCO数据集中的car, truck, bus等)
        self.vehicle_classes = [2, 5, 7]  # car, bus, truck
        self.class_names = {2: 'car', 5: 'bus', 7: 'truck'}

        self._load_model()

    def _load_model(self):
        """加载YOLOv8模型"""
        try:
            from ultralytics import YOLO
            logger.info("正在加载YOLOv8模型...")

            if self.model_path and Path(self.model_path).exists():
                # 加载自定义训练的模型
                self.model = YOLO(self.model_path)
                logger.info(f"✓ 已加载自定义YOLOv8模型: {self.model_path}")
            else:
                # 加载预训练模型
                self.model = YOLO('yolov8n.pt')  # 使用nano版本，轻量级
                logger.warning(f"⚠️ 未找到指定模型，使用预训练模型 yolov8n.pt")

            # 设置模型参数
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold

            logger.info("✓ YOLOv8模型加载成功")

        except ImportError as e:
            logger.error(f"❌ 无法导入ultralytics: {e}")
            logger.error("请安装: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise

    def detect_vehicles(self, image) -> List[Dict[str, Any]]:
        """
        检测图像中的车辆

        Args:
            image: OpenCV图像 (BGR格式)

        Returns:
            车辆检测结果列表，每个包含:
            - bbox: [x1, y1, x2, y2]
            - confidence: 置信度
            - class_id: 类别ID
            - class_name: 类别名称
        """
        if self.model is None:
            logger.error("模型未加载")
            return []

        try:
            # 运行推理
            results = self.model(image, verbose=False)

            vehicles = []

            # 处理结果
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    for box in boxes:
                        # 获取边界框
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = bbox.astype(int)

                        # 获取置信度和类别
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        # 只保留车辆类别
                        if class_id in self.vehicle_classes:
                            vehicle = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.class_names.get(class_id, 'vehicle')
                            }
                            vehicles.append(vehicle)

            # 按置信度排序
            vehicles.sort(key=lambda x: x['confidence'], reverse=True)

            # 过滤后视镜等误检
            vehicles = self._filter_false_positives(vehicles)

            logger.debug(f"检测到 {len(vehicles)} 辆车")

            return vehicles

        except Exception as e:
            logger.error(f"车辆检测出错: {e}")
            return []

    def _filter_false_positives(self, vehicles: List[Dict]) -> List[Dict]:
        """过滤后视镜等误检：过小框、与车体重叠的小框"""
        if not vehicles:
            return vehicles

        # 1. 最小面积过滤
        if self.min_box_area > 0:
            vehicles = [v for v in vehicles
                       if (v['bbox'][2] - v['bbox'][0]) * (v['bbox'][3] - v['bbox'][1]) >= self.min_box_area]

        # 2. 重叠抑制：小框与更大框重叠时，判定为车体部件（如后视镜），剔除小框
        if self.suppress_part_detections and len(vehicles) > 1:
            # 按面积从大到小处理，优先保留大框（车身）
            by_area = sorted(vehicles, key=lambda v: (v['bbox'][2]-v['bbox'][0])*(v['bbox'][3]-v['bbox'][1]), reverse=True)
            kept = []
            for v in by_area:
                bbox = v['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                is_part = False
                for k in kept:
                    k_area = (k['bbox'][2] - k['bbox'][0]) * (k['bbox'][3] - k['bbox'][1])
                    # 当前框明显更小，且多半在大框内 → 视为车体部件
                    if area < k_area * 0.35 and self._intersection_over_small(bbox, k['bbox']) > 0.4:
                        is_part = True
                        break
                if not is_part:
                    kept.append(v)
            # 恢复按置信度排序
            vehicles = sorted(kept, key=lambda x: x['confidence'], reverse=True)

        return vehicles

    @staticmethod
    def _intersection_over_small(bbox_small: list, bbox_large: list) -> float:
        """小框与大框的交集占小框面积的比例，>0.4 表示小框多半在大框内"""
        x1 = max(bbox_small[0], bbox_large[0])
        y1 = max(bbox_small[1], bbox_large[1])
        x2 = min(bbox_small[2], bbox_large[2])
        y2 = min(bbox_small[3], bbox_large[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        small_area = (bbox_small[2] - bbox_small[0]) * (bbox_small[3] - bbox_small[1])
        return inter / small_area if small_area > 0 else 0.0

    def detect_vehicles_with_tracking(self, image) -> List[Dict[str, Any]]:
        """
        使用 YOLO 内置 ByteTracker 进行车辆检测和跟踪

        调用 model.track() 而非 model.predict()，启用 Ultralytics 的 ByteTracker，
        返回带 track_id 的检测结果，用于视频流中保持跨帧 ID 一致性。

        Args:
            image: OpenCV图像 (BGR格式)

        Returns:
            车辆检测结果列表，每个包含:
            - bbox: [x1, y1, x2, y2]
            - confidence: 置信度
            - class_id: 类别ID
            - class_name: 类别名称
            - track_id: 跟踪ID (由 ByteTracker 分配)
        """
        if self.model is None:
            logger.error("模型未加载")
            return []

        try:
            # 使用 model.track() 启用 ByteTracker，置信度沿用用户配置（与 detect_vehicles 一致）
            results = self.model.track(
                image, verbose=False,
                conf=self.conf_threshold,  # 与 detect_vehicles 相同阈值，保证检测质量
                persist=True
            )

            vehicles = []

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    for i, box in enumerate(boxes):
                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = bbox.astype(int)

                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        if class_id in self.vehicle_classes:
                            # track_id: ByteTracker 分配的跨帧 ID
                            track_id = None
                            if box.id is not None:
                                track_id = int(box.id[0].cpu().numpy())

                            vehicle = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.class_names.get(class_id, 'vehicle')
                            }
                            if track_id is not None:
                                vehicle['track_id'] = track_id
                            vehicles.append(vehicle)

            vehicles.sort(key=lambda x: x['confidence'], reverse=True)

            # 过滤后视镜等误检
            vehicles = self._filter_false_positives(vehicles)

            logger.debug(f"ByteTracker 检测到 {len(vehicles)} 辆车")

            return vehicles

        except Exception as e:
            logger.error(f"车辆跟踪检测出错: {e}")
            return []

    def extract_vehicle_regions(self, image, vehicles: List[Dict]) -> List[Dict[str, Any]]:
        """
        从图像中提取车辆区域

        Args:
            image: OpenCV图像
            vehicles: 车辆检测结果

        Returns:
            包含车辆区域图像的结果列表
        """
        import cv2

        vehicle_regions = []

        for vehicle in vehicles:
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = bbox

            # 确保边界框在图像范围内
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # 提取车辆区域
            if x2 > x1 and y2 > y1:
                vehicle_image = image[y1:y2, x1:x2]

                vehicle_region = vehicle.copy()
                vehicle_region['vehicle_image'] = vehicle_image
                vehicle_region['image_shape'] = vehicle_image.shape

                vehicle_regions.append(vehicle_region)

        return vehicle_regions

    def visualize_detections(self, image, vehicles: List[Dict], show_labels: bool = True):
        """
        在图像上可视化检测结果

        Args:
            image: OpenCV图像
            vehicles: 车辆检测结果
            show_labels: 是否显示标签

        Returns:
            标注后的图像
        """
        import cv2

        vis_image = image.copy()

        for i, vehicle in enumerate(vehicles):
            bbox = vehicle['bbox']
            confidence = vehicle['confidence']
            class_name = vehicle['class_name']
            track_id = vehicle.get('track_id')

            x1, y1, x2, y2 = bbox

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if show_labels:
                label = f"{class_name} {confidence:.2f}"
                if track_id is not None:
                    label = f"ID:{track_id} " + label
                cv2.putText(vis_image, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_image

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"status": "not_loaded"}

        try:
            return {
                "status": "loaded",
                "model_path": self.model_path,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "device": self.device,
                "supported_classes": self.class_names
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def create_vehicle_detector(model_path: str = None, **kwargs) -> YOLOv8VehicleDetector:
    """
    创建车辆检测器

    Args:
        model_path: 模型路径
        **kwargs: 其他参数

    Returns:
        YOLOv8VehicleDetector实例
    """
    return YOLOv8VehicleDetector(model_path=model_path, **kwargs)


# ========== 测试和示例代码 ==========
def test_vehicle_detector():
    """测试车辆检测器"""
    import cv2

    # 创建检测器
    detector = create_vehicle_detector()

    # 读取测试图片
    test_image_path = "test_plate_1.jpg"
    if Path(test_image_path).exists():
        image = cv2.imread(test_image_path)

        # 检测车辆
        vehicles = detector.detect_vehicles(image)

        print(f"检测到 {len(vehicles)} 辆车:")
        for i, vehicle in enumerate(vehicles):
            print(".3f"
                  f"位置: {vehicle['bbox']}")

        # 可视化结果
        vis_image = detector.visualize_detections(image, vehicles)
        cv2.imwrite("detection_result.jpg", vis_image)
        print("结果已保存到 detection_result.jpg")

    else:
        print("未找到测试图片")


if __name__ == "__main__":
    # 测试代码
    test_vehicle_detector()
