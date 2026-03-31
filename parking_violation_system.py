import sys
import os
import json
import time
import logging
import cv2
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperlpr3_license_plate import recognize_license_plate
from vehicle_detector import create_vehicle_detector
from plate_voter import PlateVoter
from chinese_text_draw import UnicodeTextOp, cv2_draw_unicode_texts

_log = logging.getLogger(__name__)

TRACKER_AVAILABLE = False
try:
    from byte_tracker import BYTETracker

    TRACKER_AVAILABLE = True
except ImportError as e:
    TRACKER_AVAILABLE = False
    _log.warning("ByteTrack 未加载，将使用简化跟踪: %s", e)

# ========== 系统配置 ==========
SYSTEM_CONFIG = {
    # 视频处理
    "video_source": "d:\\vehicle_detection_project\\runs\\detect\\test_results\\predictions",  # 默认视频源
    "frame_width": 1920,
    "frame_height": 1080,

    # 车辆检测
    "vehicle_conf_threshold": 0.5,
    "vehicle_iou_threshold": 0.45,
    "vehicle_model_path": None,  # YOLOv8模型路径 (.pt文件)
    "vehicle_device": "cpu",  # cpu 或 cuda
    "vehicle_min_box_area": 0,  # 最小框面积(像素)，0=不启用；可设 2000+ 过滤孤立小目标
    "vehicle_suppress_part_detections": True,  # 抑制与车体重叠的小框（如后视镜误检）

    # 车牌识别
    "plate_conf_threshold": 0.3,
    "plate_vote_enabled": True,       # 启用投票纠正识别错误
    "plate_vote_method": "char",      # char=逐字投票(纠单字), string=整串投票
    "plate_vote_min_samples": 2,      # 至少几次识别才输出投票结果
    "plate_vote_max_history": 10,     # 保留最近N次识别用于投票
    # 省份首字：finalize 内按优先级(赣>桂、粤>宁等) + 易混组裁决；bias 为强制替换首字（宁夏/广西车多请删 "宁" 或 "桂"）
    "plate_confusion_resolve": True,
    "plate_province_bias": {"桂": "赣", "宁": "粤"},

    # 违停判断
    "parking_time_limit": 120,  # 2分钟 (120秒)
    "violation_check_interval": 30,  # 30秒检查一次
    "max_tracking_age": 100,  # 最大跟踪帧数
    # 每 N 帧做一次检测/跟踪/车牌（越大越快，轨迹越跳）；整段视频仍会逐帧叠字与写文件
    "frame_skip": 5,
    "stability_threshold": 10,  # 车辆稳定帧数阈值（车辆静止后才进行车牌识别）
    # 跟踪框抖动时适当加大（如 70~100），否则 is_stable 难成立、违停计时会一直重置
    "position_tolerance": 80,  # 位置变化容差（像素）
    "no_parking_zones": [],  # 禁停区域定义

    # 车辆跟踪
    "use_yolo_track": False,  # 使用 YOLO 内置 ByteTracker；False=原检测模式（质量更好）
    "use_tracker": True,  # use_yolo_track=False 时使用内置 BYTETracker
    "track_high_thresh": 0.6,  # 高置信度阈值（提高以减少噪声）
    "track_low_thresh": 0.15,  # 低置信度阈值（提高以减少噪声）
    "new_track_thresh": 0.7,   # 新轨迹创建阈值（提高以减少误创轨迹）
    "track_buffer": 50,        # 轨迹缓冲帧数（增加以提高轨迹稳定性）
    "match_thresh": 0.75,      # 匹配阈值（降低以提高匹配成功率）
    "min_box_area": 150,       # 最小框面积（提高以过滤小目标）

    # 系统参数
    "output_dir": "output",
    "log_level": "INFO",
    "save_violations": True,
    "save_evidence": True
}

# ========== 日志配置 ==========
def setup_logging(log_level: str = "INFO", log_file: str = "parking_system.log"):
    """设置日志系统"""
    # 创建日志目录
    log_dir = Path(SYSTEM_CONFIG["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file

    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 设置日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = level_map.get(log_level.upper(), logging.INFO)

    # 配置根日志器
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 设置第三方库日志级别
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("智能违停检测系统启动")
    logger.info("="*70)

    return logger

# ========== 数据结构 ==========

class VideoProcessor:
    """视频流处理器"""

    def __init__(self, video_source: str, output_path: str = None, fps: int = 30):
        self.video_source = video_source
        self.output_path = output_path
        self.fps = fps
        self.cap = None
        self.writer = None
        self.frame_count = 0

    def initialize(self) -> bool:
        """初始化视频流"""
        try:
            # 打开视频源
            if self.video_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                # 尝试作为摄像头索引
                try:
                    camera_index = int(self.video_source)
                    self.cap = cv2.VideoCapture(camera_index)
                except ValueError:
                    self.cap = cv2.VideoCapture(self.video_source)

            if not self.cap.isOpened():
                _log.error("无法打开视频源: %s", self.video_source)
                return False

            # 获取视频信息
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)

            _log.info(
                "视频流就绪 %sx%s FPS=%s",
                self.frame_width,
                self.frame_height,
                self.video_fps,
            )

            # 初始化视频写入器
            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(
                    self.output_path, fourcc, self.fps or self.video_fps,
                    (self.frame_width, self.frame_height)
                )
                _log.info("输出视频: %s", self.output_path)

            return True

        except Exception as e:
            _log.exception("视频流初始化失败: %s", e)
            return False

    def read_frame(self):
        """读取下一帧"""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            return frame
        else:
            return None

    def write_frame(self, frame):
        """写入帧到输出视频"""
        if self.writer:
            self.writer.write(frame)

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()

    def get_frame_info(self) -> Dict:
        """获取帧信息"""
        return {
            "frame_count": self.frame_count,
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.video_fps
        }


def _bbox_to_int_xyxy(bbox) -> List[int]:
    """将检测/跟踪框转为整数像素坐标（OpenCV 切片、绘制要求整型；ByteTrack/YOLO 常为 float/numpy）。"""
    if bbox is None:
        return [0, 0, 0, 0]
    try:
        coords = list(bbox) if not isinstance(bbox, (list, tuple)) else bbox
        if len(coords) < 4:
            return [0, 0, 0, 0]
        return [int(round(float(coords[i]))) for i in range(4)]
    except (TypeError, ValueError, IndexError):
        return [0, 0, 0, 0]


# ========== 数据结构 ==========
class VehicleInfo:
    """车辆信息类"""
    def __init__(self, vehicle_id: str, bbox: List[int], confidence: float,
                 timestamp: datetime, location: str = "unknown",
                 stability_threshold: int = 10,
                 position_tolerance: int = 50):
        self.vehicle_id = vehicle_id
        self.bbox = _bbox_to_int_xyxy(bbox)  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.timestamp = timestamp
        self.location = location
        self.stability_threshold = stability_threshold
        self.position_tolerance = position_tolerance
        self.plate_info = None
        self.plate_voter = None  # 投票器，按需创建
        self.last_seen = timestamp
        self.frame_count = 1
        self.is_violating = False
        self.violation_start = None

        # 跟踪和稳定性相关
        self.position_history = [self.bbox]  # 位置历史
        self.stability_frames = 0  # 稳定帧数
        self.last_plate_check = None  # 最后车牌检测时间
        self.is_stable = False  # 是否稳定
        self.center_point = self._calculate_center(self.bbox)  # 中心点
        self.last_center = self.center_point  # 上一次中心点
        # 同时在「禁停区内」且「已静止」时起算违停时长（见 update_parking_clock）
        self.zone_stable_since: Optional[datetime] = None

    def _calculate_center(self, bbox: List[int]) -> Tuple[int, int]:
        """计算边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, bbox: List[int], confidence: float, timestamp: datetime):
        """更新车辆信息"""
        bbox = _bbox_to_int_xyxy(bbox)
        # 计算新中心点
        new_center = self._calculate_center(bbox)

        # 检查位置变化
        if self._is_position_stable(new_center):
            self.stability_frames += 1
        else:
            self.stability_frames = 0
            self.is_stable = False

        # 更新位置历史（保留最近10个位置）
        self.position_history.append(bbox)
        if len(self.position_history) > 10:
            self.position_history.pop(0)

        # 更新信息
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.frame_count += 1
        self.last_center = new_center
        self.center_point = new_center

        # 检查是否稳定
        if self.stability_frames >= self.stability_threshold and not self.is_stable:
            self.is_stable = True

    def _is_position_stable(self, new_center: Tuple[int, int], tolerance: Optional[int] = None) -> bool:
        """检查位置是否稳定"""
        if tolerance is None:
            tolerance = self.position_tolerance
        dx = abs(new_center[0] - self.last_center[0])
        dy = abs(new_center[1] - self.last_center[1])
        return dx <= tolerance and dy <= tolerance

    def update_parking_clock(self, in_zone: bool, current_time: datetime) -> None:
        """仅在禁停区内且已判定静止时开始/保持违停计时；离开区域或不再静止则清零。"""
        if in_zone and self.is_stable:
            if self.zone_stable_since is None:
                self.zone_stable_since = current_time
        else:
            self.zone_stable_since = None

    def needs_plate_recognition(self, current_time: datetime) -> bool:
        """判断是否需要进行车牌识别"""
        # 如果没有车牌信息，立即识别
        if self.plate_info is None:
            return True

        # 如果车辆刚刚稳定，开始识别
        if self.is_stable and self.last_plate_check is None:
            return True

        # 如果距离上次识别已经过了一段时间（30秒），重新识别
        if self.last_plate_check and (current_time - self.last_plate_check).total_seconds() > 30:
            return True

        return False

    def update_plate_check_time(self, timestamp: datetime):
        """更新车牌检测时间"""
        self.last_plate_check = timestamp

    def set_plate_info(self, plate_info: Dict):
        """设置车牌信息（直接覆盖，用于兼容）"""
        self.plate_info = plate_info

    def add_plate_recognition_result(self, plate_text: str, plate_result: Dict,
                                     voter_config: Dict) -> Optional[str]:
        """
        添加一次识别结果到投票池，返回投票后的车牌号（可能仍为单次结果）
        voter_config: plate_vote_enabled, plate_vote_method, plate_vote_min_samples, plate_vote_max_history
        """
        if not plate_text or not plate_text.strip():
            return None
        text = plate_text.strip()
        if not voter_config.get("plate_vote_enabled", True):
            self.plate_info = {**(plate_result or {}), "plate_text": text}
            return text
        if self.plate_voter is None:
            self.plate_voter = PlateVoter(
                method=voter_config.get("plate_vote_method", "char"),
                min_samples=voter_config.get("plate_vote_min_samples", 2),
                max_history=voter_config.get("plate_vote_max_history", 10),
                province_bias=voter_config.get("plate_province_bias"),
                confusion_resolve=voter_config.get("plate_confusion_resolve", True),
            )
        self.plate_voter.add(text)
        voted = self.plate_voter.get_voted()
        if voted:
            self.plate_info = {**(plate_result or {}), "plate_text": voted}
            return voted
        # 样本不足，暂用最后一次
        self.plate_info = {**(plate_result or {}), "plate_text": text}
        return text

    def is_in_no_parking_zone(self, no_parking_zones: List[Dict]) -> bool:
        """检查车辆是否在禁停区域"""
        if not no_parking_zones:
            return False

        center_x, center_y = self.center_point

        for zone in no_parking_zones:
            # 简单的矩形区域检查
            if zone.get('type') == 'rectangle':
                x1, y1, x2, y2 = zone['bbox']
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    return True

        return False

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "vehicle_id": self.vehicle_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "plate_info": self.plate_info,
            "last_seen": self.last_seen.isoformat(),
            "frame_count": self.frame_count,
            "is_violating": self.is_violating,
            "violation_start": self.violation_start.isoformat() if self.violation_start else None,
            "stability_frames": self.stability_frames,
            "is_stable": self.is_stable,
            "center_point": self.center_point,
            "last_plate_check": self.last_plate_check.isoformat() if self.last_plate_check else None,
            "zone_stable_since": self.zone_stable_since.isoformat() if self.zone_stable_since else None,
        }

# ========== 车辆跟踪器 ==========
class VehicleTracker:
    """
    车辆跟踪器封装类
    基于ByteTrack的多目标跟踪
    """

    def __init__(self, config: Dict):
        """
        初始化车辆跟踪器

        参数:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化ByteTrack跟踪器
        if TRACKER_AVAILABLE and config.get("use_tracker", True):
            self.tracker = BYTETracker(
                track_high_thresh=config.get("track_high_thresh", 0.5),
                track_low_thresh=config.get("track_low_thresh", 0.1),
                new_track_thresh=config.get("new_track_thresh", 0.6),
                track_buffer=config.get("track_buffer", 30),
                match_thresh=config.get("match_thresh", 0.8),
                min_box_area=config.get("min_box_area", 100),
                frame_rate=30
            )
            self.use_tracker = True
            self.logger.info("[OK] ByteTrack车辆跟踪器初始化成功")
        else:
            self.tracker = None
            self.use_tracker = False
            self.logger.warning("使用简化的车辆跟踪逻辑")

        # 跟踪状态记录
        self.tracked_vehicles = {}  # track_id -> VehicleInfo
        self.frame_count = 0
        self.next_vehicle_id = 1  # 全局车辆ID计数器，确保ID唯一性

    def _new_vehicle_info(self, vehicle_id: str, bbox, confidence: float, current_time: datetime) -> VehicleInfo:
        return VehicleInfo(
            vehicle_id,
            bbox,
            confidence,
            current_time,
            stability_threshold=self.config.get("stability_threshold", 10),
            position_tolerance=self.config.get("position_tolerance", 50),
        )

    def update_tracks(self, detections: List[Dict], current_time: datetime) -> List[VehicleInfo]:
        """
        更新车辆跟踪

        参数:
            detections: 检测结果列表，每个元素包含 'bbox', 'confidence', 'class_name'
                       若来自 YOLO model.track() 则包含 'track_id'
            current_time: 当前时间

        返回:
            跟踪到的车辆列表
        """
        self.frame_count += 1

        # 若检测结果已带 track_id（来自 YOLO model.track() ByteTracker），直接使用
        if any(d.get('track_id') is not None for d in detections):
            return self._update_from_yolo_tracks(detections, current_time)

        if self.use_tracker and self.tracker:
            return self._update_with_bytetrack(detections, current_time)
        else:
            return self._update_simple(detections, current_time)

    def _update_with_bytetrack(self, detections: List[Dict], current_time: datetime) -> List[VehicleInfo]:
        """使用ByteTrack更新跟踪"""
        if not detections:
            # 没有检测结果，清空跟踪器状态
            self.tracker.update([], [], [])
            return list(self.tracked_vehicles.values())

        # 提取检测框和置信度
        bboxes = []
        scores = []
        class_ids = []

        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            confidence = det.get('confidence', 0.0)
            class_name = det.get('class_name', 'vehicle')

            # 转换为ByteTrack期望的格式
            bboxes.append(bbox)
            scores.append(confidence)
            class_ids.append(0)  # 车辆类别ID设为0

        # 更新跟踪器
        tracked_stracks = self.tracker.update(bboxes, scores, class_ids)

        # 处理跟踪结果
        current_vehicles = []

        for strack in tracked_stracks:
            track_id = strack.track_id
            bbox = strack.tlbr  # [x1, y1, x2, y2]
            confidence = strack.score

            # 获取或创建车辆信息
            if track_id in self.tracked_vehicles:
                vehicle = self.tracked_vehicles[track_id]
                # 更新车辆信息
                vehicle.update(bbox, confidence, current_time)
            else:
                # 创建新车辆
                vehicle = self._new_vehicle_info(f"vehicle_{track_id}", bbox, confidence, current_time)
                self.tracked_vehicles[track_id] = vehicle

            current_vehicles.append(vehicle)

        # 清理长时间未出现的车辆（更保守的策略）
        expired_tracks = []
        for track_id, vehicle in self.tracked_vehicles.items():
            time_since_last_seen = (current_time - vehicle.last_seen).total_seconds()
            # 如果车辆从未被确认跟踪且超过30秒，或确认跟踪但超过120秒
            if (vehicle.frame_count <= 3 and time_since_last_seen > 30) or time_since_last_seen > 120:
                expired_tracks.append(track_id)

        for track_id in expired_tracks:
            self.logger.debug(f"清理过期车辆: {track_id}")
            del self.tracked_vehicles[track_id]

        return current_vehicles

    def _update_from_yolo_tracks(self, detections: List[Dict], current_time: datetime) -> List[VehicleInfo]:
        """使用 YOLO model.track() 返回的 track_id 更新车辆（内置 ByteTracker）"""
        current_vehicles = []

        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue

            bbox = det['bbox']
            confidence = det.get('confidence', 0.0)
            vehicle_id = f"vehicle_{track_id}"

            if vehicle_id in self.tracked_vehicles:
                vehicle = self.tracked_vehicles[vehicle_id]
                vehicle.update(bbox, confidence, current_time)
            else:
                vehicle = self._new_vehicle_info(vehicle_id, bbox, confidence, current_time)
                self.tracked_vehicles[vehicle_id] = vehicle

            current_vehicles.append(vehicle)

        # 清理长时间未出现的车辆
        expired = [vid for vid, v in self.tracked_vehicles.items()
                   if (current_time - v.last_seen).total_seconds() > 120]
        for vid in expired:
            del self.tracked_vehicles[vid]

        return current_vehicles

    def _update_simple(self, detections: List[Dict], current_time: datetime) -> List[VehicleInfo]:
        """简化的跟踪逻辑（基于位置重叠和时间连续性）"""
        current_vehicles = []

        for det in detections:
            bbox = det['bbox']
            confidence = det.get('confidence', 0.0)

            # 改进的车辆匹配逻辑：结合IOU和时间连续性
            best_match = None
            best_score = 0.0

            for vehicle_id, vehicle in self.tracked_vehicles.items():
                iou = self._calculate_iou(bbox, vehicle.bbox)

                # 计算时间间隔（秒）
                time_diff = (current_time - vehicle.last_seen).total_seconds()

                # 如果时间间隔太长（超过5秒），降低匹配优先级
                time_weight = max(0.1, 1.0 - time_diff / 10.0)  # 10秒内权重线性衰减

                # 综合评分：IOU * 时间权重 * 置信度权重
                confidence_weight = min(1.0, confidence / 0.5)  # 置信度高于0.5时权重为1
                score = iou * time_weight * confidence_weight

                if score > best_score and (iou > 0.3 or (iou > 0.1 and time_diff < 2.0)):
                    best_match = vehicle_id
                    best_score = score

            if best_match:
                # 更新现有车辆
                vehicle = self.tracked_vehicles[best_match]
                vehicle.update(bbox, confidence, current_time)
                current_vehicles.append(vehicle)
            else:
                # 创建新车辆，使用全局ID计数器确保ID唯一性
                vehicle_id = f"vehicle_{self.next_vehicle_id}"
                self.next_vehicle_id += 1
                vehicle = self._new_vehicle_info(vehicle_id, bbox, confidence, current_time)
                self.tracked_vehicles[vehicle_id] = vehicle
                current_vehicles.append(vehicle)

        return current_vehicles

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算两个边界框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_tracking_stats(self) -> Dict:
        """获取跟踪统计信息"""
        # 若通过 update_tracks 传入带 track_id 的检测，则为 YOLO ByteTrack
        tracker_type = "YOLO ByteTrack" if self.config.get("use_yolo_track", False) else (
            "ByteTrack" if self.use_tracker else "Simple"
        )
        return {
            "total_tracked_vehicles": len(self.tracked_vehicles),
            "tracker_type": tracker_type,
            "frame_count": self.frame_count
        }

    def reset(self):
        """重置跟踪器"""
        if self.tracker:
            self.tracker.reset()
        self.tracked_vehicles.clear()
        self.frame_count = 0
        self.logger.info("车辆跟踪器已重置")


class ViolationRecord:
    """违停记录类"""
    def __init__(self, violation_id: str, vehicle: VehicleInfo):
        self.violation_id = violation_id
        self.vehicle_id = vehicle.vehicle_id
        self.plate_number = vehicle.plate_info.get('plate_text') if vehicle.plate_info else None
        self.start_time = vehicle.violation_start or vehicle.timestamp
        self.end_time = None
        self.duration = 0
        self.location = vehicle.location
        self.evidence_images = []
        self.status = "active"  # active, resolved, false_positive

    def update_duration(self, current_time: datetime):
        """更新违停时长"""
        self.end_time = current_time
        self.duration = (current_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "violation_id": self.violation_id,
            "vehicle_id": self.vehicle_id,
            "plate_number": self.plate_number,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "location": self.location,
            "evidence_images": self.evidence_images,
            "status": self.status
        }

# ========== 核心模块类 ==========
class ParkingViolationSystem:
    """智能违停检测系统主类"""

    def __init__(self, config: Dict = None):
        """初始化系统"""
        self.config = SYSTEM_CONFIG.copy()
        if config:
            self.config.update(config)

        self.logger = setup_logging(self.config["log_level"])

        # 创建输出目录
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(exist_ok=True)

        # 初始化组件
        self.vehicle_detector = None  # YOLOv8车辆检测器
        self.plate_recognizer = None  # 车牌识别器
        self.data_manager = None      # 数据管理器
        self.video_processor = None   # 视频处理器
        self.vehicle_tracker = None   # 车辆跟踪器

        # 跟踪状态
        self.violations = {}      # violation_id -> ViolationRecord
        self.next_violation_id = 1

        # 禁停区域配置
        self.no_parking_zones = self._load_no_parking_zones()

        # 统计信息
        self.stats = {
            "total_frames": 0,
            "vehicles_detected": 0,
            "plates_recognized": 0,
            "violations_detected": 0,
            "start_time": datetime.now()
        }

        # 视频/摄像头模式：视频文件使用视频时间线，摄像头使用真实时间
        self._use_video_time = False
        self._video_fps = 30
        self._video_time_base = datetime(2000, 1, 1)

        self.logger.info("系统初始化完成")

    def initialize_components(self):
        """初始化各个组件"""
        self.logger.info("初始化系统组件...")

        # 初始化车牌识别器（HyperLPR3）
        try:
            self.plate_recognizer = {
                "recognize": recognize_license_plate,
                "config": {
                    "conf_threshold": self.config["plate_conf_threshold"]
                }
            }
            self.logger.info("[OK] HyperLPR3车牌识别器初始化成功")
        except Exception as e:
            self.logger.error(f"[ERROR] 车牌识别器初始化失败: {e}")
            return False

        # 初始化车辆检测器 (YOLOv8)
        try:
            self.vehicle_detector = create_vehicle_detector(
                model_path=self.config["vehicle_model_path"],
                conf_threshold=self.config["vehicle_conf_threshold"],
                iou_threshold=self.config["vehicle_iou_threshold"],
                device=self.config["vehicle_device"],
                min_box_area=self.config.get("vehicle_min_box_area", 0),
                suppress_part_detections=self.config.get("vehicle_suppress_part_detections", True)
            )
            self.logger.info("[OK] YOLOv8车辆检测器初始化成功")

            # 显示模型信息
            model_info = self.vehicle_detector.get_model_info()
            if model_info["status"] == "loaded":
                self.logger.info(f"  模型路径: {model_info.get('model_path', '预训练模型')}")
                self.logger.info(f"  置信度阈值: {model_info['conf_threshold']}")
                self.logger.info(f"  IOU阈值: {model_info['iou_threshold']}")
                self.logger.info(f"  运行设备: {model_info['device']}")

        except Exception as e:
            self.logger.error(f"[ERROR] YOLOv8车辆检测器初始化失败: {e}")
            self.logger.warning("将使用模拟车辆检测进行测试")
            self.vehicle_detector = None

        # 初始化数据管理器
        self.data_manager = DataManager(self.output_dir / "data")
        self.logger.info("[OK] 数据管理器初始化成功")

        # 初始化车辆跟踪器
        self.vehicle_tracker = VehicleTracker(self.config)
        tracking_stats = self.vehicle_tracker.get_tracking_stats()
        self.logger.info(f"[OK] 车辆跟踪器初始化成功 - 类型: {tracking_stats['tracker_type']}")
        if self.config.get("use_yolo_track", True):
            self.logger.info("   [ByteTracker] 使用 YOLO model.track() 内置 ByteTracker")

        return True

    def process_video_source(self, video_source: str):
        """处理视频源"""
        self.logger.info(f"开始处理视频源: {video_source}")

        # 检查视频源类型
        if os.path.isdir(video_source):
            # 处理图片文件夹
            return self.process_image_folder(video_source)
        elif os.path.isfile(video_source) and video_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # 处理视频文件
            return self.process_video_file(video_source)
        else:
            # 尝试作为摄像头或图片
            try:
                camera_index = int(video_source)
                return self.process_camera(camera_index)
            except ValueError:
                # 检查是否是图片文件
                if os.path.isfile(video_source) and video_source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    return self.process_single_image(video_source)
                else:
                    self.logger.error(f"无效的视频源: {video_source}")
                    return False

    def process_video_file(self, video_path: str) -> bool:
        """处理视频文件"""
        self.logger.info(f"处理视频文件: {video_path}")

        # 初始化视频处理器
        output_video = self.output_dir / f"processed_{Path(video_path).stem}.mp4"
        self.video_processor = VideoProcessor(video_path, str(output_video))

        if not self.video_processor.initialize():
            return False

        # 使用视频时间而非真实时间，确保违停判定基于视频内容时长（不受处理速度影响）
        video_fps = self.video_processor.video_fps or 30
        if video_fps <= 0:
            video_fps = 30
        video_time_base = datetime(2000, 1, 1)  # 视频时间基准
        self._use_video_time = True
        self._video_fps = video_fps
        self._video_time_base = video_time_base
        self.logger.info(f"视频模式: 使用视频时间线 (FPS={video_fps}), 违停阈值={self.config['parking_time_limit']}秒")

        frame_count = 0
        wall_start = time.perf_counter()
        try:
            while True:
                frame = self.video_processor.read_frame()
                if frame is None:
                    break

                frame_count += 1

                # 每 frame_skip 帧做一次检测/跟踪/车牌（其余帧仍叠字与输出，便于回看）
                if frame_count % self.config["frame_skip"] == 0:
                    self.logger.debug(f"处理第 {frame_count} 帧")
                    # 使用视频时间戳：帧号/帧率 = 视频内经过的秒数
                    current_video_time = video_time_base + timedelta(seconds=frame_count / video_fps)
                    self.process_video_frame(frame, frame_count, current_time=current_video_time)

                # 绘制结果并写入输出视频
                result_frame = self.draw_results(frame)
                self.video_processor.write_frame(result_frame)

        except KeyboardInterrupt:
            self.logger.info("用户中断视频处理")
        except Exception as e:
            self.logger.error(f"视频处理出错: {e}")
        finally:
            self.video_processor.release()
            wall_s = time.perf_counter() - wall_start
            video_duration_s = (frame_count / video_fps) if frame_count and video_fps else 0.0
            self.logger.info(
                f"视频处理完成，共 {frame_count} 帧；墙钟耗时 {wall_s:.1f}s，"
                f"视频时长约 {video_duration_s:.1f}s（检测每 {self.config['frame_skip']} 帧一次）"
            )

        return True

    def process_camera(self, camera_index: int) -> bool:
        """处理摄像头输入"""
        self.logger.info(f"处理摄像头: {camera_index}")
        self._use_video_time = False  # 摄像头使用真实时间

        self.video_processor = VideoProcessor(str(camera_index))

        if not self.video_processor.initialize():
            return False

        frame_count = 0
        try:
            while True:
                frame = self.video_processor.read_frame()
                if frame is None:
                    break

                frame_count += 1

                # 每5帧处理一次
                if frame_count % self.config["frame_skip"] == 0:
                    self.logger.debug(f"处理第 {frame_count} 帧")
                    self.process_video_frame(frame, frame_count)

                # 显示结果
                result_frame = self.draw_results(frame)
                cv2.imshow('Parking Violation Detection', result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            self.logger.info("用户中断摄像头处理")
        finally:
            cv2.destroyAllWindows()
            self.video_processor.release()
            self.logger.info(f"摄像头处理完成，共处理 {frame_count} 帧")

        return True

    def process_single_image(self, image_path: str) -> bool:
        """处理单张图片"""
        self.logger.info(f"处理单张图片: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"无法读取图片: {image_path}")
            return False

        # 处理图片
        self.process_video_frame(image, 1)

        # 保存结果
        result_image = self.draw_results(image)
        output_path = self.output_dir / f"result_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(output_path), result_image)
        self.logger.info(f"结果已保存: {output_path}")

        return True

    def process_video_frame(self, frame, frame_id: int, current_time: datetime = None):
        """处理视频帧（每5帧调用一次）
        current_time: 可选，视频文件模式下传入视频时间戳；摄像头模式使用 datetime.now()
        """
        self.stats["total_frames"] += 1
        if current_time is None:
            current_time = datetime.now()

        # 车辆检测（use_yolo_track 时使用 model.track() 走 ByteTracker）
        if self.vehicle_detector is not None:
            if self.config.get("use_yolo_track", True):
                detections = self.vehicle_detector.detect_vehicles_with_tracking(frame)
            else:
                detections = self.vehicle_detector.detect_vehicles(frame)
        else:
            detections = self._mock_vehicle_detection(frame, frame_id)
            self.logger.warning("使用模拟车辆检测，请集成YOLOv8模型")

        # 使用跟踪器更新车辆状态
        tracked_vehicles = self.vehicle_tracker.update_tracks(detections, current_time)

        # 处理跟踪到的车辆
        for vehicle in tracked_vehicles:
            self.process_tracked_vehicle(frame, vehicle, current_time)

        # 检查违停情况
        self.check_violations(current_time)

        # 更新统计信息
        tracking_stats = self.vehicle_tracker.get_tracking_stats()
        self.stats["tracked_vehicles"] = tracking_stats["total_tracked_vehicles"]

    def process_tracked_vehicle(self, frame, vehicle: VehicleInfo, current_time: datetime):
        """处理已跟踪的车辆"""
        bbox = vehicle.bbox
        x1, y1, x2, y2 = bbox

        # 检查是否需要进行车牌识别
        if vehicle.needs_plate_recognition(current_time):
            # 提取车辆区域进行车牌识别
            vehicle_region = frame[y1:y2, x1:x2]

            if vehicle_region.size > 0:
                try:
                    plate_result = self.plate_recognizer["recognize"](vehicle_region)

                    if plate_result and plate_result.get('plate_text'):
                        plate_text = vehicle.add_plate_recognition_result(
                            plate_result['plate_text'], plate_result, self.config
                        )
                        vehicle.update_plate_check_time(current_time)
                        if plate_text:
                            self.stats["plates_recognized"] += 1
                            self.logger.info(f"车辆 {vehicle.vehicle_id}: 车牌 {plate_text}")
                    else:
                        self.logger.debug(f"车辆 {vehicle.vehicle_id}: 未识别到车牌")

                except Exception as e:
                    self.logger.error(f"车牌识别出错: {e}")

        # 更新统计
        self.stats["vehicles_detected"] += 1


    def draw_results(self, frame):
        """在帧上绘制结果"""
        result_frame = frame.copy()
        text_ops: List[UnicodeTextOp] = []

        # 获取当前跟踪的车辆
        tracked_vehicles = self.vehicle_tracker.tracked_vehicles if self.vehicle_tracker else {}

        for vehicle_id, vehicle in tracked_vehicles.items():
            bbox = vehicle.bbox
            x1, y1, x2, y2 = bbox

            # 选择颜色：红色表示违停，绿色表示正常
            color = (0, 0, 255) if vehicle.is_violating else (0, 255, 0)
            thickness = 3 if vehicle.is_violating else 2

            # 绘制车辆边界框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)

            # 绘制车辆信息
            info_lines = []

            # 车牌号
            if vehicle.plate_info and vehicle.plate_info.get('plate_text'):
                plate_text = vehicle.plate_info['plate_text']
                info_lines.append(f"车牌: {plate_text}")

            # 车辆ID和状态
            status = "违停" if vehicle.is_violating else "正常"
            info_lines.append(f"ID: {vehicle_id} | {status}")

            # 稳定状态
            if vehicle.is_stable:
                info_lines.append(f"稳定: {vehicle.stability_frames}帧")

            # 中文标注收集后批量绘制（避免每行整帧 BGR↔PIL，大幅加速视频处理）
            for i, line in enumerate(info_lines):
                text_ops.append(
                    UnicodeTextOp(
                        text=line,
                        org=(x1, y1 - 10 - i * 25),
                        font_size_px=18,
                        color_bgr=color,
                        anchor="ls",
                    )
                )

            # 如果是违停车辆，添加特殊标记
            if vehicle.is_violating:
                # 在车辆中心绘制违停标志
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(result_frame, (center_x, center_y), 20, (0, 0, 255), -1)
                text_ops.append(
                    UnicodeTextOp(
                        text="违停",
                        org=(center_x, center_y),
                        font_size_px=22,
                        color_bgr=(255, 255, 255),
                        anchor="mm",
                    )
                )

        # 绘制禁停区域（几何 + 收集文字）
        result_frame, zone_text_ops = self.draw_no_parking_zones(result_frame)
        text_ops.extend(zone_text_ops)

        # 绘制统计信息（仅显示违停数量）
        violating_count = len([v for v in self.vehicle_tracker.tracked_vehicles.values() if v.is_violating]) if self.vehicle_tracker else 0
        stats_text = f"违停: {violating_count}"
        text_ops.append(
            UnicodeTextOp(
                text=stats_text,
                org=(10, 30),
                font_size_px=24,
                color_bgr=(255, 255, 255),
                anchor="ls",
            )
        )

        # 绘制时间戳
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text_ops.append(
            UnicodeTextOp(
                text=current_time,
                org=(10, result_frame.shape[0] - 10),
                font_size_px=18,
                color_bgr=(255, 255, 255),
                anchor="ls",
            )
        )

        result_frame = cv2_draw_unicode_texts(result_frame, text_ops)
        return result_frame

    def _load_no_parking_zones(self):
        """加载禁停区域配置"""
        # 默认禁停区域配置（可以从配置文件加载）
        default_zones = [
            {
                "name": "主路段禁停区",
                "type": "rectangle",
                "bbox": [100, 200, 800, 600],  # [x1, y1, x2, y2]
                "color": (0, 0, 255),  # 红色
                "description": "主干道禁停区域"
            },
            {
                "name": "入口禁停区",
                "type": "rectangle",
                "bbox": [1200, 300, 1800, 500],  # [x1, y1, x2, y2]
                "color": (0, 0, 255),  # 红色
                "description": "入口区域禁止停车"
            }
        ]

        # 尝试从配置文件加载
        config_file = self.output_dir / "no_parking_zones.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    custom_zones = json.load(f)
                self.logger.info(f"已加载自定义禁停区域配置: {config_file}")
                return custom_zones
            except Exception as e:
                self.logger.warning(f"加载禁停区域配置失败，使用默认配置: {e}")

        self.logger.info("使用默认禁停区域配置")
        return default_zones

    def save_no_parking_zones(self):
        """保存禁停区域配置"""
        config_file = self.output_dir / "no_parking_zones.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.no_parking_zones, f, indent=2, ensure_ascii=False)
            self.logger.info(f"禁停区域配置已保存: {config_file}")
        except Exception as e:
            self.logger.error(f"保存禁停区域配置失败: {e}")

    def add_no_parking_zone(self, name: str, bbox: List[int], zone_type: str = "rectangle",
                           description: str = ""):
        """添加禁停区域"""
        zone = {
            "name": name,
            "type": zone_type,
            "bbox": bbox,
            "color": (0, 0, 255),  # 红色
            "description": description
        }
        self.no_parking_zones.append(zone)
        self.save_no_parking_zones()
        self.logger.info(f"已添加禁停区域: {name}")

    def draw_no_parking_zones(self, frame):
        """在帧上绘制禁停区域几何；中文标签以 UnicodeTextOp 列表形式返回，由 draw_results 批量绘制。"""
        text_ops: List[UnicodeTextOp] = []
        for zone in self.no_parking_zones:
            if zone["type"] == "rectangle":
                x1, y1, x2, y2 = zone["bbox"]
                color = zone["color"]
                thickness = 3
                color_bgr = tuple(int(c) for c in color)

                # 绘制禁停区域边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # 绘制半透明填充
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                alpha = 0.2  # 透明度
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # 区域名称（延后批量绘制）
                label = f"禁停区: {zone['name']}"
                text_ops.append(
                    UnicodeTextOp(
                        text=label,
                        org=(x1, y1 - 10),
                        font_size_px=18,
                        color_bgr=color_bgr,
                        anchor="ls",
                    )
                )

                # 绘制禁止停车图标（简化版）
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 30, color, 3)
                cv2.line(frame, (center_x - 20, center_y - 20),
                        (center_x + 20, center_y + 20), color, 3)

        return frame, text_ops

    def process_image_folder(self, folder_path: str) -> bool:
        """处理图片文件夹（用于测试）"""
        folder = Path(folder_path)
        if not folder.exists():
            self.logger.error(f"文件夹不存在: {folder_path}")
            return False

        # 获取图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))

        if not image_files:
            self.logger.error(f"文件夹中没有找到图片文件: {folder_path}")
            return False

        self.logger.info(f"找到 {len(image_files)} 个图片文件，开始批量处理")

        # 处理每张图片
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"[{i}/{len(image_files)}] 处理: {image_file.name}")

            try:
                # 读取图片
                import cv2
                image = cv2.imread(str(image_file))
                if image is None:
                    self.logger.warning(f"无法读取图片: {image_file}")
                    continue

                # 处理单帧
                self.process_frame(image, f"frame_{i}", image_file.name)

                # 模拟时间间隔
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"处理图片出错: {e}")

        self.logger.info("图片文件夹处理完成")
        return True

    def process_frame(self, frame, frame_id: str, source_info: str = ""):
        """处理单帧图像"""
        self.stats["total_frames"] += 1

        # 车辆检测
        if self.vehicle_detector is not None:
            # 使用YOLOv8进行真实车辆检测
            vehicles = self.vehicle_detector.detect_vehicles(frame)
            self.logger.debug(f"检测到 {len(vehicles)} 辆车")
        else:
            # 备用：模拟车辆检测（用于测试）
            vehicles = self._mock_vehicle_detection(frame, frame_id)
            self.logger.warning("使用模拟车辆检测，请集成YOLOv8模型以获得更好效果")

        # 处理检测到的车辆
        for vehicle_data in vehicles:
            self.process_vehicle_frame(frame, vehicle_data, datetime.now())

        # 检查违停情况
        self.check_violations(datetime.now())

        # 定期清理过期跟踪
        if self.stats["total_frames"] % 100 == 0:
            self.cleanup_expired_tracks()

    def _mock_vehicle_detection(self, frame, frame_id: str) -> List[List[int]]:
        """模拟车辆检测（用于测试）"""
        # 这里应该调用真实的车辆检测模型
        # 暂时返回一些模拟的车辆边界框

        height, width = frame.shape[:2]

        # 模拟检测到1-3辆车
        import random
        num_vehicles = random.randint(1, 3)

        vehicles = []
        for i in range(num_vehicles):
            # 生成随机边界框
            x1 = random.randint(0, width // 2)
            y1 = random.randint(height // 3, 2 * height // 3)
            x2 = x1 + random.randint(100, 300)
            y2 = y1 + random.randint(50, 150)

            # 确保边界框在图像范围内
            x2 = min(x2, width)
            y2 = min(y2, height)

            vehicles.append([x1, y1, x2, y2])

        return vehicles


    def check_violations(self, current_time: datetime):
        """检查违停情况"""
        tracked_vehicles = self.vehicle_tracker.tracked_vehicles if self.vehicle_tracker else {}
        for vehicle_id, vehicle in tracked_vehicles.items():
            in_zone = vehicle.is_in_no_parking_zone(self.no_parking_zones)
            vehicle.update_parking_clock(in_zone, current_time)

            if not in_zone:
                continue

            if not vehicle.is_stable:
                continue

            if vehicle.zone_stable_since is None:
                continue

            parking_duration = (current_time - vehicle.zone_stable_since).total_seconds()

            # 检查是否超过违停阈值（2分钟）
            if parking_duration > self.config["parking_time_limit"]:
                if not vehicle.is_violating:
                    # 开始违停
                    vehicle.is_violating = True
                    vehicle.violation_start = current_time

                    # 创建违停记录
                    violation = ViolationRecord(f"violation_{self.next_violation_id}", vehicle)
                    self.violations[violation.violation_id] = violation
                    self.next_violation_id += 1
                    self.stats["violations_detected"] += 1

                    plate_text = "未知"
                    if vehicle.plate_info and vehicle.plate_info.get('plate_text'):
                        plate_text = vehicle.plate_info['plate_text']

                    self.logger.warning(f"检测到违停: 车辆 {vehicle_id}, "
                                      f"车牌 {plate_text}, "
                                      f"停车时长 {parking_duration:.1f}秒")

                else:
                    # 更新违停时长
                    if vehicle_id in [v.vehicle_id for v in self.violations.values()]:
                        violation = next(v for v in self.violations.values() if v.vehicle_id == vehicle_id)
                        violation.update_duration(current_time)


    def save_results(self):
        """保存处理结果"""
        try:
            # 保存统计信息
            stats_file = self.output_dir / "stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False, default=str)

            # 保存当前车辆跟踪
            tracks_file = self.output_dir / "current_tracks.json"
            if self.vehicle_tracker:
                tracks_data = {vid: vehicle.to_dict() for vid, vehicle in self.vehicle_tracker.tracked_vehicles.items()}
            else:
                tracks_data = {}
            with open(tracks_file, 'w', encoding='utf-8') as f:
                json.dump(tracks_data, f, indent=2, ensure_ascii=False, default=str)

            # 保存违停记录
            violations_file = self.output_dir / "violations.json"
            violations_data = {vid: violation.to_dict() for vid, violation in self.violations.items()}
            with open(violations_file, 'w', encoding='utf-8') as f:
                json.dump(violations_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info("结果已保存到输出目录")

        except Exception as e:
            self.logger.error(f"保存结果出错: {e}")

    def print_stats(self):
        """打印统计信息"""
        self.logger.info("="*70)
        self.logger.info("处理统计:")
        self.logger.info("="*70)
        self.logger.info(f"总帧数: {self.stats['total_frames']}")
        self.logger.info(f"跟踪到车辆: {self.stats.get('tracked_vehicles', 0)}")
        self.logger.info(f"识别到车牌: {self.stats['plates_recognized']}")
        self.logger.info(f"检测到违停: {self.stats['violations_detected']}")
        self.logger.info(f"运行时间: {(datetime.now() - self.stats['start_time']).total_seconds():.1f} 秒")
        self.logger.info("="*70)

    def run(self, video_source: str = None):
        """运行系统"""
        if video_source is None:
            video_source = self.config["video_source"]

        try:
            # 初始化组件
            if not self.initialize_components():
                self.logger.error("组件初始化失败")
                return False

            # 处理视频源
            success = self.process_video_source(video_source)

            if success:
                # 保存结果
                self.save_results()
                self.print_stats()
                self.logger.info("系统运行完成")
                return True
            else:
                self.logger.error("视频源处理失败")
                return False

        except KeyboardInterrupt:
            self.logger.info("用户中断")
            self.save_results()
            self.print_stats()
            return True

        except Exception as e:
            self.logger.error(f"系统运行出错: {e}")
            import traceback
            traceback.print_exc()
            return False

# ========== 数据管理模块 ==========
class DataManager:
    """数据管理类"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

    def save_violation(self, violation: ViolationRecord):
        """保存违停记录"""
        # 这里可以实现数据库存储或其他持久化方案
        pass

    def get_violations(self, start_time=None, end_time=None, location=None):
        """查询违停记录"""
        # 这里可以实现查询功能
        pass

# ========== 工具函数 ==========
def load_config(config_file: str) -> Dict:
    """加载配置文件"""
    config_path = Path(config_file)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config: Dict, config_file: str):
    """保存配置文件"""
    config_path = Path(config_file)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

# ========== 主函数 ==========
def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='智能违停检测系统')
    parser.add_argument('video_source', nargs='?',
                       default=SYSTEM_CONFIG['video_source'],
                       help='视频源路径（文件夹、视频文件或摄像头索引）')
    parser.add_argument('-c', '--config',
                       help='配置文件路径')
    parser.add_argument('-o', '--output',
                       default=SYSTEM_CONFIG['output_dir'],
                       help='输出目录')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default=SYSTEM_CONFIG['log_level'],
                       help='日志级别')

    args = parser.parse_args()

    # 加载配置
    config = SYSTEM_CONFIG.copy()
    if args.config:
        config.update(load_config(args.config))

    # 更新命令行参数
    config['video_source'] = args.video_source
    config['output_dir'] = args.output
    config['log_level'] = args.log_level

    # 创建系统实例
    system = ParkingViolationSystem(config)

    # 运行系统
    success = system.run(args.video_source)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
