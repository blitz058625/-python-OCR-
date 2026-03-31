import json
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import hyperlpr3 as lpr3

os.environ.setdefault("PYTHONIOENCODING", "UTF-8")

_log = logging.getLogger(__name__)

try:
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
    _log.debug("HyperLPR3 已加载 (DETECT_LEVEL_HIGH)")
except Exception as e:
    _log.warning("HyperLPR3 高精度初始化失败: %s，尝试默认级别", e)
    try:
        catcher = lpr3.LicensePlateCatcher()
        _log.debug("HyperLPR3 已加载 (默认检测级别)")
    except Exception:
        catcher = None
        _log.error("HyperLPR3 初始化失败，车牌识别将不可用")

# ========== 车牌号格式验证函数 ==========
def is_valid_license_plate(plate_text):
    """
    验证车牌号格式是否有效
    
    参数:
        plate_text: 车牌号文本
    
    返回:
        bool: 是否为有效车牌号
    """
    if not plate_text or not isinstance(plate_text, str):
        return False
    
    plate_text = plate_text.strip()
    
    # 基本长度检查（中国车牌标准长度7位，新能源8位）
    if len(plate_text) < 5 or len(plate_text) > 9:
        return False
    
    # 省份简称列表
    province_chars = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领'
    
    # 检查是否以省份简称开头
    if plate_text[0] not in province_chars:
        # 新能源车牌可能是"电"开头
        if plate_text[0] != '电':
            return False
    
    # 标准格式：省份+字母+5位字母数字（7位）
    # 或：省份+字母+6位字母数字（8位，新能源）
    # 或：电+省份+字母+数字（新能源）
    standard_pattern = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领电][A-Z][A-Z0-9]{5,6}$'
    
    if re.match(standard_pattern, plate_text):
        return True
    
    # 宽松格式：允许6-8位（处理OCR错误）
    loose_pattern = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领电][A-Z][A-Z0-9]{4,7}$'
    if re.match(loose_pattern, plate_text) and 6 <= len(plate_text) <= 8:
        return True
    
    return False

# ========== 车牌OCR识别函数 ==========
def recognize_license_plate(image_input):
    """
    使用 HyperLPR3 识别车牌（优化版：只识别框出的车牌号）

    参数:
        image_input: 车牌图像路径（字符串）或图像数组（numpy array）

    返回:
        识别结果字典
    """
    try:
        # 处理输入：支持文件路径或图像数组
        if isinstance(image_input, str) or hasattr(image_input, '__fspath__'):
            # 文件路径
            image = cv2.imread(str(image_input))
            if image is None:
                _log.warning("无法读取图像: %s", image_input)
                return {
                    "plate_text": None,
                    "plate_texts": [],
                    "plate_count": 0,
                    "all_texts": [],
                    "detections": [],
                    "status": "failed",
                    "error": "无法读取图像"
                }
        else:
            # 图像数组
            image = image_input
        
        # 图像预处理：提高识别率
        # 1. 转换为灰度图（如果需要）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 2. 增强对比度（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. 转回BGR格式（HyperLPR3需要BGR格式）
        if len(image.shape) == 3:
            processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # 调用HyperLPR3识别
        # 返回格式: [[车牌号, 置信度, 类型, 位置], ...]
        if catcher is None:
            _log.error("HyperLPR3 未初始化")
            return {
                "plate_text": None,
                "plate_texts": [],
                "plate_count": 0,
                "all_texts": [],
                "detections": [],
                "status": "failed",
                "error": "HyperLPR3未初始化"
            }
        
        # 使用预处理后的图像进行识别
        results = catcher(processed_image)
        
        if not results:
            # HyperLPR3未检测到车牌
            return {
                "plate_text": None,
                "plate_texts": [],
                "plate_count": 0,
                "all_texts": [],
                "detections": [],
                "status": "no_plate_found",
                "model": "hyperlpr3"
            }
        
        # 提取所有识别到的车牌号，并进行格式验证
        all_plate_candidates = []
        
        for result in results:
            # HyperLPR3返回格式: [车牌号, 置信度, 类型, [x1, y1, x2, y2]]
            if isinstance(result, list) and len(result) >= 2:
                plate_text = result[0] if result[0] else ''
                confidence = result[1] if len(result) > 1 else 0.0
                plate_type = result[2] if len(result) > 2 else 0
                bbox = result[3] if len(result) > 3 else []
            elif hasattr(result, 'plate'):
                plate_text = result.plate
                confidence = getattr(result, 'confidence', 0.0)
                bbox = getattr(result, 'box', [])
            elif isinstance(result, dict):
                plate_text = result.get('plate', '')
                confidence = result.get('confidence', 0.0)
                bbox = result.get('box', [])
            else:
                continue
            
            # 处理numpy类型
            if isinstance(plate_text, np.ndarray):
                plate_text = plate_text.tolist()[0] if plate_text.size > 0 else ''
            if isinstance(confidence, (np.float32, np.float64)):
                confidence = float(confidence)
            
            if plate_text and str(plate_text).strip():
                plate_text = str(plate_text).strip()
                
                # 格式验证：只保留有效的车牌号
                if is_valid_license_plate(plate_text):
                    # 置信度阈值：只保留置信度>=0.3的车牌号（可根据需要调整）
                    if confidence >= 0.3:
                        all_plate_candidates.append({
                            "text": plate_text,
                            "confidence": float(confidence),
                            "type": plate_type,
                            "bbox": bbox if isinstance(bbox, (list, tuple)) else []
                        })
        
        # 如果没有找到有效车牌号
        if not all_plate_candidates:
            return {
                "plate_text": None,
                "plate_texts": [],
                "plate_count": 0,
                "all_texts": [],
                "detections": [],
                "status": "no_valid_plate",
                "model": "hyperlpr3"
            }
        
        # 按置信度排序（降序）
        all_plate_candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # 去重（保留顺序和最高置信度）
        seen = set()
        unique_plates = []
        unique_detections = []
        
        for candidate in all_plate_candidates:
            plate_text = candidate.get('text', '')
            if plate_text and plate_text not in seen:
                seen.add(plate_text)
                unique_plates.append(plate_text)
                unique_detections.append(candidate)
        
        # 对于已裁剪的车牌区域，限制结果数量（通常只有1-3个车牌）
        # 但保留所有有效的车牌号，不强制限制
        if len(unique_plates) > 3:
            # 如果超过3个，只保留置信度最高的3个
            unique_plates = unique_plates[:3]
            unique_detections = unique_detections[:3]
        
        # 主要车牌号（置信度最高的）
        plate_text = unique_plates[0] if unique_plates else None
        
        return {
            "plate_text": plate_text,  # 主要车牌号
            "plate_texts": unique_plates,  # 所有识别到的车牌号列表
            "plate_count": len(unique_plates),  # 车牌号数量
            "all_texts": unique_plates,  # 兼容字段
            "detections": unique_detections,  # 检测结果详情
            "status": "success",
            "model": "hyperlpr3"  # 标记使用的模型
        }
        
    except Exception as e:
        _log.exception("OCR 识别出错: %s", e)
        return {
            "plate_text": None,
            "plate_texts": [],
            "plate_count": 0,
            "all_texts": [],
            "detections": [],
            "status": "error",
            "error": str(e)
        }

# ========== 批量处理函数 ==========
def batch_process_folder(folder_path, output_file="hyperlpr3_results.json"):
    """
    批量处理文件夹中的所有图片
    
    参数:
        folder_path: 包含车牌图片的文件夹路径
        output_file: 输出结果文件路径（JSON格式）
    
    返回:
        所有识别结果的列表
    """
    folder = Path(folder_path)
    if not folder.exists():
        _log.error("文件夹不存在: %s", folder_path)
        return []
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图片文件
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        _log.error("文件夹中无图片: %s", folder_path)
        return []

    _log.info("批量识别 %s 张图片", len(image_files))
    results = []
    success_count = 0
    fail_count = 0
    
    # 处理每张图片
    for idx, image_file in enumerate(image_files, 1):
        try:
            # 调用OCR识别
            result = recognize_license_plate(str(image_file))
            
            if result:
                # 保存结果
                result_item = {
                    "image_file": str(image_file),
                    "image_name": image_file.name,
                    "plate_text": result.get("plate_text"),
                    "plate_texts": result.get("plate_texts", []),
                    "plate_count": result.get("plate_count", 0),
                    "all_texts": result.get("all_texts", []),
                    "detections": result.get("detections", []),
                    "status": "success",
                    "model": "hyperlpr3"  # 只使用HyperLPR3
                }
                results.append(result_item)
                success_count += 1
                
                all_plates = result.get("plate_texts", [])
                if all_plates:
                    _log.debug("[%s/%s] %s -> %s", idx, len(image_files), image_file.name, all_plates)
                else:
                    _log.debug("[%s/%s] %s 无车牌文本", idx, len(image_files), image_file.name)
            else:
                result_item = {
                    "image_file": str(image_file),
                    "image_name": image_file.name,
                    "plate_text": None,
                    "plate_texts": [],
                    "plate_count": 0,
                    "all_texts": [],
                    "detections": [],
                    "status": "failed"
                }
                results.append(result_item)
                fail_count += 1
                _log.warning("[%s/%s] %s 识别失败", idx, len(image_files), image_file.name)

        except Exception as e:
            _log.warning("处理 %s 出错: %s", image_file.name, e)
            result_item = {
                "image_file": str(image_file),
                "image_name": image_file.name,
                "plate_text": None,
                "plate_texts": [],
                "plate_count": 0,
                "all_texts": [],
                "detections": [],
                "status": "failed",
                "error": str(e)
            }
            results.append(result_item)
            fail_count += 1
    
    # 保存结果到JSON文件
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    _log.info("完成: 成功 %s 失败 %s，结果 -> %s", success_count, fail_count, output_file)
    return results

# ========== 主程序 ==========
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="使用 HyperLPR3 进行车牌 OCR")
    parser.add_argument('folder', nargs='?', 
                       default=r'D:\vehicle_detection_project\runs\detect\test_results\predictions',
                       help='包含车牌图片的文件夹路径')
    parser.add_argument('-o', '--output', 
                       default='hyperlpr3_results.json',
                       help='输出结果文件路径（默认: hyperlpr3_results.json）')
    
    args = parser.parse_args()
    
    # 批量处理
    batch_process_folder(args.folder, args.output)
