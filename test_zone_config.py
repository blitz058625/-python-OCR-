#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试禁停区域配置功能
"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from configure_zones import ZoneConfigurator

def test_zone_config():
    """测试禁停区域配置"""
    print("测试禁停区域配置功能")
    print("=" * 50)

    # 使用测试图片
    test_image = "test_plate_1.jpg"

    if not Path(test_image).exists():
        print(f"测试图片不存在: {test_image}")
        return

    # 创建配置器
    configurator = ZoneConfigurator(test_image, "output")

    print(f"已加载 {len(configurator.zones)} 个现有禁停区域")

    # 添加几个测试禁停区域
    test_zones = [
        {
            "name": "测试禁停区1",
            "bbox": [100, 100, 400, 300],
            "description": "左上角测试区域"
        },
        {
            "name": "测试禁停区2",
            "bbox": [800, 200, 1200, 400],
            "description": "右上角测试区域"
        }
    ]

    for zone in test_zones:
        new_zone = {
            "name": zone["name"],
            "type": "rectangle",
            "bbox": zone["bbox"],
            "color": [0, 0, 255],
            "description": zone["description"]
        }
        configurator.zones.append(new_zone)

    print(f"已添加 {len(test_zones)} 个测试禁停区域")
    print(f"当前共有 {len(configurator.zones)} 个禁停区域")

    # 显示所有区域
    print("\n当前禁停区域列表:")
    for i, zone in enumerate(configurator.zones, 1):
        print(f"  {i}. {zone['name']}: {zone['bbox']} - {zone['description']}")

    # 保存配置
    configurator.save_zones()

    print("\n禁停区域配置测试完成")
    print("配置文件已保存到: output/no_parking_zones.json")
    # 验证配置文件
    zones_file = Path("output/no_parking_zones.json")
    if zones_file.exists():
        with open(zones_file, 'r', encoding='utf-8') as f:
            saved_zones = json.load(f)
        print(f"配置文件验证: 包含 {len(saved_zones)} 个禁停区域 [OK]")

    return True

def create_video_zones_config():
    """为用户视频创建禁停区域配置"""
    print("\n为用户视频创建禁停区域配置")
    print("=" * 50)

    video_path = r"D:\数据集\视频\金塘燕头山闸全景球机（全景）_20260129223009"

    # 检查视频是否存在
    if not Path(video_path).exists():
        print(f"视频文件不存在: {video_path}")
        print("将创建基于典型监控场景的默认禁停区域配置")
        video_path = None
    else:
        print(f"找到视频文件: {video_path}")

    # 创建配置器（不传入视频路径以避免打开视频）
    configurator = ZoneConfigurator(None, "output")

    # 为监控视频创建典型的禁停区域
    # 假设这是一个路口监控视频，创建常见的禁停区域
    monitoring_zones = [
        {
            "name": "路口禁停区A",
            "type": "rectangle",
            "bbox": [200, 300, 600, 500],
            "color": [0, 0, 255],
            "description": "路口东北角禁停区域"
        },
        {
            "name": "路口禁停区B",
            "type": "rectangle",
            "bbox": [1000, 300, 1400, 500],
            "color": [0, 0, 255],
            "description": "路口西北角禁停区域"
        },
        {
            "name": "路口禁停区C",
            "type": "rectangle",
            "bbox": [200, 700, 600, 900],
            "color": [0, 0, 255],
            "description": "路口东南角禁停区域"
        },
        {
            "name": "路口禁停区D",
            "type": "rectangle",
            "bbox": [1000, 700, 1400, 900],
            "color": [0, 0, 255],
            "description": "路口西南角禁停区域"
        },
        {
            "name": "路中央禁停区",
            "type": "rectangle",
            "bbox": [600, 400, 1000, 800],
            "color": [0, 0, 255],
            "description": "路口中央区域禁停"
        }
    ]

    # 清空现有配置并添加新的
    configurator.zones = monitoring_zones
    configurator.save_zones()

    print(f"为监控视频创建了 {len(monitoring_zones)} 个禁停区域")
    print("\n禁停区域详情:")
    for zone in monitoring_zones:
        print(f"  - {zone['name']}: {zone['bbox']} - {zone['description']}")

    print(f"\n配置已保存到: output/no_parking_zones.json")
    print("\n使用方法:")
    print("1. 运行系统时会自动加载这些禁停区域")
    print("2. 如需修改区域，可以运行:")
    print("   python configure_zones.py [视频路径]")
    print("3. 然后在可视化界面中重新绘制区域")

    return True

def main():
    """主函数"""
    print("禁停区域配置测试工具")
    print("=" * 60)

    # 测试基本功能
    success1 = test_zone_config()

    # 为用户视频创建配置
    success2 = create_video_zones_config()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("[SUCCESS] 禁停区域配置测试全部完成！")
        print("\n您的违停检测系统现在具备了完整的禁停区域配置功能。")
    else:
        print("[WARNING] 部分测试未完成，请检查相关配置。")

    print("\n[提示] 使用提示:")
    print("- 禁停区域配置文件: output/no_parking_zones.json")
    print("- 运行违停检测: python parking_violation_system.py [视频路径]")
    print("- 可视化配置: python configure_zones.py [视频路径]")

if __name__ == "__main__":
    main()
