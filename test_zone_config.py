import sys
import json
from pathlib import Path
import cv2

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from chinese_text_draw import cv2_put_text_unicode

class ZoneConfigurator:
    """禁停区域配置器"""

    def __init__(self, media_path: str = None, output_dir: str = "output"):
        self.media_path = media_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.zones_file = self.output_dir / "no_parking_zones.json"

        # 加载现有配置
        self.zones = self.load_zones()

        # 鼠标交互状态
        self.drawing = False
        self.start_point = None
        self.current_zone = None

        # 媒体信息
        self.is_video = self._check_is_video(media_path)
        self.background_frame = None
        self.cap = None

    def _check_is_video(self, path: str) -> bool:
        """检查文件是否为视频"""
        if not path:
            return False
        path_obj = Path(path)
        if not path_obj.exists():
            return False

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        return path_obj.suffix.lower() in video_extensions

    def _extract_background_frame(self):
        """从视频中提取背景帧"""
        if not self.is_video or not self.media_path:
            return None

        try:
            self.cap = cv2.VideoCapture(self.media_path)
            if not self.cap.isOpened():
                print(f"无法打开视频: {self.media_path}")
                return None

            # 读取第一帧
            ret, frame = self.cap.read()
            if ret:
                print(f"已提取视频背景帧，尺寸: {frame.shape}")
                return frame
            else:
                print("无法读取视频帧")
                return None

        except Exception as e:
            print(f"提取视频帧失败: {e}")
            return None

    def load_zones(self):
        """加载禁停区域配置"""
        if self.zones_file.exists():
            try:
                with open(self.zones_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置失败: {e}")
        return []

    def save_zones(self):
        """保存禁停区域配置"""
        try:
            with open(self.zones_file, 'w', encoding='utf-8') as f:
                json.dump(self.zones, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到: {self.zones_file}")
        except Exception as e:
            print(f"保存配置失败: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制
            self.drawing = True
            self.start_point = (x, y)
            print(f"开始绘制禁停区域: ({x}, {y})")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 更新当前区域预览
                self.current_zone = [self.start_point[0], self.start_point[1], x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                # 完成绘制
                self.drawing = False
                end_point = (x, y)

                # 确保坐标正确
                x1 = min(self.start_point[0], end_point[0])
                y1 = min(self.start_point[1], end_point[1])
                x2 = max(self.start_point[0], end_point[0])
                y2 = max(self.start_point[1], end_point[1])

                # 添加新区域
                zone_name = f"禁停区_{len(self.zones) + 1}"
                new_zone = {
                    "name": zone_name,
                    "type": "rectangle",
                    "bbox": [x1, y1, x2, y2],
                    "color": [0, 0, 255],
                    "description": f"用户定义的禁停区域 {len(self.zones) + 1}"
                }

                self.zones.append(new_zone)
                print(f"已添加禁停区域: {zone_name} -> [{x1},{y1},{x2},{y2}]")
                self.current_zone = None

    def draw_zones(self, frame):
        """绘制禁停区域"""
        display_frame = frame.copy()

        # 绘制已保存的区域
        for zone in self.zones:
            if zone["type"] == "rectangle":
                x1, y1, x2, y2 = zone["bbox"]
                color = tuple(zone["color"])
                thickness = 3

                # 绘制边框
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                # 绘制半透明填充
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

                # 绘制标签
                label = zone['name']
                display_frame = cv2_put_text_unicode(
                    display_frame, label, (x1, y1 - 10),
                    font_size_px=18, color_bgr=tuple(int(c) for c in color), anchor="ls",
                )

                # 绘制禁止图标
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(display_frame, (center_x, center_y), 25, color, 3)
                cv2.line(display_frame, (center_x - 15, center_y - 15),
                        (center_x + 15, center_y + 15), color, 3)

        # 绘制当前正在绘制的区域
        if self.current_zone:
            x1, y1, x2, y2 = self.current_zone
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 绘制操作提示
        instructions = [
            "鼠标左键拖拽绘制禁停区域",
            "按 's' 保存配置",
            "按 'c' 清空所有区域",
            "按 'q' 退出"
        ]

        for i, instruction in enumerate(instructions):
            display_frame = cv2_put_text_unicode(
                display_frame, instruction, (10, 30 + i * 30),
                font_size_px=18, color_bgr=(255, 255, 255), anchor="ls",
            )

        return display_frame

    def run_configuration(self):
        """运行配置界面"""
        if not self.media_path:
            print("请提供背景媒体路径（图片或视频）")
            return

        # 获取背景帧
        if self.is_video:
            image = self._extract_background_frame()
            if image is None:
                print("无法从视频提取背景帧")
                return
        else:
            # 读取背景图片
            image = cv2.imread(self.media_path)
            if image is None:
                print(f"无法读取图片: {self.media_path}")
                return

        print("=" * 60)
        print("禁停区域配置工具")
        print("=" * 60)
        print("操作说明:")
        print("  鼠标左键拖拽: 绘制禁停区域")
        print("  按 's': 保存配置")
        print("  按 'c': 清空所有区域")
        print("  按 'q': 退出")
        print("=" * 60)

        # 创建窗口
        cv2.namedWindow('禁停区域配置', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('禁停区域配置', self.mouse_callback)

        while True:
            # 绘制界面
            display_frame = self.draw_zones(image)

            # 显示区域统计
            stats_text = f"当前禁停区域数量: {len(self.zones)}"
            display_frame = cv2_put_text_unicode(
                display_frame, stats_text, (10, display_frame.shape[0] - 40),
                font_size_px=22, color_bgr=(255, 255, 255), anchor="ls",
            )

            cv2.imshow('禁停区域配置', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("退出配置")
                break
            elif key == ord('s'):
                self.save_zones()
                print("配置已保存")
            elif key == ord('c'):
                self.zones = []
                print("已清空所有禁停区域")

        cv2.destroyAllWindows()

        # 释放视频资源
        if self.cap:
            self.cap.release()

def create_default_zones(image_width: int = 1920, image_height: int = 1080):
    """创建默认禁停区域配置"""
    zones = [
        {
            "name": "路边禁停区",
            "type": "rectangle",
            "bbox": [100, 200, int(image_width * 0.6), int(image_height * 0.8)],
            "color": [0, 0, 255],
            "description": "道路两侧禁停区域"
        },
        {
            "name": "交叉口禁停区",
            "type": "rectangle",
            "bbox": [int(image_width * 0.7), int(image_height * 0.3),
                    int(image_width * 0.9), int(image_height * 0.7)],
            "color": [0, 0, 255],
            "description": "交叉路口禁停区域"
        }
    ]
    return zones

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='禁停区域配置工具')
    parser.add_argument('media', help='背景媒体路径（图片或视频，用于可视化配置）')
    parser.add_argument('-o', '--output', default='output', help='输出目录')

    args = parser.parse_args()

    # 检查媒体文件是否存在
    media_path = Path(args.media)
    if not media_path.exists():
        print(f"错误: 媒体文件不存在 {args.media}")
        return

    print(f"媒体文件: {args.media}")
    print(f"文件类型: {'视频' if media_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'] else '图片'}")

    # 创建配置器
    configurator = ZoneConfigurator(args.media, args.output)

    # 运行配置界面
    configurator.run_configuration()

    print("\n配置完成！")
    print(f"禁停区域配置已保存到: {configurator.zones_file}")

if __name__ == "__main__":
    main()
