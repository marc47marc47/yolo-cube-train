"""品質檢測系統 - 顯示 0-9 品質分數"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


class QualityInspector:
    """品質檢測系統"""

    def __init__(
        self,
        model_path: str,
        source: str = "0",
        conf: float = 0.25,
        device: str = None
    ):
        """
        初始化品質檢測器

        Args:
            model_path: 模型路徑
            source: 影像來源（攝影機索引或影片路徑）
            conf: 信心度閾值
            device: 運算設備（cuda/cpu）
        """
        self.model = YOLO(model_path)
        self.source = int(source) if source.isdigit() else source
        self.conf = conf
        self.device = device

        # 品質等級顏色映射（綠 -> 黃 -> 紅）
        self.quality_colors = {
            0: (0, 0, 255),      # 紅色 - 最差
            1: (0, 50, 255),     # 橙紅色
            2: (0, 100, 255),    # 橙色
            3: (0, 150, 255),    # 淺橙色
            4: (0, 200, 255),    # 黃橙色
            5: (0, 255, 255),    # 黃色
            6: (0, 255, 200),    # 黃綠色
            7: (0, 255, 100),    # 淺綠色
            8: (0, 255, 50),     # 綠色
            9: (0, 255, 0),      # 亮綠色 - 滿分
        }

        # 統計
        self.stats = {i: 0 for i in range(10)}
        self.total_detections = 0

    def get_quality_color(self, quality: int) -> tuple:
        """根據品質等級返回顏色"""
        return self.quality_colors.get(quality, (128, 128, 128))

    def draw_quality_box(
        self,
        frame: np.ndarray,
        box: tuple,
        quality: int,
        confidence: float
    ) -> np.ndarray:
        """
        繪製品質檢測框

        Args:
            frame: 影像幀
            box: 邊界框 (x1, y1, x2, y2)
            quality: 品質等級 0-9
            confidence: 信心度

        Returns:
            繪製後的影像
        """
        x1, y1, x2, y2 = map(int, box)
        color = self.get_quality_color(quality)

        # 繪製邊界框（加粗）
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 準備標籤文字
        label = f"Quality: {quality}/9"
        conf_text = f"Conf: {confidence:.2f}"

        # 文字背景大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale * 0.7, font_thickness - 1)

        # 繪製標籤背景
        bg_height = label_h + conf_h + 15
        bg_width = max(label_w, conf_w) + 10

        # 確保標籤在影像內
        if y1 - bg_height < 0:
            # 標籤放在框下方
            label_y1 = y2
            label_y2 = y2 + bg_height
        else:
            # 標籤放在框上方
            label_y1 = y1 - bg_height
            label_y2 = y1

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + bg_width, label_y2),
            color,
            -1  # 填充
        )

        # 繪製品質分數文字（白色）
        text_y = label_y1 + label_h + 5 if label_y1 < y1 else label_y1 + label_h + 5
        cv2.putText(
            frame,
            label,
            (x1 + 5, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

        # 繪製信心度文字（白色，較小）
        conf_y = text_y + conf_h + 5
        cv2.putText(
            frame,
            conf_text,
            (x1 + 5, conf_y),
            font,
            font_scale * 0.7,
            (255, 255, 255),
            font_thickness - 1
        )

        return frame

    def draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """繪製統計資訊"""
        height, width = frame.shape[:2]

        # 統計面板背景
        panel_height = 200
        panel_width = 250
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (width - panel_width - 10, 10),
            (width - 10, panel_height + 10),
            (0, 0, 0),
            -1
        )
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 標題
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "Quality Statistics",
            (width - panel_width, 30),
            font,
            0.6,
            (255, 255, 255),
            2
        )

        # 總檢測數
        cv2.putText(
            frame,
            f"Total: {self.total_detections}",
            (width - panel_width, 55),
            font,
            0.5,
            (255, 255, 255),
            1
        )

        # 各品質等級統計
        y_offset = 80
        for quality in range(10):
            count = self.stats[quality]
            if count > 0:
                percentage = (count / self.total_detections * 100) if self.total_detections > 0 else 0
                color = self.get_quality_color(quality)

                # 品質等級文字
                text = f"Q{quality}: {count} ({percentage:.1f}%)"
                cv2.putText(
                    frame,
                    text,
                    (width - panel_width, y_offset),
                    font,
                    0.45,
                    color,
                    1
                )
                y_offset += 20

        return frame

    def run(self):
        """執行品質檢測"""
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")

        print("="*60)
        print("Quality Inspector - 品質檢測系統")
        print("="*60)
        print("Controls:")
        print("  'q' or ESC - Quit")
        print("  'r' - Reset statistics")
        print("="*60)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 推論
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                device=self.device,
                verbose=False
            )

            # 處理檢測結果
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 提取資訊
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        # 品質等級就是類別 ID（0-9）
                        quality = cls

                        # 更新統計
                        self.stats[quality] += 1
                        self.total_detections += 1

                        # 繪製檢測框
                        frame = self.draw_quality_box(frame, xyxy, quality, conf)

            # 繪製統計資訊
            frame = self.draw_statistics(frame)

            # 顯示
            cv2.imshow("Quality Inspector", frame)

            # 鍵盤控制
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('r'):  # Reset statistics
                self.stats = {i: 0 for i in range(10)}
                self.total_detections = 0
                print("[Reset] Statistics cleared")

        cap.release()
        cv2.destroyAllWindows()

        # 顯示最終統計
        self.print_final_stats()

    def print_final_stats(self):
        """顯示最終統計"""
        print("\n" + "="*60)
        print("Final Statistics")
        print("="*60)
        print(f"Total detections: {self.total_detections}")

        if self.total_detections > 0:
            print("\nQuality distribution:")
            for quality in range(10):
                count = self.stats[quality]
                percentage = (count / self.total_detections * 100)
                if count > 0:
                    bar = "█" * int(percentage / 2)
                    print(f"  Quality {quality}: {count:4d} ({percentage:5.1f}%) {bar}")

            # 計算平均品質
            total_quality = sum(q * count for q, count in self.stats.items())
            avg_quality = total_quality / self.total_detections
            print(f"\nAverage quality: {avg_quality:.2f} / 9.0")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="品質檢測系統 - 顯示 0-9 品質分數")
    parser.add_argument(
        "--model",
        required=True,
        help="訓練好的模型路徑（quality 模式）"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="影像來源（攝影機索引或影片路徑）"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="信心度閾值（預設: 0.25）"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="運算設備（cuda/cpu）"
    )
    args = parser.parse_args()

    inspector = QualityInspector(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        device=args.device
    )

    try:
        inspector.run()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Exiting...")
    except Exception as e:
        print(f"\n[Error] {e}")
        raise


if __name__ == "__main__":
    main()
