"""將品質標記截圖轉換為 YOLO 訓練格式"""
from pathlib import Path
import shutil
import random
import argparse
import yaml


def prepare_quality_dataset(
    screenshots_dir: str = "artifacts/screenshots",
    output_dir: str = "data/quality_control",
    split_ratio: tuple = (0.7, 0.2, 0.1),
    classification_mode: str = "binary",
    seed: int = 42
):
    """
    準備品質管理訓練資料集

    Args:
        screenshots_dir: 截圖來源目錄
        output_dir: 輸出資料集目錄
        split_ratio: (train, val, test) 比例
        classification_mode: 分類模式
            - "binary": 二分類（good, defect）
            - "triclass": 三分類（good, minor_defect, major_defect）
            - "quality": 10 分類（quality_0 ~ quality_9）
        seed: 隨機種子
    """
    random.seed(seed)

    screenshots_path = Path(screenshots_dir)
    output_path = Path(output_dir)

    # 檢查來源目錄
    if not screenshots_path.exists():
        print(f"[!] Error: Screenshot directory not found: {screenshots_dir}")
        return False

    # 定義分類映射
    if classification_mode == "binary":
        # 二分類：0-5 為缺陷，6-9 為良品
        class_mapping = {
            0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1,  # defect
            6: 0, 7: 0, 8: 0, 9: 0                 # good
        }
        class_names = {0: "good", 1: "defect"}
        print("\n[*] Classification Mode: Binary")
        print("   good (0): Quality 6-9")
        print("   defect (1): Quality 0-5")

    elif classification_mode == "triclass":
        # 三分類
        class_mapping = {
            0: 2, 1: 2, 2: 2, 3: 2,     # major_defect
            4: 1, 5: 1, 6: 1,           # minor_defect
            7: 0, 8: 0, 9: 0            # good
        }
        class_names = {0: "good", 1: "minor_defect", 2: "major_defect"}
        print("\n[*] Classification Mode: Tri-class")
        print("   good (0): Quality 7-9")
        print("   minor_defect (1): Quality 4-6")
        print("   major_defect (2): Quality 0-3")

    elif classification_mode == "quality":
        # 10 分類：保持原品質等級
        class_mapping = {i: i for i in range(10)}
        class_names = {i: f"quality_{i}" for i in range(10)}
        print("\n[*] Classification Mode: 10-level Quality")
        print("   quality_0 (0) ~ quality_9 (9)")

    else:
        print(f"[!] Error: Unknown classification mode {classification_mode}")
        return False

    # 收集所有影像
    all_images = []
    quality_stats = {}

    print("\n[*] Scanning screenshot directory...")
    for quality_level in range(10):
        quality_dir = screenshots_path / f"quality_{quality_level}"
        if not quality_dir.exists():
            continue

        images = list(quality_dir.glob("*.jpg"))
        if images:
            mapped_class = class_mapping[quality_level]
            quality_stats[quality_level] = len(images)

            for img in images:
                all_images.append({
                    "path": img,
                    "original_quality": quality_level,
                    "mapped_class": mapped_class,
                })

            print(f"   quality_{quality_level} → {class_names[mapped_class]}: {len(images)} 張")

    if not all_images:
        print("\n[!] 錯誤：沒有找到任何標記的影像")
        print("   請先執行: python -m src.app --source 0 --show")
        print("   並按 0-9 鍵標記品質")
        return False

    total_images = len(all_images)
    print(f"\n[+] 共找到 {total_images} 張標記影像")

    # 檢查類別平衡
    class_counts = {}
    for img in all_images:
        cls = img["mapped_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print("\n[*] 轉換後的類別分布:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_images) * 100
        print(f"   {class_names[cls]} ({cls}): {count} 張 ({percentage:.1f}%)")

    # 檢查不平衡
    if len(class_counts) > 1:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        if max_count / min_count > 3:
            print(f"\n[!]  警告：類別不平衡（比例 {max_count/min_count:.1f}:1）")
            print("   建議收集更多少數類別的樣本")

    # 隨機打亂
    random.shuffle(all_images)

    # 分割資料集
    train_end = int(total_images * split_ratio[0])
    val_end = train_end + int(total_images * split_ratio[1])

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    print(f"\n[*] 資料集分割:")
    print(f"   訓練集: {len(splits['train'])} 張 ({split_ratio[0]*100:.0f}%)")
    print(f"   驗證集: {len(splits['val'])} 張 ({split_ratio[1]*100:.0f}%)")
    print(f"   測試集: {len(splits['test'])} 張 ({split_ratio[2]*100:.0f}%)")

    # 建立目錄結構
    print(f"\n[*] 建立資料集目錄: {output_dir}")
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 複製影像並創建標記檔案
    print("\n[*] 複製影像並創建標記檔案...")
    for split_name, images in splits.items():
        for img_data in images:
            src_img = img_data["path"]
            dst_img = output_path / "images" / split_name / src_img.name

            # 複製影像
            shutil.copy2(src_img, dst_img)

            # 創建 YOLO 標記檔案（分類任務：整張影像為一個類別）
            # 對於分類任務，我們創建包含整張影像的邊界框
            label_file = output_path / "labels" / split_name / f"{src_img.stem}.txt"
            with open(label_file, "w") as f:
                # YOLO 格式：class_id x_center y_center width height
                # 整張影像：中心點 (0.5, 0.5)，寬高 (1.0, 1.0)
                f.write(f"{img_data['mapped_class']} 0.5 0.5 1.0 1.0\n")

    # 創建資料集 YAML 配置檔
    yaml_path = output_path / "dataset.yaml"
    yaml_data = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[+] 資料集準備完成！")
    print(f"\n[*] 資料集配置檔: {yaml_path}")
    print(f"\n目錄結構:")
    print(f"{output_dir}/")
    print(f"├── dataset.yaml        # 資料集配置")
    print(f"├── images/")
    print(f"│   ├── train/         # {len(splits['train'])} 張")
    print(f"│   ├── val/           # {len(splits['val'])} 張")
    print(f"│   └── test/          # {len(splits['test'])} 張")
    print(f"└── labels/")
    print(f"    ├── train/         # {len(splits['train'])} 個標記檔")
    print(f"    ├── val/           # {len(splits['val'])} 個標記檔")
    print(f"    └── test/          # {len(splits['test'])} 個標記檔")

    print(f"\n[*] 下一步：開始訓練")
    print(f"\n執行以下指令開始訓練：")
    print(f"\n  python scripts/train_qc.py \\")
    print(f"      --data {yaml_path} \\")
    print(f"      --model yolov8n.pt \\")
    print(f"      --epochs 50 \\")
    print(f"      --imgsz 640 \\")
    print(f"      --name quality_control_v1")

    return True


def main():
    parser = argparse.ArgumentParser(description="準備品質管理訓練資料集")
    parser.add_argument(
        "--screenshots",
        default="artifacts/screenshots",
        help="截圖來源目錄（預設: artifacts/screenshots）"
    )
    parser.add_argument(
        "--output",
        default="data/quality_control",
        help="輸出資料集目錄（預設: data/quality_control）"
    )
    parser.add_argument(
        "--mode",
        default="binary",
        choices=["binary", "triclass", "quality"],
        help="分類模式（預設: binary）"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="訓練集比例（預設: 0.7）"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.2,
        help="驗證集比例（預設: 0.2）"
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="測試集比例（預設: 0.1）"
    )
    args = parser.parse_args()

    print("="*60)
    print("品質管理資料集準備工具")
    print("="*60)

    success = prepare_quality_dataset(
        screenshots_dir=args.screenshots,
        output_dir=args.output,
        split_ratio=(args.train, args.val, args.test),
        classification_mode=args.mode,
    )

    if success:
        print("\n[+] 成功！")
    else:
        print("\n[!] 失敗！")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
