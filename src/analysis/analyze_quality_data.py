"""分析品質標記的截圖資料"""
from pathlib import Path
import argparse


def analyze_quality_data(screenshots_dir: str = "artifacts/screenshots"):
    """
    分析品質標記資料

    Args:
        screenshots_dir: 截圖目錄路徑
    """
    base_path = Path(screenshots_dir)

    if not base_path.exists():
        print(f"❌ 目錄不存在: {screenshots_dir}")
        return

    # 統計各品質等級的圖片數量
    quality_stats = {}
    total_images = 0

    for quality in range(10):  # 0-9
        quality_dir = base_path / f"quality_{quality}"
        if quality_dir.exists():
            images = list(quality_dir.glob("*.jpg"))
            count = len(images)
            quality_stats[quality] = count
            total_images += count
        else:
            quality_stats[quality] = 0

    # 統計未標記的圖片
    unlabeled_dir = base_path / "unlabeled"
    unlabeled_count = 0
    if unlabeled_dir.exists():
        unlabeled_count = len(list(unlabeled_dir.glob("*.jpg")))

    # 顯示報告
    print("\n" + "=" * 60)
    print("品質標記資料分析報告")
    print("=" * 60)

    if total_images == 0 and unlabeled_count == 0:
        print("❌ 沒有找到任何截圖資料")
        print(f"   請先執行: python -m src.app --source 0 --show")
        print(f"   並按 0-9 鍵進行品質標記截圖")
        return

    print(f"\n總截圖數: {total_images + unlabeled_count}")
    print(f"已標記: {total_images} 張")
    print(f"未標記: {unlabeled_count} 張")

    if total_images > 0:
        print("\n品質分布:")
        print("-" * 60)
        print(f"{'品質等級':<15} {'數量':<10} {'百分比':<15} {'視覺化'}")
        print("-" * 60)

        for quality in range(10):
            count = quality_stats[quality]
            if total_images > 0:
                percentage = (count / total_images) * 100
            else:
                percentage = 0

            # 視覺化長條圖
            bar_length = int(percentage / 2)  # 最長 50 字元
            bar = "█" * bar_length

            quality_label = f"Quality {quality}"
            if quality == 0:
                quality_label += " (最差)"
            elif quality == 9:
                quality_label += " (滿分)"

            print(f"{quality_label:<15} {count:<10} {percentage:>5.1f}%{'':>7} {bar}")

        print("-" * 60)

        # 計算平均品質
        total_quality_score = sum(q * count for q, count in quality_stats.items())
        avg_quality = total_quality_score / total_images if total_images > 0 else 0
        print(f"\n平均品質: {avg_quality:.2f} / 9.0")

        # 建議
        print("\n建議:")
        if quality_stats[0] + quality_stats[1] + quality_stats[2] > total_images * 0.3:
            print("  ⚠️  低品質樣本 (0-2) 較多，可能需要改善環境或設備")

        if quality_stats[7] + quality_stats[8] + quality_stats[9] > total_images * 0.7:
            print("  ✅ 高品質樣本 (7-9) 充足，適合訓練")

        # 檢查分布是否平衡
        max_count = max(quality_stats.values())
        min_count = min(c for c in quality_stats.values() if c > 0) if any(
            c > 0 for c in quality_stats.values()
        ) else 0

        if max_count > 0 and min_count > 0 and max_count / min_count > 5:
            print("  ⚠️  品質分布不平衡，建議收集更多不同品質的樣本")

    if unlabeled_count > 0:
        print(f"\n⚠️  有 {unlabeled_count} 張未標記的截圖")
        print(f"   位置: {unlabeled_dir}")

    # 顯示目錄結構
    print("\n資料目錄結構:")
    print(f"{screenshots_dir}/")
    for quality in range(10):
        count = quality_stats[quality]
        if count > 0:
            print(f"  ├── quality_{quality}/ ({count} 張)")
    if unlabeled_count > 0:
        print(f"  └── unlabeled/ ({unlabeled_count} 張)")

    print("\n" + "=" * 60)


def export_quality_list(screenshots_dir: str = "artifacts/screenshots",
                        output_file: str = "artifacts/quality_list.txt"):
    """
    匯出品質標記清單

    Args:
        screenshots_dir: 截圖目錄
        output_file: 輸出檔案
    """
    base_path = Path(screenshots_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for quality in range(10):
        quality_dir = base_path / f"quality_{quality}"
        if quality_dir.exists():
            for img_file in sorted(quality_dir.glob("*.jpg")):
                lines.append(f"{img_file},{quality}\n")

    if lines:
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"\n✅ 品質清單已匯出: {output_file}")
        print(f"   共 {len(lines)} 筆記錄")
    else:
        print("\n❌ 沒有資料可匯出")


def main():
    parser = argparse.ArgumentParser(description="分析品質標記的截圖資料")
    parser.add_argument(
        "--dir",
        default="artifacts/screenshots",
        help="截圖目錄（預設: artifacts/screenshots）"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="匯出品質清單至 CSV 檔案"
    )
    parser.add_argument(
        "--output",
        default="artifacts/quality_list.txt",
        help="匯出檔案路徑（預設: artifacts/quality_list.txt）"
    )
    args = parser.parse_args()

    analyze_quality_data(args.dir)

    if args.export:
        export_quality_list(args.dir, args.output)


if __name__ == "__main__":
    main()
