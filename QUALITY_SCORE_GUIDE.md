# 品質評分系統使用指南

顯示 0-9 品質分數的完整解決方案。

---

## 🎯 系統概述

本系統可以：
- ✅ 自動框出產品
- ✅ 顯示 0-9 的品質分數
- ✅ 即時統計各品質等級分布
- ✅ 顏色編碼（紅色=差，綠色=好）

---

## 📊 訓練狀態

### 當前訓練配置

```
資料集: data/quality_scores/
類別數: 10 (quality_0 ~ quality_9)
模型: YOLOv8n
訓練輪數: 50 epochs
影像大小: 640
```

### 資料分布

```
quality_0: 6 張 (5.6%)
quality_1: 7 張 (6.5%)
quality_2: 9 張 (8.4%)
quality_3: 6 張 (5.6%)
quality_5: 11 張 (10.3%)
quality_6: 28 張 (26.2%)
quality_8: 17 張 (15.9%)
quality_9: 23 張 (21.5%)

總計: 107 張
訓練集: 74 張 | 驗證集: 21 張 | 測試集: 12 張
```

**注意**：品質 4 和 7 沒有樣本，建議補充這些等級的資料。

---

## 🚀 訓練完成後使用

### 1. 等待訓練完成

訓練時間約 5-15 分鐘（CPU）或 2-5 分鐘（GPU）。

訓練完成後，模型位於：
```
artifacts/runs/qc/quality_scores_v1/weights/best.pt
```

### 2. 啟動品質檢測系統

```bash
# 基本使用（攝影機 0）
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.25

# 使用 RTSP 串流
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source rtsp://192.168.1.100:554/stream \
    --conf 0.3

# 處理影片檔案
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source video.mp4 \
    --conf 0.3

# 使用 GPU 加速
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.25 \
    --device cuda
```

### 3. 操作說明

**鍵盤控制**：
- `q` 或 `ESC`：退出程式
- `r`：重置統計資訊

**顯示說明**：
- 每個產品會被框起來
- 框的顏色表示品質：
  - 🟢 綠色：高品質（8-9）
  - 🟡 黃色：中等品質（5-7）
  - 🔴 紅色：低品質（0-4）
- 框上顯示：
  - `Quality: X/9` - 品質分數
  - `Conf: 0.XX` - 檢測信心度
- 右上角顯示統計資訊

---

## 🎨 視覺效果範例

```
┌─────────────────────────────────────────┐
│                                         │
│   ┏━━━━━━━━━━━━┓  [綠色框]             │
│   ┃ Quality: 9/9 ┃                      │
│   ┃ Conf: 0.87   ┃                      │
│   ┃  [產品影像]  ┃                      │
│   ┗━━━━━━━━━━━━┛                       │
│                                         │
│      ┏━━━━━━━━━━┓  [黃色框]            │
│      ┃ Quality: 5/9 ┃                   │
│      ┃ Conf: 0.72   ┃                   │
│      ┃ [產品影像]   ┃                   │
│      ┗━━━━━━━━━━┛                      │
│                                         │
│  ┏━━━━━━━━━━━┓  [紅色框]              │
│  ┃ Quality: 2/9  ┃                     │
│  ┃ Conf: 0.65    ┃  [統計面板]         │
│  ┃ [產品影像]    ┃  Total: 243         │
│  ┗━━━━━━━━━━━┛  Q0: 12 (4.9%)        │
│                      Q2: 18 (7.4%)      │
│                      Q5: 45 (18.5%)     │
│                      Q6: 89 (36.6%)     │
│                      Q8: 42 (17.3%)     │
│                      Q9: 37 (15.2%)     │
│                      Avg: 6.45          │
└─────────────────────────────────────────┘
```

---

## 📈 性能調整

### 調整信心度閾值

```bash
# 低閾值（檢測更多，可能有誤報）
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.1

# 標準閾值（推薦）
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.25

# 高閾值（只顯示高信心度檢測）
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.5
```

---

## 🔧 改善模型性能

### 問題 1：某些品質等級檢測不準

**原因**：該品質等級的訓練樣本太少

**解決方案**：
```bash
# 1. 檢查當前資料分布
python scripts/analyze_quality_data.py

# 2. 收集缺少的品質等級資料
python -m src.app --source 0 --show
# 按對應的數字鍵（0-9）收集缺少的等級

# 3. 重新準備資料集
python scripts/prepare_quality_dataset.py \
    --mode quality \
    --output data/quality_scores

# 4. 重新訓練
python scripts/train_qc.py \
    --data data/quality_scores/dataset.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --name quality_scores_v2
```

### 問題 2：檢測框不準確

**解決方案**：
```bash
# 使用更大的模型
python scripts/train_qc.py \
    --data data/quality_scores/dataset.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --name quality_scores_v2

# 或增加影像解析度
python scripts/train_qc.py \
    --data data/quality_scores/dataset.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --imgsz 1280 \
    --name quality_scores_v2
```

### 問題 3：品質等級分類錯誤

**解決方案 A**：收集更多樣本
```bash
# 每個品質等級目標：至少 50 張
# 當前不足的等級：
# - quality_4: 0 張 ← 需要收集
# - quality_7: 0 張 ← 需要收集
# - quality_0: 6 張 ← 建議增加到 50 張
# - quality_1: 7 張 ← 建議增加到 50 張
```

**解決方案 B**：簡化分類
```bash
# 如果 10 級太多，可以改用 5 級或 3 級

# 修改 prepare_quality_dataset.py，加入自訂映射
# 例如：5 級分類
# 0-1 → quality_0 (很差)
# 2-3 → quality_1 (差)
# 4-5 → quality_2 (中等)
# 6-7 → quality_3 (好)
# 8-9 → quality_4 (很好)
```

---

## 💡 實際應用場景

### 場景 1：生產線品質監控

```bash
# 即時監控，自動記錄
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.3

# 設定品質閾值告警
# 如果檢測到 quality < 4，觸發警報（需自行開發）
```

### 場景 2：抽檢驗證

```bash
# 每小時抽檢 10 個產品
# 記錄品質分數
# 生成報告

# 執行抽檢
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.4

# 按 'r' 鍵重置計數器
# 檢查 10 個產品後查看統計
```

### 場景 3：品質趨勢分析

```bash
# 記錄每天的平均品質
# 追蹤品質變化趨勢

# 早班（8:00-12:00）
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0

# 晚班（13:00-17:00）
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0

# 比較早晚班的平均品質差異
```

---

## 📊 評估模型

### 在測試集上評估

```bash
# 評估模型性能
python scripts/evaluate_qc.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --data data/quality_scores/dataset.yaml \
    --split test
```

### 視覺化預測結果

```bash
# 生成預測結果視覺化
python scripts/visualize_predictions.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --images data/quality_scores/images/test \
    --output artifacts/quality_predictions \
    --conf 0.25
```

### 查看混淆矩陣

訓練完成後，查看以下檔案：
```
artifacts/runs/qc/quality_scores_v1/confusion_matrix.png
```

理想的混淆矩陣應該：
- 對角線值高（正確預測）
- 非對角線值低（錯誤預測）

---

## 🎓 最佳實踐

### 1. 資料收集建議

```bash
# 每個品質等級至少收集 50 張
# 總計至少 500 張（10 個等級 × 50 張）

# 多樣性：
# - 不同光線條件
# - 不同角度
# - 不同時間點
# - 不同產品批次
```

### 2. 標記一致性

```markdown
建立品質評分標準：

Quality 0: 完全損壞、無法使用
Quality 1-2: 嚴重缺陷、不合格
Quality 3-4: 明顯缺陷、需返工
Quality 5-6: 輕微瑕疵、可接受
Quality 7-8: 良好、少量瑕疵
Quality 9: 完美、無瑕疵
```

### 3. 定期重新訓練

```bash
# 每月收集新資料
# 加入訓練集
# 重新訓練模型

# 版本管理
python scripts/train_qc.py \
    --data data/quality_scores/dataset.yaml \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --epochs 50 \
    --name quality_scores_v2
```

---

## ❓ 常見問題

### Q1: 為什麼有些品質等級檢測不到？

**答**：可能是該品質等級的訓練樣本太少或沒有。

**解決**：
- 檢查 `python scripts/analyze_quality_data.py`
- 補充缺少的品質等級樣本
- 重新訓練

### Q2: 檢測框閃爍或不穩定？

**答**：可能是信心度閾值太低。

**解決**：
```bash
# 提高信心度閾值
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.4  # 從 0.25 提高到 0.4
```

### Q3: 顏色顯示不正確？

**答**：檢查品質分數映射是否正確。

**確認**：
- Quality 0-4: 紅色系
- Quality 5-6: 黃色系
- Quality 7-9: 綠色系

### Q4: 如何匯出檢測結果？

**答**：可以修改 `quality_inspector.py`，將檢測結果寫入 CSV 或資料庫。

範例：
```python
# 在 quality_inspector.py 中加入
import csv

with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'quality', 'confidence'])
    # 每次檢測後寫入
```

---

## 📚 相關文件

- [README.md](README.md) - 專案總覽
- [HOWTO.md](HOWTO.md) - 完整訓練流程
- [QUICK_START_QC.md](QUICK_START_QC.md) - 快速上手
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 訓練完成後指南

---

## ✅ 快速檢查清單

開始使用前：

- [ ] 訓練已完成（檢查 `artifacts/runs/qc/quality_scores_v1/weights/best.pt` 是否存在）
- [ ] 已測試攝影機（`python -m src.app --source 0 --show`）
- [ ] 已確認模型性能（mAP50 > 0.6）
- [ ] 已調整信心度閾值
- [ ] 已準備好生產環境

開始檢測：

```bash
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_scores_v1/weights/best.pt \
    --source 0 \
    --conf 0.25
```

享受自動品質評分！🎉
