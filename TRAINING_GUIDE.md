# 訓練完成後使用指南

## 📋 訓練摘要

您已成功開始訓練品質管理模型！

### 資料集統計
- **總影像數**: 113 張
- **訓練集**: 79 張 (70%)
- **驗證集**: 22 張 (20%)
- **測試集**: 12 張 (10%)

### 類別分布
- **good (良品)**: 66 張 (58.4%)
- **defect (缺陷)**: 47 張 (41.6%)

### 訓練參數
- **模型**: YOLOv8n
- **訓練輪數**: 50 epochs
- **影像大小**: 640
- **批次大小**: 16

---

## 🔍 監控訓練過程

### 查看訓練輸出

```bash
# 查看訓練進度（如果在背景執行）
tail -f artifacts/runs/qc/quality_control_v1/train.log

# 或直接查看終端輸出
```

### 訓練完成標誌

當您看到以下訊息時，訓練已完成：

```
Results saved to artifacts/runs/qc/quality_control_v1
```

---

## 📊 評估訓練結果

### 1. 查看訓練指標

訓練完成後，檢查以下檔案：

```bash
# 訓練結果目錄
cd artifacts/runs/qc/quality_control_v1

# 主要檔案：
# - best.pt           # 最佳模型
# - last.pt           # 最後一輪模型
# - results.csv       # 訓練指標
# - results.png       # 訓練曲線圖
# - confusion_matrix.png  # 混淆矩陣
```

### 2. 關鍵指標解讀

打開 `results.png` 查看：

- **mAP50**: 應該 > 0.7（良好）或 > 0.8（優秀）
- **mAP50-95**: 應該 > 0.5（良好）
- **Precision**: 精確率，越高越好（減少誤報）
- **Recall**: 召回率，越高越好（減少漏報）

### 3. 混淆矩陣分析

查看 `confusion_matrix.png`：

```
            Predicted
            good  defect
Actual good   TP    FP
      defect  FN    TN

TP (True Positive): 正確識別為良品
TN (True Negative): 正確識別為缺陷
FP (False Positive): 誤報（實際是缺陷但判為良品）⚠️
FN (False Negative): 漏報（實際是良品但判為缺陷）
```

**重要提示**：
- FP（誤報）較嚴重：讓缺陷品流入市場
- FN（漏報）影響較小：良品被誤判為缺陷

---

## 🚀 使用訓練好的模型

### 1. 即時檢測

```bash
# 使用訓練好的模型進行即時檢測
python -m src.app \
    --source 0 \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --conf 0.5 \
    --show
```

### 2. 評估模型

```bash
# 在測試集上評估模型
python scripts/evaluate_qc.py \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --data data/quality_control/dataset.yaml \
    --split test
```

### 3. 視覺化預測結果

```bash
# 生成預測結果視覺化
python scripts/visualize_predictions.py \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --images data/quality_control/images/test \
    --output artifacts/predictions \
    --conf 0.5
```

### 4. 部署到生產線

```bash
# 啟動品質管理系統（含統計功能）
python scripts/deploy_qc.py \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --source 0 \
    --conf 0.5 \
    --save \
    --output artifacts/qc_results
```

---

## 🎯 模型性能標準

根據您的應用場景，設定合適的閾值：

### 標準品檢（推薦）

```bash
python -m src.app \
    --source 0 \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --conf 0.5 \
    --show

# 期望性能：
# - mAP50 > 0.8
# - Precision > 0.85
# - Recall > 0.85
```

### 嚴格品檢（降低誤報）

```bash
python -m src.app \
    --source 0 \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --conf 0.7 \
    --show

# 高信心度閾值（0.7）
# - 減少誤報（FP）
# - 可能增加漏報（FN）
```

### 寬鬆檢測（減少漏報）

```bash
python -m src.app \
    --source 0 \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --conf 0.3 \
    --show

# 低信心度閾值（0.3）
# - 減少漏報（FN）
# - 可能增加誤報（FP）
```

---

## 🔧 模型優化建議

### 如果 mAP < 0.7（性能不佳）

#### 1. 收集更多資料

```bash
# 繼續收集資料
python -m src.app --source 0 --show

# 目標：每個類別至少 200 張影像
# 確保類別平衡（good:defect = 1:1）
```

#### 2. 檢查資料品質

```bash
# 檢查標記錯誤
python scripts/analyze_errors.py \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --images data/quality_control/images/test \
    --labels data/quality_control/labels/test
```

#### 3. 調整訓練參數

```bash
# 使用更大的模型
python scripts/train_qc.py \
    --data data/quality_control/dataset.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --imgsz 640 \
    --name quality_control_v2

# 或增加訓練輪數
python scripts/train_qc.py \
    --data data/quality_control/dataset.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --name quality_control_v2
```

### 如果過擬合（訓練集好但驗證集差）

```bash
# 增加資料增強
# 修改 scripts/train_qc.py 中的增強參數

# 或收集更多多樣化的資料
# - 不同光線條件
# - 不同角度
# - 不同時間點
```

---

## 📈 持續改進流程

### 1. 收集困難樣本

```bash
# 找出模型不確定的樣本
python scripts/active_learning.py \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --images data/production_images \
    --output data/uncertain_samples \
    --threshold 0.4
```

### 2. 重新標記

```bash
# 標記不確定的樣本
python -m src.app --source 0 --show
# 或使用 LabelImg 重新標記
```

### 3. 合併資料集

```bash
# 將新資料加入訓練集
cp data/uncertain_samples/*.jpg artifacts/screenshots/quality_X/

# 重新準備資料集
python scripts/prepare_quality_dataset.py --mode binary

# 繼續訓練（使用之前的模型）
python scripts/train_qc.py \
    --data data/quality_control/dataset.yaml \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --epochs 50 \
    --name quality_control_v2
```

---

## 📊 生成報告

### 訓練報告

```markdown
# 品質管理模型訓練報告

## 資料集
- 訓練集: 79 張
- 驗證集: 22 張
- 測試集: 12 張
- 類別: good (58.4%), defect (41.6%)

## 訓練配置
- 模型: YOLOv8n
- Epochs: 50
- Image Size: 640
- Batch Size: 16

## 性能指標
- mAP50: [從 results.csv 查看]
- mAP50-95: [從 results.csv 查看]
- Precision: [從 results.csv 查看]
- Recall: [從 results.csv 查看]

## 結論
[分析模型是否達到預期性能]

## 下一步
[列出改進建議]
```

---

## ❓ 常見問題

### Q1: 訓練需要多久？

**答**：
- 小資料集（<200 張）+ YOLOv8n: 5-10 分鐘（GPU）或 30-60 分鐘（CPU）
- 中型資料集（200-1000 張）+ YOLOv8s: 15-30 分鐘（GPU）或 1-3 小時（CPU）

### Q2: 訓練時記憶體不足？

**解決方法**：
```bash
# 減少批次大小
python scripts/train_qc.py \
    --data data/quality_control/dataset.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 8 \
    --name quality_control_v1
```

### Q3: 如何知道訓練是否成功？

**檢查項目**：
1. 訓練是否完成（沒有錯誤訊息）
2. mAP50 > 0.7
3. 混淆矩陣顯示大部分預測正確
4. 在測試集上手動檢查幾張影像

### Q4: 模型可以用在不同的攝影機嗎？

**建議**：
- 如果光線、角度相似：可以直接使用
- 如果環境差異大：建議收集新環境的資料重新訓練
- 可以使用遷移學習（用現有模型繼續訓練）

---

## 📚 相關資源

- [README.md](README.md) - 專案總覽
- [HOWTO.md](HOWTO.md) - 完整訓練流程
- [QUICK_START_QC.md](QUICK_START_QC.md) - 快速上手指南
- [Ultralytics YOLO 文件](https://docs.ultralytics.com/)

---

## ✅ 下一步檢查清單

訓練完成後：

- [ ] 查看 `results.png` - 訓練曲線
- [ ] 查看 `confusion_matrix.png` - 混淆矩陣
- [ ] 檢查 mAP50 是否 > 0.7
- [ ] 在測試集上評估模型
- [ ] 視覺化預測結果
- [ ] 在真實環境中測試
- [ ] 調整信心度閾值
- [ ] 如需要，收集更多資料重新訓練

**開始使用您的模型**：

```bash
python -m src.app \
    --source 0 \
    --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
    --conf 0.5 \
    --show
```

恭喜！您已經訓練出自己的品質管理模型！🎉
