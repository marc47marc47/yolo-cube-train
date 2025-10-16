# YOLO 即時影像辨識系統

使用 YOLOv8 透過攝影機進行物件辨識的即時系統，支援 USB 攝影機、RTSP 串流與影片檔案輸入。

## 專案特色

- 🎯 **即時推論**：支援攝影機、RTSP 串流、影片檔案
- 🏭 **品質管理系統**：按 0-9 鍵快速標記品質，自動訓練品質評分模型
- 📊 **品質評分顯示**：框出產品並顯示 0-9 品質分數，彩色編碼（紅→黃→綠）
- 🧪 **完整測試**：單元測試覆蓋率 100%
- 🔧 **模組化設計**：清晰的架構，易於擴展
- 📈 **訓練支援**：完整的資料集準備與模型訓練流程
- 🚀 **快速開始**：5 分鐘內完成環境建置與測試

## 快速開始

### 1. 環境建置

```bash
# 安裝 uv（如果尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 建立虛擬環境並安裝依賴
bash ./scripts/install_with_uv.sh

# 啟動虛擬環境
source yolo/bin/activate          # Linux/macOS
source yolo/Scripts/activate      # Windows Git Bash
```

### 2. 即時辨識

#### 基本使用

```bash
# 使用預設攝影機（會自動下載 YOLOv8n 模型）
python -m src.app --source 0 --model yolov8n.pt --show

# 使用 RTSP 串流
python -m src.app --source rtsp://192.168.1.100:554/stream --model yolov8n.pt --show

# 使用影片檔案
python -m src.app --source video.mp4 --model yolov8n.pt --show
```

#### 進階參數設定

```bash
# 調整信心度閾值（只顯示信心度 > 0.5 的偵測結果）
python -m src.app --source 0 --model yolov8n.pt --conf 0.5 --show

# 使用 GPU 加速（需要 CUDA）
python -m src.app --source 0 --model yolov8n.pt --device cuda --show

# 使用 CPU 運算
python -m src.app --source 0 --model yolov8n.pt --device cpu --show

# 調整 IoU 閾值（非最大值抑制）
python -m src.app --source 0 --model yolov8n.pt --iou 0.5 --show

# 使用自訂類別名稱檔案
python -m src.app --source 0 --model yolov8n.pt --names data/custom.yaml --show

# 不顯示視窗（背景執行）
python -m src.app --source 0 --model yolov8n.pt
```

#### 多攝影機設定

```bash
# 攝影機 0（通常是內建鏡頭）
python -m src.app --source 0 --model yolov8n.pt --show

# 攝影機 1（通常是外接 USB 鏡頭）
python -m src.app --source 1 --model yolov8n.pt --show

# 攝影機 2
python -m src.app --source 2 --model yolov8n.pt --show
```

#### RTSP 串流範例

```bash
# IP 攝影機
python -m src.app --source rtsp://admin:password@192.168.1.100:554/stream --model yolov8n.pt --show

# ONVIF 相容攝影機
python -m src.app --source rtsp://192.168.1.100:554/onvif1 --model yolov8n.pt --show

# 海康威視
python -m src.app --source rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101 --model yolov8n.pt --show

# 大華攝影機
python -m src.app --source rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0 --model yolov8n.pt --show
```

#### 模型選擇指南

| 模型 | 參數量 | 速度 | 精度 | 適用場景 |
|------|--------|------|------|----------|
| yolov8n.pt | 3.2M | 最快 | 基礎 | 即時性要求高、硬體資源有限 |
| yolov8s.pt | 11.2M | 快 | 良好 | 平衡速度與精度 |
| yolov8m.pt | 25.9M | 中等 | 優秀 | 精度優先，可接受延遲 |
| yolov8l.pt | 43.7M | 慢 | 極佳 | 離線處理、高精度需求 |
| yolov8x.pt | 68.2M | 最慢 | 最佳 | 離線處理、極高精度需求 |

```bash
# 使用不同大小的模型
python -m src.app --source 0 --model yolov8s.pt --show  # 小型模型
python -m src.app --source 0 --model yolov8m.pt --show  # 中型模型
python -m src.app --source 0 --model yolov8l.pt --show  # 大型模型
```

#### 鍵盤操作

- **`q` 或 `ESC`**：退出程式
- **`0-9` 鍵**：快速截圖並標記品質（0=最差，9=滿分）
- **`s`**：截圖儲存（無品質標記）

**品質標記截圖說明**：
```bash
# 執行程式
python -m src.app --source 0 --show

# 看到良品 → 按 '9' (滿分)
# 看到小瑕疵 → 按 '7' 或 '8'
# 看到中等缺陷 → 按 '4-6'
# 看到嚴重缺陷 → 按 '0-3'

# 截圖會自動分類到對應目錄：
# artifacts/screenshots/quality_0/  (最差)
# artifacts/screenshots/quality_9/  (滿分)
```

**分析收集的資料**：
```bash
# 查看品質分布統計
python scripts/analyze_quality_data.py

# 匯出品質清單
python scripts/analyze_quality_data.py --export
```

### 3. 執行測試

```bash
# 執行所有單元測試
./yolo/Scripts/python.exe -m pytest tests/ -v -m unit

# 執行特定測試
./yolo/Scripts/python.exe -m pytest tests/test_camera_stream.py -v
```

## 專案結構

```
yolo01/
├── src/                                  # 核心程式碼
│   ├── app/                             # 主程式與 CLI
│   │   ├── realtime.py                  # 即時推論（含品質標記）
│   │   ├── __main__.py                  # CLI 入口點
│   │   └── __init__.py
│   ├── camera/                          # 攝影機串流模組
│   │   ├── stream.py                    # VideoStream 類別
│   │   └── __init__.py
│   ├── detector/                        # YOLO 偵測器
│   │   ├── yolo.py                      # YoloDetector 類別
│   │   └── __init__.py
│   └── visualize/                       # 視覺化工具
│       ├── overlay.py                   # Overlay 類別
│       └── __init__.py
│
├── scripts/                              # 工具腳本
│   ├── 品質管理系統
│   │   ├── analyze_quality_data.py      # 分析品質標記統計
│   │   ├── prepare_quality_dataset.py   # 準備品質訓練資料集
│   │   ├── train_qc.py                  # 訓練品質管理模型
│   │   └── quality_inspector.py         # 品質檢測系統（顯示 0-9 分數）
│   ├── 行人偵測
│   │   ├── train_pedestrian.py          # 訓練行人偵測模型
│   │   └── evaluate_pedestrian.py       # 評估模型
│   ├── 工具腳本
│   │   ├── check_cuda.py                # 檢查 CUDA
│   │   ├── verify_dataset.py            # 驗證資料集
│   │   ├── install_with_uv.sh           # 安裝環境
│   │   ├── download_pedestrian_data.sh  # 下載資料集
│   │   ├── train_all.sh                 # 訓練所有模型
│   │   └── quality-inspect.sh           # 快速啟動品質檢測
│
├── tests/                                # 測試程式
│   ├── conftest.py                      # Pytest 配置
│   ├── test_camera_stream.py            # 攝影機測試
│   ├── test_yolo_detector.py            # 偵測器測試
│   ├── test_visualize.py                # 視覺化測試
│   ├── test_inference_pipeline.py       # 推論管道測試
│   └── test_dataset_integrity.py        # 資料集測試
│
├── data/                                 # 資料集
│   ├── pedestrian.yaml                  # 行人偵測配置
│   ├── quality_control/                 # 品質管理（二分類）
│   └── quality_scores/                  # 品質評分（10 分類）
│
├── artifacts/                            # 輸出和模型
│   ├── screenshots/                     # 品質標記截圖
│   │   ├── quality_0/ ~ quality_9/      # 品質等級 0-9
│   │   └── unlabeled/                   # 未標記
│   └── runs/qc/                         # 訓練結果
│       └── quality_control_v12/         # 最新模型
│           └── weights/best.pt
│
├── 文件                                  # 說明文件
│   ├── README.md                        # 本文件
│   ├── DEVELOP.md                       # 開發文件
│   ├── TODO.md                          # 待辦事項
│   ├── HOWTO.md                         # 完整訓練指南
│   ├── QUICK_START_QC.md                # 品質管理快速上手
│   ├── TRAINING_GUIDE.md                # 訓練完成後指南
│   ├── QUALITY_SCORE_GUIDE.md           # 品質評分系統指南
│   └── PROJECT_FILES.md                 # 專案檔案清單
│
├── requirements.txt                      # Python 依賴
├── pytest.ini                           # Pytest 配置
└── .gitignore                           # Git 忽略規則
```

## 常見使用情境

### 情境 1：家庭安全監控

**需求**：使用 USB 攝影機進行即時人物偵測

```bash
# 基本設定
python -m src.app --source 0 --model yolov8n.pt --conf 0.4 --show

# 如果有 GPU（推薦）
python -m src.app --source 0 --model yolov8s.pt --conf 0.4 --device cuda --show
```

**建議配置**：
- 模型：`yolov8n.pt` 或 `yolov8s.pt`
- 信心度：`0.4-0.5`（避免過多誤報）
- 設備：有 GPU 優先使用 GPU

### 情境 2：商店人流統計

**需求**：使用 IP 攝影機統計進出人數

```bash
# RTSP 串流偵測
python -m src.app \
    --source rtsp://admin:password@192.168.1.100:554/stream \
    --model yolov8s.pt \
    --conf 0.5 \
    --device cuda \
    --show
```

**建議配置**：
- 模型：`yolov8s.pt`（平衡速度與精度）
- 信心度：`0.5`（提高準確度）
- RTSP：使用攝影機主串流（較高解析度）

### 情境 3：停車場車輛偵測

**需求**：偵測停車場車輛並記錄

```bash
# 使用較大模型提高精度
python -m src.app \
    --source rtsp://192.168.1.101:554/stream \
    --model yolov8m.pt \
    --conf 0.6 \
    --device cuda \
    --show
```

**建議配置**：
- 模型：`yolov8m.pt`（更高精度）
- 信心度：`0.6`（減少誤報）
- 解析度：1280（修改程式碼）

### 情境 4：影片檔案分析

**需求**：分析錄影檔案中的物件

```bash
# 離線處理影片
python -m src.app \
    --source video.mp4 \
    --model yolov8l.pt \
    --conf 0.5 \
    --device cuda
```

**建議配置**：
- 模型：`yolov8l.pt` 或 `yolov8x.pt`（最高精度）
- 不使用 `--show`（背景處理）
- 結果可儲存至檔案（需額外開發）

### 情境 5：多攝影機系統

**需求**：同時監控多個攝影機

```bash
# 終端機 1 - 攝影機 0
python -m src.app --source 0 --model yolov8n.pt --show

# 終端機 2 - 攝影機 1
python -m src.app --source 1 --model yolov8n.pt --show

# 終端機 3 - RTSP 攝影機
python -m src.app --source rtsp://192.168.1.100:554/stream --model yolov8n.pt --show
```

**建議配置**：
- 使用輕量模型（`yolov8n.pt`）減少系統負擔
- 如果 GPU 記憶體不足，部分使用 CPU
- 考慮使用多 GPU 分散運算

### 情境 6：高解析度精準偵測

**需求**：4K 攝影機高精度偵測

```bash
# 修改 src/detector/yolo.py line 41
# imgsz=kwargs.get("imgsz", 1920)  # 使用 1920 解析度

python -m src.app \
    --source 0 \
    --model yolov8x.pt \
    --conf 0.7 \
    --device cuda \
    --show
```

**建議配置**：
- 模型：`yolov8x.pt`（最大模型）
- 解析度：1920（需要修改程式碼）
- 硬體：RTX 3080 或以上 GPU
- 信心度：`0.7`（極高精度）

## 品質管理系統

### 快速上手（5 分鐘）

```bash
# 1. 收集品質資料（按 0-9 鍵標記）
python -m src.app --source 0 --show

# 2. 分析資料分布
python scripts/analyze_quality_data.py

# 3. 準備訓練資料集
python scripts/prepare_quality_dataset.py --mode quality

# 4. 訓練模型
bash scripts/train_all.sh

# 5. 啟動品質檢測（顯示 0-9 品質分數）
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_control_v12/weights/best.pt \
    --source 0 \
    --device cuda
```

### 品質標記工作流程

1. **收集資料**（按 0-9 鍵）：
   ```bash
   python -m src.app --source 0 --show
   # 看到產品 → 按對應品質等級（0=最差, 9=滿分）
   # 截圖自動分類至 artifacts/screenshots/quality_X/
   ```

2. **分析統計**：
   ```bash
   python scripts/analyze_quality_data.py
   # 顯示各品質等級分布和平均品質
   ```

3. **準備資料集**：
   ```bash
   # 二分類（good/defect）
   python scripts/prepare_quality_dataset.py --mode binary

   # 三分類（good/minor_defect/major_defect）
   python scripts/prepare_quality_dataset.py --mode triclass

   # 10 級品質分類（quality_0 ~ quality_9）
   python scripts/prepare_quality_dataset.py --mode quality
   ```

4. **訓練模型**：
   ```bash
   python scripts/train_qc.py \
       --data data/quality_scores/dataset.yaml \
       --model yolov8n.pt \
       --epochs 50 \
       --device cuda
   ```

5. **品質檢測**：
   ```bash
   python scripts/quality_inspector.py \
       --model artifacts/runs/qc/quality_control_v12/weights/best.pt \
       --source 0 \
       --conf 0.25 \
       --device cuda
   ```

### 品質檢測系統特色

- ✅ **自動框選產品**：偵測並框出每個產品
- ✅ **顯示品質分數**：0-9 分（0=最差, 9=滿分）
- ✅ **彩色編碼**：
  - 🟢 綠色：高品質（7-9 分）
  - 🟡 黃色：中等品質（4-6 分）
  - 🔴 紅色：低品質（0-3 分）
- ✅ **即時統計**：右上角顯示各品質等級分布
- ✅ **平均品質**：自動計算平均品質分數

### 品質管理文件

- **[QUICK_START_QC.md](QUICK_START_QC.md)** - 5 分鐘快速上手
- **[HOWTO.md](HOWTO.md)** - 完整實作指南
- **[QUALITY_SCORE_GUIDE.md](QUALITY_SCORE_GUIDE.md)** - 品質評分系統使用指南
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - 訓練完成後使用指南

## 進階使用

### 訓練自訂模型

1. **準備資料集**：
```bash
# 下載行人標註範例資料（可選）
bash ./scripts/download_pedestrian_data.sh

# 驗證資料集
python scripts/verify_dataset.py --yaml data/pedestrian.yaml
```

2. **訓練模型**：
```bash
python scripts/train_pedestrian.py \
    --data data/pedestrian.yaml \
    --model yolov8s.pt \
    --epochs 50 \
    --imgsz 640
```

3. **評估模型**：
```bash
python scripts/evaluate_pedestrian.py
```

### 模型匯出

```bash
# 匯出為 ONNX 格式
python scripts/export_model.py --model yolov8s.pt --format onnx

# 匯出為 TensorRT（需要 NVIDIA GPU）
python scripts/export_model.py --model yolov8s.pt --format tensorrt
```

## CLI 參數詳細說明

### src.app 參數

```bash
python -m src.app [OPTIONS]
```

#### 所有參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--source` | int/str | `0` | 影像來源 |
| `--model` | str | `yolov8n.pt` | YOLO 模型路徑 |
| `--conf` | float | `0.25` | 信心度閾值（0.0-1.0） |
| `--iou` | float | `0.7` | IoU 閾值（0.0-1.0） |
| `--device` | str | 自動偵測 | 運算設備（cpu/cuda） |
| `--show` | flag | `False` | 是否顯示視窗 |
| `--names` | str | `data/pedestrian.yaml` | 類別名稱檔案 |

#### 參數詳細說明

**`--source`** - 影像來源
- **整數**：攝影機索引
  - `0`：預設攝影機（通常是內建鏡頭）
  - `1`, `2`, `3`...：外接攝影機
- **字串**：RTSP URL 或影片檔案路徑
  - RTSP: `rtsp://192.168.1.100:554/stream`
  - 影片: `video.mp4`, `path/to/video.avi`
  - 網路影片: `http://example.com/video.mp4`

**`--model`** - YOLO 模型
- 預訓練模型：`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- 自訓練模型：`artifacts/models/custom.pt`
- 首次執行會自動下載預訓練模型

**`--conf`** - 信心度閾值
- 範圍：`0.0` 到 `1.0`
- 預設：`0.25`（25%）
- **較低值**（如 `0.1`）：偵測更多物件，但可能有誤報
- **較高值**（如 `0.5`）：只顯示高信心度結果，減少誤報
- 建議值：
  - 即時監控：`0.3-0.4`
  - 精準偵測：`0.5-0.7`

**`--iou`** - IoU 閾值（非最大值抑制）
- 範圍：`0.0` 到 `1.0`
- 預設：`0.7`
- 用於移除重疊的偵測框
- **較低值**（如 `0.4`）：保留更多重疊框
- **較高值**（如 `0.8`）：移除更多重疊框
- 建議值：`0.5-0.7`

**`--device`** - 運算設備
- `cpu`：使用 CPU（較慢但相容性好）
- `cuda`：使用 NVIDIA GPU（需要 CUDA，速度快 10-50 倍）
- `cuda:0`：指定第一張 GPU
- `cuda:1`：指定第二張 GPU
- 不指定時自動偵測（有 GPU 就用 GPU）

**`--show`** - 顯示視窗
- 不加此參數：背景執行，不顯示視窗
- 加入此參數：顯示即時偵測視窗
- 適合遠端伺服器時不加此參數

**`--names`** - 類別名稱檔案
- YAML 格式檔案，定義類別名稱
- 預設：`data/pedestrian.yaml`
- 自訂：`data/custom.yaml`

#### 使用範例

```bash
# 範例 1：基本使用（攝影機 0，顯示視窗）
python -m src.app --source 0 --show

# 範例 2：高精度設定（提高信心度閾值）
python -m src.app --source 0 --model yolov8s.pt --conf 0.5 --show

# 範例 3：RTSP 串流 + GPU 加速
python -m src.app --source rtsp://192.168.1.100:554/stream --device cuda --show

# 範例 4：影片檔案處理（背景執行）
python -m src.app --source video.mp4 --model yolov8m.pt --conf 0.4

# 範例 5：多攝影機 + 自訂類別
python -m src.app --source 1 --names data/custom.yaml --show

# 範例 6：低延遲即時偵測
python -m src.app --source 0 --model yolov8n.pt --conf 0.3 --device cuda --show
```

### train_pedestrian.py 參數

```bash
python scripts/train_pedestrian.py [OPTIONS]

選項:
  --data PATH           資料集 YAML 檔案
  --model MODEL         基礎模型（yolov8n/s/m/l/x.pt）
  --epochs N            訓練輪數（預設：50）
  --imgsz SIZE          影像大小（預設：640）
  --batch SIZE          批次大小（預設：16）
  --device DEVICE       運算設備（cpu/cuda）
```

## 系統需求

### 最低需求
- Python 3.10+
- 4GB RAM
- CPU：Intel i5 或同等級

### 建議配置
- Python 3.10.12
- 8GB+ RAM
- NVIDIA GPU（CUDA 11.8 或 12.1）
- Git Bash（Windows 使用者）

## 依賴套件

核心依賴：
- `torch==2.4.1` - PyTorch 深度學習框架
- `ultralytics==8.3.49` - YOLOv8 實作
- `opencv-python>=4.9` - 影像處理
- `numpy>=1.24` - 數值運算
- `pytest>=7.0` - 測試框架

完整清單請參考 `requirements.txt`。

## 疑難排解

### 常見問題與解決方案

#### 1. imgsz 警告訊息

**問題**：
```
WARNING ⚠️ imgsz=[1080] must be multiple of max stride 32, updating to [1088]
```

**原因**：YOLO 模型要求輸入影像尺寸必須是 32 的倍數

**解決方案**：
- 系統會自動調整為最接近的 32 倍數
- 或手動指定符合的尺寸：
  ```python
  # 常用的合法尺寸（32 的倍數）
  640, 960, 1024, 1280, 1920
  ```

**程式碼說明**：
```python
# src/detector/yolo.py 預設使用 1280
imgsz=kwargs.get("imgsz", 1280)  # 可改為 640 提升速度
```

#### 2. 攝影機無法開啟

**錯誤訊息**：
```
ERROR: Cannot open camera source 0
```

**解決步驟**：
1. **檢查攝影機權限**（Windows 設定 → 隱私權 → 相機）
2. **嘗試不同的設備索引**：
   ```bash
   python -m src.app --source 0 --show  # 內建鏡頭
   python -m src.app --source 1 --show  # 外接鏡頭
   python -m src.app --source 2 --show  # 第二個外接鏡頭
   ```
3. **檢查攝影機是否被其他程式佔用**（如 Zoom、Teams、Skype）
4. **使用測試影片**確認程式正常：
   ```bash
   python -m src.app --source test.mp4 --show
   ```

#### 3. GPU 無法使用

**檢查 CUDA 是否可用**：
```python
import torch
print(torch.cuda.is_available())  # 應該顯示 True
print(torch.cuda.get_device_name(0))  # 顯示 GPU 名稱
```

**安裝 CUDA 版本的 PyTorch**：
```bash
# CUDA 11.8
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

**確認 CUDA 版本**：
```bash
nvidia-smi  # 查看 CUDA 版本
nvcc --version  # 確認 CUDA toolkit
```

#### 4. RTSP 串流連線失敗

**常見原因**：
- 網路不通或防火牆阻擋
- RTSP URL 格式錯誤
- 攝影機帳號密碼錯誤
- 攝影機不支援 RTSP

**測試步驟**：
```bash
# 1. 測試網路連線
ping 192.168.1.100

# 2. 使用 ffplay 測試 RTSP（如果已安裝 ffmpeg）
ffplay rtsp://admin:password@192.168.1.100:554/stream

# 3. 確認 RTSP URL 格式
# 海康威視: rtsp://admin:password@IP:554/Streaming/Channels/101
# 大華: rtsp://admin:password@IP:554/cam/realmonitor?channel=1&subtype=0
# 通用: rtsp://admin:password@IP:554/stream
```

#### 5. Windows 環境問題

**建議使用 Git Bash**：
```bash
# 下載 Git for Windows
https://git-scm.com/download/win

# 或使用 WSL（Windows Subsystem for Linux）
wsl --install
```

**路徑問題**：
```bash
# Windows 使用反斜線
python -m src.app --source C:\Videos\test.mp4

# Git Bash 可用正斜線
python -m src.app --source C:/Videos/test.mp4

# 或使用引號
python -m src.app --source "C:\Videos\test.mp4"
```

#### 6. 記憶體不足

**症狀**：程式崩潰或變慢

**解決方案**：
```bash
# 1. 使用較小的模型
python -m src.app --source 0 --model yolov8n.pt --show

# 2. 降低輸入解析度（修改 src/detector/yolo.py）
imgsz=kwargs.get("imgsz", 640)  # 改為 640

# 3. 關閉其他應用程式釋放記憶體
```

#### 7. FPS 太低

**優化建議**：

1. **使用 GPU**：
   ```bash
   python -m src.app --source 0 --device cuda --show
   ```

2. **使用較小模型**：
   ```bash
   python -m src.app --source 0 --model yolov8n.pt --show
   ```

3. **降低解析度**（修改 `src/detector/yolo.py` line 41）：
   ```python
   imgsz=kwargs.get("imgsz", 640)  # 從 1280 改為 640
   ```

4. **降低信心度閾值**：
   ```bash
   python -m src.app --source 0 --conf 0.3 --show
   ```

#### 8. 測試失敗

**執行測試時出錯**：
```bash
# 確保使用虛擬環境的 Python
./yolo/Scripts/python.exe -m pytest tests/ -v

# Windows PowerShell
.\yolo\Scripts\python.exe -m pytest tests\ -v

# Linux/macOS
./yolo/bin/python -m pytest tests/ -v
```

**跳過整合測試**：
```bash
# 只執行單元測試（不需模型）
pytest tests/ -v -m unit

# 跳過慢速測試
pytest tests/ -v -m "not slow"
```

### 效能基準

| 硬體配置 | 模型 | 解析度 | FPS |
|---------|------|--------|-----|
| CPU (i5-10400) | yolov8n | 640 | 8-12 |
| CPU (i5-10400) | yolov8s | 640 | 4-6 |
| RTX 3060 | yolov8n | 640 | 120+ |
| RTX 3060 | yolov8s | 640 | 80-100 |
| RTX 3060 | yolov8m | 1280 | 40-50 |
| RTX 4090 | yolov8x | 1280 | 80-100 |

## 測試

專案包含完整的測試套件：

```bash
# 執行所有測試
pytest tests/ -v

# 只執行單元測試
pytest tests/ -v -m unit

# 執行整合測試（需要模型）
pytest tests/ -v -m integration

# 生成測試覆蓋率報告
pytest --cov=src tests/
```

## 開發

詳細開發指引請參考 [DEVELOP.md](DEVELOP.md)。

開發流程：
1. 閱讀 `DEVELOP.md` 了解架構
2. 查看 `TODO.md` 了解待辦項目
3. 執行測試確保品質
4. 提交前檢查 `.gitignore`

## 授權

本專案採用 MIT 授權條款。

## 聯絡方式

如有問題或建議，請開啟 Issue。

## 致謝

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
