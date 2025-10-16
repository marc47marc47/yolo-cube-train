# 專案檔案清單

本文件列出專案中所有的腳本、模組和文件。

## 📂 專案結構

```
yolo01/
├── src/                          # 核心程式碼
│   ├── __init__.py
│   ├── app/                      # 主應用程式
│   │   ├── __init__.py
│   │   ├── __main__.py          # CLI 入口點
│   │   └── realtime.py          # 即時推論主程式
│   ├── camera/                   # 攝影機串流模組
│   │   ├── __init__.py
│   │   └── stream.py            # VideoStream 類別
│   ├── detector/                 # YOLO 偵測器
│   │   ├── __init__.py
│   │   └── yolo.py              # YoloDetector 類別
│   └── visualize/                # 視覺化工具
│       ├── __init__.py
│       └── overlay.py           # Overlay 類別
│
├── scripts/                      # 工具腳本
│   ├── Python 腳本
│   │   ├── analyze_quality_data.py       # 分析品質標記資料
│   │   ├── check_cuda.py                 # 檢查 CUDA 可用性
│   │   ├── evaluate_pedestrian.py        # 評估行人偵測模型
│   │   ├── prepare_quality_dataset.py    # 準備品質管理資料集
│   │   ├── quality_inspector.py          # 品質檢測系統（顯示 0-9 分數）
│   │   ├── train_pedestrian.py           # 訓練行人偵測模型
│   │   ├── train_qc.py                   # 訓練品質管理模型
│   │   └── verify_dataset.py             # 驗證資料集
│   │
│   └── Shell 腳本
│       ├── download_pedestrian_data.sh   # 下載行人資料集
│       ├── install_with_uv.sh            # 使用 uv 安裝環境
│       ├── label-image.sh                # 標記影像（LabelImg）
│       ├── label-result.sh               # 標記結果視覺化
│       ├── quality-inspect.sh            # 啟動品質檢測系統
│       ├── run_basic.sh                  # 基本執行腳本
│       ├── test_basic.sh                 # 基本測試腳本
│       └── train_all.sh                  # 訓練所有模型
│
├── tests/                        # 測試程式
│   ├── conftest.py              # Pytest 配置和 fixtures
│   ├── test_camera_stream.py    # 攝影機串流測試
│   ├── test_dataset_integrity.py # 資料集完整性測試
│   ├── test_inference_pipeline.py # 推論管道測試
│   ├── test_visualize.py        # 視覺化測試
│   └── test_yolo_detector.py    # YOLO 偵測器測試
│
├── data/                         # 資料集
│   ├── pedestrian.yaml          # 行人偵測資料集配置
│   ├── quality_control/         # 品質管理資料集（二分類）
│   │   ├── dataset.yaml
│   │   ├── images/
│   │   └── labels/
│   └── quality_scores/          # 品質評分資料集（10 分類）
│       ├── dataset.yaml
│       ├── images/
│       └── labels/
│
├── artifacts/                    # 輸出和模型
│   ├── screenshots/             # 品質標記截圖
│   │   ├── quality_0/          # 品質等級 0
│   │   ├── quality_1/          # 品質等級 1
│   │   ├── ...
│   │   ├── quality_9/          # 品質等級 9
│   │   └── unlabeled/          # 未標記的截圖
│   └── runs/qc/                # 訓練結果
│       └── quality_control_v12/ # 最新訓練結果
│           └── weights/
│               ├── best.pt      # 最佳模型
│               └── last.pt      # 最後模型
│
├── 文件
│   ├── README.md                # 專案總覽
│   ├── DEVELOP.md               # 開發文件
│   ├── TODO.md                  # 待辦事項
│   ├── HOWTO.md                 # 完整訓練指南
│   ├── QUICK_START_QC.md        # 品質管理快速上手
│   ├── TRAINING_GUIDE.md        # 訓練完成後使用指南
│   ├── QUALITY_SCORE_GUIDE.md   # 品質評分系統指南
│   └── PROJECT_FILES.md         # 本文件
│
├── 配置檔案
│   ├── requirements.txt         # Python 依賴
│   ├── pytest.ini              # Pytest 配置
│   └── .gitignore              # Git 忽略規則
│
└── yolo/                        # 虛擬環境（uv 創建）
```

## 📝 核心模組說明

### src/app/realtime.py
即時推論主程式，整合攝影機、偵測器和視覺化。

**主要功能**：
- 解析命令列參數
- 初始化 YOLO 模型
- 從攝影機/影片讀取幀
- 執行物件偵測
- 繪製檢測結果
- **品質標記截圖**（按 0-9 鍵）

**使用方式**：
```bash
python -m src.app --source 0 --model yolov8n.pt --show
```

### src/camera/stream.py
攝影機串流管理模組。

**類別**：
- `StreamConfig`：串流配置（source, width, height, fps）
- `VideoStream`：影片串流類別（支援攝影機、RTSP、影片檔案）

### src/detector/yolo.py
YOLO 偵測器封裝。

**類別**：
- `YoloDetector`：YOLO 模型封裝
  - `predict()`：執行推論
  - `load_labels()`：載入類別名稱

### src/visualize/overlay.py
視覺化繪製工具。

**類別**：
- `Box`：邊界框資料類別
- `Overlay`：繪製偵測結果
  - `draw()`：繪製邊界框和標籤
  - FPS 計算和顯示

## 🛠️ 腳本說明

### 品質管理相關

#### analyze_quality_data.py
分析品質標記截圖的統計資訊。

```bash
python scripts/analyze_quality_data.py
python scripts/analyze_quality_data.py --export  # 匯出清單
```

#### prepare_quality_dataset.py
將品質標記截圖轉換為 YOLO 訓練格式。

```bash
# 二分類（good/defect）
python scripts/prepare_quality_dataset.py --mode binary

# 三分類
python scripts/prepare_quality_dataset.py --mode triclass

# 10 級品質分類
python scripts/prepare_quality_dataset.py --mode quality
```

#### train_qc.py
訓練品質管理模型。

```bash
python scripts/train_qc.py \
    --data data/quality_control/dataset.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --device cuda
```

#### quality_inspector.py
品質檢測系統，顯示 0-9 品質分數。

```bash
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_control_v12/weights/best.pt \
    --source 0 \
    --conf 0.25 \
    --device cuda
```

**特色功能**：
- 框出產品並顯示品質分數（0-9）
- 彩色編碼（紅→黃→綠）
- 即時統計資訊
- 按 'r' 重置統計

### 行人偵測相關

#### train_pedestrian.py
訓練行人偵測模型。

#### evaluate_pedestrian.py
評估行人偵測模型性能。

### 工具腳本

#### check_cuda.py
檢查 CUDA 和 GPU 可用性。

```bash
python scripts/check_cuda.py
```

#### verify_dataset.py
驗證資料集格式和完整性。

```bash
python scripts/verify_dataset.py --yaml data/pedestrian.yaml
```

### Shell 腳本

#### install_with_uv.sh
使用 uv 建立虛擬環境並安裝依賴。

```bash
bash scripts/install_with_uv.sh
```

#### train_all.sh
訓練所有模型（一鍵執行）。

```bash
bash scripts/train_all.sh
```

#### quality-inspect.sh
快速啟動品質檢測系統。

```bash
bash scripts/quality-inspect.sh
```

## 📊 資料集格式

### YOLO 格式標記檔案

```
<class_id> <x_center> <y_center> <width> <height>
```

所有座標值為 0-1 之間的歸一化值。

### 資料集 YAML 配置

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 10  # 類別數量

names:
  0: quality_0
  1: quality_1
  ...
  9: quality_9
```

## 🧪 測試

### 執行測試

```bash
# 所有測試
pytest tests/ -v

# 只執行單元測試
pytest tests/ -v -m unit

# 特定測試檔案
pytest tests/test_camera_stream.py -v
```

### 測試覆蓋率

```bash
pytest --cov=src tests/
```

## 📦 依賴套件

### 核心依賴
- `torch==2.4.1` - PyTorch
- `ultralytics==8.3.49` - YOLOv8
- `opencv-python>=4.9` - 影像處理
- `numpy>=1.24` - 數值運算

### 開發依賴
- `pytest>=7.0` - 測試框架
- `pytest-cov` - 測試覆蓋率（可選）

## 🔄 工作流程

### 1. 品質資料收集
```bash
python -m src.app --source 0 --show
# 按 0-9 鍵標記品質
```

### 2. 資料分析
```bash
python scripts/analyze_quality_data.py
```

### 3. 準備資料集
```bash
python scripts/prepare_quality_dataset.py --mode quality
```

### 4. 訓練模型
```bash
python scripts/train_qc.py \
    --data data/quality_scores/dataset.yaml \
    --model yolov8n.pt \
    --epochs 50
```

### 5. 品質檢測
```bash
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_control_v12/weights/best.pt \
    --source 0
```

## 📝 文件索引

- **README.md** - 專案總覽、安裝和基本使用
- **DEVELOP.md** - 開發環境設定和架構說明
- **TODO.md** - 開發待辦事項和階段規劃
- **HOWTO.md** - 生產線品質管理完整實作指南
- **QUICK_START_QC.md** - 品質管理 5 分鐘快速上手
- **TRAINING_GUIDE.md** - 訓練完成後使用指南
- **QUALITY_SCORE_GUIDE.md** - 品質評分系統（0-9 分數）使用指南
- **PROJECT_FILES.md** - 本文件，專案檔案清單

## 🎯 重要路徑

### 已訓練模型
```
artifacts/runs/qc/quality_control_v12/weights/best.pt
```

### 資料集
```
data/quality_control/         # 二分類資料集
data/quality_scores/          # 10 級品質評分資料集
```

### 品質標記截圖
```
artifacts/screenshots/quality_0/  到  quality_9/
```

## 🚀 快速指令

```bash
# 收集品質資料
python -m src.app --source 0 --show

# 分析資料
python scripts/analyze_quality_data.py

# 訓練模型
bash scripts/train_all.sh

# 品質檢測
python scripts/quality_inspector.py \
    --model artifacts/runs/qc/quality_control_v12/weights/best.pt \
    --source 0 \
    --device cuda
```
