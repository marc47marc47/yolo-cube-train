# 開發指引

## 1. 專案目標
- 建立一個即時影像辨識流程，透過攝影機取得畫面並使用 YOLO 模型偵測畫面中的物件。
- 以可維護、可擴充為優先，後續能夠持續優化偵測效能、擴充模型或整合其他感測來源。

## 2. 快速開始（Quick Start）
想要快速測試功能的使用者，可依照以下步驟在 5 分鐘內完成：

1. **建立虛擬環境並安裝依賴**
   ```bash
   bash ./scripts/install_with_uv.sh
   source yolo/bin/activate  # Linux/macOS
   # 或 Windows Git Bash: source yolo/Scripts/activate
   ```

2. **執行即時辨識（使用預訓練模型）**
   ```bash
   python -m src.app --source 0 --model yolov8n.pt
   ```
   - `--source 0`：使用預設攝影機（可改為 RTSP URL 或影片檔案路徑）
   - `--model yolov8n.pt`：使用 YOLOv8 Nano 預訓練模型（首次執行會自動下載）
   - 按 `q` 離開，`p` 暫停/繼續，`s` 截圖

3. **驗證結果**
   - 畫面上會顯示偵測框、類別標籤與信心度
   - 左上角顯示即時 FPS

**注意**：若需要自訂訓練模型，請參考後續的「資料準備」與「訓練與微調」章節。

## 3. 技術選項評估
- **YOLO 版本**
  - 建議以 Ultralytics YOLOv8 為主（`pip install ultralytics`），提供 CLI 與 Python API，部署彈性高。
  - 若需要穩定的社群範例，可考慮 YOLOv5（GitHub: `ultralytics/yolov5`）；同樣支援即時推論與自訂訓練。
  - 針對低資源或邊緣硬體，可以改用 YOLO-NAS、YOLO-N 或 Nano 款模型。
- **模型輸入來源**
  - 開發階段先使用 USB 攝影機或筆電內建攝影機，透過 OpenCV (`cv2.VideoCapture`) 取得畫面。
  - 需要遠端攝影機時，可整合 RTSP / HTTP 串流，搭配 `ffmpeg` 或 `opencv-videoio` 解析。
- **開發語言與框架**
  - Python 3.9–3.11 均可；為兼顧相依套件支援與穩定性，推薦 Python 3.10.12。
  - 使用虛擬環境（`venv`、`uv venv` 或 `conda`）隔離依賴，避免系統相依衝突。

## 4. 開發環境需求
- **建議 Python 版本**
  - `Python 3.10.12`：Torch 2.4.x、Ultralytics YOLOv8 與主流影像處理套件皆已驗證支援此版本；同時仍可安裝 CUDA 11.x/12.x 的 GPU 相依套件。
- **必備 Python 套件**
  - `torch==2.4.1` 與 `torchvision==0.19.1`：YOLOv8 推論與訓練的核心運算引擎
    - **CPU 版本**（預設）：
      ```bash
      pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
      ```
    - **GPU 版本（CUDA 11.8）**：
      ```bash
      pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
      ```
    - **GPU 版本（CUDA 12.1）**：
      ```bash
      pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
      ```
  - `ultralytics==8.3.49`：提供 YOLOv8 CLI 與 Python API。
  - `opencv-python>=4.9,<5.0`：處理攝影機影像與繪製偵測結果。
  - `numpy>=1.24,<2.1`：數值運算與陣列操作。
  - `onnxruntime>=1.18,<1.19`：後續若需將模型匯出為 ONNX 時可直接推論。
  - `pyyaml>=6.0,<7.0`：處理 `data.yaml` 之類的設定檔。
  - 若需資料增強，可另外安裝 `albumentations`；若需標註工具，可安裝 `labelImg`（可視需求加到腳本中）。

## 5. 環境建置流程

### 5.1 安裝 uv 工具
建議透過 [`uv`](https://astral.sh/uv) 管理 Python 版本與套件，確保安裝速度與可重現性,不考慮powershell環境。

- **Linux/macOS**：
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows Git Bash**（推薦）：
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 5.2 建立虛擬環境並安裝依賴

執行專案附上的 `scripts/install_with_uv.sh`，建立虛擬環境並安裝所有必要套件：

- **Linux/macOS/Git Bash**：
  ```bash
  bash ./scripts/install_with_uv.sh
  source yolo/bin/activate  # Linux/macOS
  # 或 Windows Git Bash
  source yolo/Scripts/activate
  ```

**腳本功能說明**：
- 預設會安裝 Python 3.10.12 並建立 `yolo` 虛擬環境
- 套件清單集中於根目錄的 `requirements.txt`，腳本會透過 uv pip 一次安裝裡面列出的版本
- 需要自訂虛擬環境名稱、Python 版本或套件清單時，可分別設定 `VENV_PATH`、`PYTHON_VERSION`、`REQ_FILE` 環境變數，例如：
  ```bash
  VENV_PATH=.venv PYTHON_VERSION=3.11.7 REQ_FILE=alt-requirements.txt bash ./scripts/install_with_uv.sh
  ```

### 5.3 工具依賴檢查
建議提前確認系統已安裝以下指令工具（資料下載腳本會用到）：
- `curl`、`unzip`、`awk`

**Windows 使用者建議**：
- 使用 **Git Bash**（隨 Git for Windows 安裝）或 **WSL**（Windows Subsystem for Linux）執行腳本
- 若使用原生 PowerShell/CMD，需手動安裝這些工具或使用等價的 Windows 指令

### 5.4 GPU 支援設定
1. 安裝對應的 Python 與 GPU 驅動（NVIDIA 驅動 + CUDA + cuDNN，若使用 GPU）
2. 若需額外套件或 GPU 版本的 PyTorch，可編輯腳本中的套件清單或在虛擬環境內另行安裝（參考第 4 節的 GPU 安裝指令）

## 6. 開發流程建議

### 6.1 資料準備（兩種路徑）

**路徑 A：快速驗證（使用預訓練模型）**
- 直接使用 YOLOv8 預訓練模型（支援 80 種 COCO 類別）
- 適合：只需辨識常見物件（人、車、動物等）
- 執行：參考第 2 節「快速開始」

**路徑 B：自訓練模型（需要自訂資料集）**
1. 蒐集自家場景影像並進行標註（YOLO TXT 格式）
2. 若偵測結果不符合需求，進行微調
3. **行人標註範例下載**（可選）：
   ```bash
   bash ./scripts/download_pedestrian_data.sh
   ```
   - 下載並整理僅含 `person` 類別的訓練/驗證樣本
   - 預設輸出至 `data/reference/pedestrian`
   - 可透過環境變數 `DATA_ROOT`、`DATASET_URL` 調整儲存位置與資料來源網址
4. 驗證資料集：
   ```bash
   python scripts/verify_dataset.py --yaml data/pedestrian.yaml
   ```

### 6.2 即時推論程式

專案已實作完整的即時推論模組：
- `src/camera/stream.py`：封裝 OpenCV 攝影機擷取與 RTSP/HTTP 來源處理
- `src/detector/yolo.py`：載入 Ultralytics YOLOv8 模型，提供 `predict(frame)` 介面
- `src/visualize/overlay.py`：將偵測結果繪製成邊框、標籤、信心度，並附帶 FPS 顯示
- `src/app/realtime.py`：整合攝影機、偵測器與視覺化模組，加入鍵盤事件（暫停、截圖、結束）

**執行指令**：
```bash
# 使用預設攝影機
python -m src.app --source 0 --model yolov8n.pt

# 使用 RTSP 串流
python -m src.app --source rtsp://192.168.1.100:554/stream --model yolov8s.pt

# 使用影片檔案
python -m src.app --source video.mp4 --model yolov8m.pt
```

### 6.3 模型訓練 / 微調

1. **準備 data.yaml**：
   ```yaml
   path: data/reference/pedestrian
   train: train2017
   val: val2017
   names:
     0: person
   ```

2. **執行訓練**（使用 Python 腳本）：
   ```bash
   python scripts/train_pedestrian.py --data data/pedestrian.yaml --model yolov8s.pt --epochs 50 --imgsz 640
   ```

3. **使用 CLI 訓練**（替代方案）：
   ```bash
   yolo detect train data=data/pedestrian.yaml model=yolov8s.pt epochs=50 imgsz=640
   ```

4. **監控訓練結果**：
   - 訓練曲線與指標（mAP、Precision、Recall）會輸出至 `runs/train/expN/`
   - 最佳權重自動儲存至 `artifacts/latest_model.txt`

5. **評估模型**：
   ```bash
   python scripts/evaluate_pedestrian.py
   ```

### 6.4 性能與最佳化
- 依硬體資源選擇模型大小（`yolov8n/s/m/l/x`）：
  - `n`（Nano）：最快，適合低資源設備
  - `s`（Small）：速度與準確度平衡
  - `m`（Medium）：較高準確度
  - `l/x`（Large/XLarge）：最高準確度，需強力硬體
- 若要降低延遲：啟用半精度（FP16）、TensorRT 匯出、或使用批次推論
- 加入異常處理，確保攝影機中斷時程式能安全退出或重新連線

## 7. 測試與驗證

### 7.1 資料集完整性測試
```bash
# 驗證標註轉換、資料夾結構、空標註處理
pytest tests/test_dataset_integrity.py -v
```

### 7.2 推論管線測試
```bash
# 對樣本影像跑一次推論，確認輸出格式正確
pytest tests/test_inference_pipeline.py -v
```

### 7.3 執行所有測試
```bash
# 執行專案所有測試（需先安裝 pytest）
pytest tests/ -v

# 或執行特定測試檔案
pytest tests/test_dataset_integrity.py::test_yaml_format -v
```

### 7.4 性能與穩定性測試
- 建立自動化測試腳本，使用錄製影片或圖片集檢驗偵測結果是否達到門檻
- 針對不同光線、角度、遮蔽情境進行壓力測試，驗證穩定性與誤報率
- 若有自訂模型，至少保留訓練 / 驗證 / 測試三分資料拆分，避免過擬合

**測試設定檔**：
- `pytest.ini` 或 `pyproject.toml` 已設定測試路徑與選項
- 確保虛擬環境內已安裝 `pytest`：`pip install pytest`

## 8. 部署與整合
- **桌面 / 開發環境**：保留 Python 腳本直接執行，可加上簡易 GUI（PyQt / Tkinter）。
- **邊緣裝置**：將模型匯出為 ONNX / TensorRT / OpenVINO，配合 NVIDIA Jetson 或工控機執行。
- **服務化**：包裝成 REST API 或 gRPC 服務，前端透過串流或批次影像呼叫。
- 記錄推論訊息（結果、時間戳、信心度）以利後續分析。

## 9. 後續優化方向
- 加入追蹤器（DeepSORT、ByteTrack）提升連續影像的追蹤穩定性。
- 整合告警或通知模組（Webhook、LINE、Slack）。
- 蒐集誤判案例定期再訓練，建立持續改進的 MLOps 流程。

> 本文件整合 README 所述的核心需求（使用 YOLO 搭配攝影機進行物件辨識），提供開發流程與環境建議；後續可依實際場域、硬體與人力資源做細部調整。
