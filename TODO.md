# TODO 清單（依 DEVELOP.md 規劃的開發步驟）

## 第一階段：環境建置與基礎設置

### 1.1 安裝 uv 工具
- [ ] 依據作業系統安裝 uv 套件管理工具
  - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows Git Bash: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] 驗證 uv 安裝成功：`uv --version`

### 1.2 建立虛擬環境
- [x] 建立 `scripts/install_with_uv.sh` 腳本（如果尚未存在）
- [ ] 執行 `bash ./scripts/install_with_uv.sh` 建立 `yolo` 虛擬環境
- [ ] 啟動虛擬環境並驗證：
  - Linux/macOS: `source yolo/bin/activate`
  - Windows Git Bash: `source yolo/Scripts/activate`
- [ ] 確認 Python 版本：`python --version`（應為 3.10.12）
- [ ] 確認已安裝套件：`pip list`

### 1.3 安裝必備工具
- [ ] 確認系統已安裝 `curl`、`unzip`、`awk`
- [ ] Windows 使用者：確認使用 Git Bash 或 WSL 環境

### 1.4 建立 requirements.txt
- [x] 建立根目錄的 `requirements.txt`，包含以下套件：
  - `torch==2.4.1`
  - `torchvision==0.19.1`
  - `ultralytics==8.3.49`
  - `opencv-python>=4.9,<5.0`
  - `numpy>=1.24,<2.1`
  - `onnxruntime>=1.18,<1.19`
  - `pyyaml>=6.0,<7.0`
  - `pytest>=7.0`（用於測試）

### 1.5 GPU 支援設定（可選）
- [ ] 確認 NVIDIA 驅動已安裝
- [ ] 確認 CUDA 版本（11.8 或 12.1）
- [ ] 依據 CUDA 版本安裝對應的 PyTorch：
  - CUDA 11.8: `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118`
  - CUDA 12.1: `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121`

## 第二階段：資料準備

### 2.1 建立資料目錄結構
- [x] 建立 `data/` 目錄
- [x] 建立 `data/reference/` 目錄
- [x] 建立 `data/reference/pedestrian/` 目錄

### 2.2 下載行人標註範例資料（可選）
- [x] 建立 `scripts/download_pedestrian_data.sh` 腳本
- [x] 腳本功能：下載 COCO 資料集中的 person 類別樣本
- [ ] 執行 `bash ./scripts/download_pedestrian_data.sh` 下載資料
- [ ] 驗證資料下載至 `data/reference/pedestrian/train2017` 與 `val2017`

### 2.3 建立資料集設定檔
- [x] 建立 `data/pedestrian.yaml`，內容包含：
  ```yaml
  path: data/reference/pedestrian
  train: train2017
  val: val2017
  names:
    0: person
  ```

### 2.4 資料集驗證工具
- [x] 建立 `scripts/verify_dataset.py`
- [x] 實作功能：
  - 讀取 `pedestrian.yaml` 設定
  - 檢查影像檔案是否存在
  - 檢查標註檔案（.txt）是否與影像對應
  - 統計訓練集與驗證集樣本數
  - 檢查標註格式是否正確（YOLO 格式）
  - 處理空標註檔案的情況
- [x] 支援 CLI 參數：`--yaml` 指定設定檔路徑
- [ ] 執行驗證：`python scripts/verify_dataset.py --yaml data/pedestrian.yaml`

## 第三階段：即時推論程式開發

### 3.1 攝影機模組
- [x] 建立 `src/` 目錄結構
- [x] 建立 `src/camera/` 目錄
- [x] 建立 `src/camera/__init__.py`
- [x] 建立 `src/camera/stream.py`，實作功能：
  - `CameraStream` 類別：封裝 OpenCV 的 `cv2.VideoCapture`
  - 支援 USB 攝影機（device ID）
  - 支援 RTSP 串流（rtsp://...）
  - 支援 HTTP 串流（http://...）
  - 支援影片檔案（.mp4, .avi 等）
  - 提供 `read()` 方法取得畫面
  - 提供 `release()` 方法釋放資源
  - 加入錯誤處理（攝影機無法開啟、連線中斷等）

### 3.2 偵測器模組
- [x] 建立 `src/detector/` 目錄
- [x] 建立 `src/detector/__init__.py`
- [x] 建立 `src/detector/yolo.py`，實作功能：
  - `YOLODetector` 類別：封裝 Ultralytics YOLO 模型
  - `__init__()` 載入模型（支援 yolov8n/s/m/l/x.pt）
  - `predict(frame)` 方法：對單一畫面進行偵測
  - 回傳偵測結果：邊界框、類別、信心度
  - 支援 GPU/CPU 自動偵測
  - 支援批次推論（可選）
  - 支援半精度推論 FP16（可選）

### 3.3 視覺化模組
- [x] 建立 `src/visualize/` 目錄
- [x] 建立 `src/visualize/__init__.py`
- [x] 建立 `src/visualize/overlay.py`，實作功能：
  - `draw_detections()` 函式：在畫面上繪製偵測結果
  - 繪製邊界框（可調整顏色、粗細）
  - 繪製類別標籤
  - 繪製信心度分數
  - 繪製 FPS 資訊（左上角）
  - 支援中文標籤顯示（可選）

### 3.4 主程式整合
- [x] 建立 `src/app/` 目錄
- [x] 建立 `src/app/__init__.py`
- [x] 建立 `src/app/realtime.py`，實作功能：
  - `RealtimeDetector` 類別：整合攝影機、偵測器、視覺化
  - 主迴圈：持續讀取畫面 → 偵測 → 繪製 → 顯示
  - FPS 計算與顯示
  - 鍵盤事件處理：
    - `q`：退出程式
    - `p`：暫停/繼續
    - `s`：截圖儲存
    - `ESC`：退出程式
  - 異常處理：攝影機中斷時安全退出

### 3.5 命令列介面
- [x] 建立 `src/app/__main__.py`，實作功能：
  - 使用 `argparse` 解析命令列參數
  - 參數 `--source`：影像來源（0 為預設攝影機，或 RTSP URL、影片路徑）
  - 參數 `--model`：YOLO 模型路徑（預設 yolov8n.pt）
  - 參數 `--conf`：信心度閾值（預設 0.25）
  - 參數 `--device`：運算設備（cpu 或 cuda，預設自動偵測）
  - 參數 `--save-dir`：截圖儲存目錄（預設 screenshots/）
- [ ] 執行測試：`python -m src.app --source 0 --model yolov8n.pt`

## 第四階段：模型訓練與微調

### 4.1 訓練腳本開發
- [x] 建立 `scripts/train_pedestrian.py`，實作功能：
  - 使用 Ultralytics Python API (`from ultralytics import YOLO`)
  - 讀取 `data/pedestrian.yaml` 設定
  - CLI 參數支援：
    - `--data`：資料集設定檔路徑
    - `--model`：基礎模型（yolov8n/s/m/l/x.pt）
    - `--epochs`：訓練輪數（預設 50）
    - `--imgsz`：影像大小（預設 640）
    - `--batch`：批次大小（預設 16）
    - `--device`：運算設備（cpu/cuda）
    - `--project`：專案目錄（預設 runs/train）
    - `--name`：實驗名稱（預設 exp）
  - 訓練完成後自動儲存最佳權重路徑至 `artifacts/latest_model.txt`

### 4.2 建立 artifacts 目錄
- [x] 建立 `artifacts/` 目錄用於儲存模型資訊
- [x] 建立 `.gitignore` 規則（排除大型模型檔案）

### 4.3 訓練執行與監控
- [ ] 執行訓練：`python scripts/train_pedestrian.py --data data/pedestrian.yaml --model yolov8s.pt --epochs 50`
- [ ] 監控訓練過程：觀察 `runs/train/expN/` 目錄
- [ ] 檢查訓練曲線：`runs/train/expN/results.png`
- [ ] 檢查最佳權重：`runs/train/expN/weights/best.pt`

### 4.4 模型評估工具
- [x] 建立 `scripts/evaluate_pedestrian.py`，實作功能：
  - 讀取 `artifacts/latest_model.txt` 取得最新模型路徑
  - 或使用 CLI 參數 `--model` 指定模型路徑
  - 對驗證集進行評估
  - 輸出評估報告：
    - mAP (mean Average Precision)
    - Precision（精確率）
    - Recall（召回率）
    - F1-Score
  - 產生混淆矩陣
  - 儲存評估結果至 `runs/eval/`
- [ ] 執行評估：`python scripts/evaluate_pedestrian.py`

## 第五階段：測試與自動化

### 5.1 測試目錄結構
- [x] 建立 `tests/` 目錄
- [x] 建立 `tests/__init__.py`
- [x] 建立 `tests/conftest.py`（pytest 設定檔）

### 5.2 資料集完整性測試
- [x] 建立 `tests/test_dataset_integrity.py`，測試項目：
  - `test_yaml_format()`：驗證 YAML 格式正確性
  - `test_yaml_paths_exist()`：檢查 YAML 中的路徑是否存在
  - `test_images_exist()`：檢查所有影像檔案是否存在
  - `test_labels_exist()`：檢查所有標註檔案是否存在
  - `test_labels_format()`：驗證標註檔案格式（YOLO TXT 格式）
  - `test_labels_values()`：驗證標註數值範圍（0-1 之間）
  - `test_class_ids()`：驗證類別 ID 是否在定義範圍內
  - `test_empty_labels()`：測試空標註檔案的處理
  - `test_dataset_statistics()`：統計資料集樣本數

### 5.3 推論管線測試
- [x] 建立 `tests/test_inference_pipeline.py`，測試項目：
  - `test_model_loading()`：測試模型載入
  - `test_image_inference()`：測試單張影像推論
  - `test_output_format()`：驗證輸出格式正確
  - `test_batch_inference()`：測試批次推論
  - `test_gpu_inference()`：測試 GPU 推論（如果可用）
  - `test_invalid_input()`：測試無效輸入的錯誤處理
  - `test_edge_cases()`：測試邊界情況（空白影像、極小影像等）

### 5.4 單元測試
- [x] 建立 `tests/test_camera_stream.py`：測試攝影機模組
- [x] 建立 `tests/test_yolo_detector.py`：測試偵測器模組
- [x] 建立 `tests/test_visualize.py`：測試視覺化模組

### 5.5 測試設定
- [x] 建立 `pytest.ini` 或在 `pyproject.toml` 中設定 pytest
- [x] 設定測試路徑：`testpaths = tests`
- [x] 設定測試選項：`addopts = -v --tb=short`
- [x] 確保虛擬環境內已安裝 `pytest`：`pip install pytest`

### 5.6 執行測試
- [ ] 執行所有測試：`pytest tests/ -v`
- [ ] 執行特定測試檔：`pytest tests/test_dataset_integrity.py -v`
- [ ] 執行特定測試函式：`pytest tests/test_dataset_integrity.py::test_yaml_format -v`
- [ ] 產生測試覆蓋率報告：`pytest --cov=src tests/`

### 5.7 性能與壓力測試
- [ ] 建立 `tests/test_performance.py`：
  - 測試不同模型大小的推論速度
  - 測試不同解析度的處理效能
  - 測試長時間運行的穩定性
- [ ] 建立測試資料集：不同光線、角度、遮蔽情境的影像
- [ ] 記錄測試結果：FPS、準確率、誤報率

## 第六階段：部署與擴充

### 6.1 模型匯出工具
- [ ] 建立 `scripts/export_model.py`，實作功能：
  - 支援匯出格式：
    - ONNX（跨平台推論）
    - TorchScript（PyTorch 部署）
    - TensorRT（NVIDIA GPU 加速）
    - OpenVINO（Intel 硬體加速）
  - CLI 參數：
    - `--model`：模型路徑
    - `--format`：匯出格式（onnx/torchscript/tensorrt/openvino）
    - `--imgsz`：影像大小
    - `--output`：輸出路徑
  - 驗證匯出模型可正常推論
- [ ] 執行測試：`python scripts/export_model.py --model yolov8s.pt --format onnx`

### 6.2 Docker 容器化（可選）
- [ ] 建立 `docker/` 目錄
- [ ] 建立 `docker/Dockerfile`，包含：
  - 基礎映像：Python 3.10
  - 安裝必要套件
  - 複製專案檔案
  - 設定入口點
- [ ] 建立 `docker/docker-compose.yml`（可選）
- [ ] 建立 `.dockerignore`
- [ ] 測試建置：`docker build -f docker/Dockerfile -t yolo-detector .`
- [ ] 測試執行：`docker run --rm -it yolo-detector`

### 6.3 使用文件
- [ ] 建立 `docs/` 目錄
- [ ] 建立 `docs/USAGE.md`，內容包含：
  - 環境啟動步驟
  - 即時推論使用說明
  - 訓練流程說明
  - 模型評估說明
  - 模型匯出說明
  - 常見問題排解（FAQ）
  - 效能調優建議

### 6.4 REST API 服務（可選）
- [ ] 建立 `src/api/` 目錄
- [ ] 使用 FastAPI 或 Flask 建立 API 服務：
  - POST `/detect`：上傳影像進行偵測
  - POST `/detect/batch`：批次偵測
  - GET `/health`：健康檢查
  - GET `/models`：列出可用模型
- [ ] 支援影像串流推論
- [ ] 加入 API 文件（Swagger/OpenAPI）

## 第七階段：後續優化方向（進階功能）

### 7.1 物件追蹤整合
- [ ] 研究追蹤演算法：DeepSORT、ByteTrack
- [ ] 建立 `src/tracking/` 目錄
- [ ] 實作追蹤器整合：
  - 為每個偵測物件分配 ID
  - 追蹤物件在連續畫面中的移動
  - 處理遮蔽與重新出現
- [ ] 加入追蹤軌跡視覺化

### 7.2 告警與通知模組
- [ ] 建立 `src/notification/` 目錄
- [ ] 實作告警條件判斷：
  - 特定物件出現
  - 物件數量超過閾值
  - 物件停留時間過長
- [ ] 整合通知介面：
  - Webhook
  - LINE Notify
  - Slack
  - Email
- [ ] 加入告警歷史記錄

### 7.3 資料回收與再訓練
- [ ] 建立 `datasets/misclassified/` 目錄
- [ ] 實作誤判案例收集機制：
  - 使用者標記錯誤偵測
  - 自動儲存低信心度預測
- [ ] 建立再訓練流程：
  - 整理誤判資料
  - 重新標註
  - 增量訓練
- [ ] 建立 MLOps 流程文件

### 7.4 GUI 介面（可選）
- [ ] 使用 PyQt6 或 Tkinter 建立圖形介面
- [ ] 功能包含：
  - 選擇影像來源
  - 選擇模型
  - 調整偵測參數
  - 顯示即時畫面
  - 統計資訊面板

### 7.5 效能最佳化
- [ ] 實作 FP16 半精度推論
- [ ] 實作 TensorRT 加速
- [ ] 實作多執行緒影像讀取
- [ ] 實作 GPU 批次推論
- [ ] 效能基準測試與比較

## 專案管理

### 文件維護
- [x] 更新 README.md：加入使用範例與安裝說明
- [x] 保持 DEVELOP.md 與實作同步
- [ ] 記錄重要決策與變更

### 版本控制
- [x] 建立 .gitignore：排除資料、模型、執行結果
- [ ] 提交基礎架構程式碼
- [ ] 為重要里程碑建立 Git tag

### 持續整合（可選）
- [ ] 設定 GitHub Actions 或 GitLab CI
- [ ] 自動執行測試
- [ ] 自動產生文件
- [ ] 程式碼品質檢查（pylint, black, mypy）

---

## 注意事項

1. **開發優先順序**：建議按照第一階段 → 第三階段 → 第二階段 → 第四階段的順序進行，可先用預訓練模型驗證功能。
2. **測試驅動**：每完成一個模組，立即撰寫對應的單元測試。
3. **漸進式開發**：先實作最小可行版本（MVP），再逐步擴充功能。
4. **文件同步**：程式碼變更時同步更新文件。
5. **效能監控**：記錄各版本的 FPS 與準確率，持續優化。
