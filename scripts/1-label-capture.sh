#!/usr/bin/env bash
echo "透過按下鍵盤0~9來進行標記,0是最低分，9是最高分"
echo "按下v可以錄下測試影片檔，至少錄製3個用於訓練2各，測試一個"
echo "ctrl + c可以中斷程式"
python -m src.app realtime --source 0 --show --mode capture
