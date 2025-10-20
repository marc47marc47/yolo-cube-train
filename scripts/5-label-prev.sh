#!/usr/bin/env bash
confidential=${1-0.25}
echo "輸入信心閥值: 0.15 to 0.4"
echo "越低越有可能誤報"
lastBestPt=` find artifacts/runs/qc/qu*/* -type f -name 'best.pt' -exec ls {} \;|sort |tail -n 1`
if [ "$lastBestPt" = "" ]; then
	echo "No Best weights found"
	exit 1
fi

python -m src.app inspect \
      --source 0 \
      --model $lastBestPt \
      --conf $confidential \
      --device cuda


#      --model artifacts/runs/qc/quality_control_v1/weights/best.pt \
