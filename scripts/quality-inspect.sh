confidential=${1-0.25}
echo "輸入信心閥值: 0.15 to 0.4"
echo "越低越有可能誤報"
python scripts/quality_inspector.py \
      --model artifacts/runs/qc/quality_control_v12/weights/best.pt \
      --source 0 \
      --conf $confidential \
      --device cuda
