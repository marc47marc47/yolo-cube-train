lastBestPt=` find artifacts/runs/qc/qu*/* -type f -name 'best.pt' -exec ls {} \;|sort |tail -n 1`
if [ "$lastBestPt" = "" ]; then
	echo "No Best weights found"
	exit 1
fi
echo "python -m src.app --source 0 --model $lastBestPt --names data/quality_control/dataset.yaml --device 0 --source 0 --show"
python -m src.app --source 0 --model $lastBestPt --names data/quality_control/dataset.yaml --device 0 --source 0 --show

