python compare_weights.py /tmp2/igor/checkpoints/S_PURS50_epoch_112.pth --from /tmp2/igor/checkpoints/COCO_R50_epoch_1.pth --report_file_path=S_PURS50_COCO.csv
python compare_weights.py /tmp2/igor/checkpoints/S_PUR50_epoch_126.pth --from /tmp2/igor/checkpoints/COCO_R50_epoch_1.pth --report_file_path=S_PUR50_COCO.csv
python compare_weights.py /tmp2/igor/checkpoints/S_R50_epoch_121.pth --from /tmp2/igor/checkpoints/COCO_R50_epoch_1.pth --report_file_path=S_R50_COCO.csv
python compare_weights.py /tmp2/igor/checkpoints/S_PURS50_epoch_112.pth --from /tmp2/igor/checkpoints/S_R50_epoch_121.pth --report_file_path=S_PURS50_S_R50.csv
python compare_weights.py /tmp2/igor/checkpoints/S_PUR50_epoch_126.pth --from /tmp2/igor/checkpoints/S_R50_epoch_121.pth --report_file_path=S_PUR50_S_R50.csv
# python compare_weights.py /tmp2/igor/checkpoints/X.pth --from /tmp2/igor/checkpoints/S_R50_epoch_121.pth --report_file_path=S_X_S_R50.csv
