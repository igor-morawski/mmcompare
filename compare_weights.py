"""
Copy to mmdetection/tools/
"""
# python compare_weights.py /tmp2/igor/checkpoints/S_PURS50_epoch_112.pth --from /tmp2/igor/checkpoints/COCO_R50_epoch_1.pth --report_file_path=S_PURS50_COCO.csv
# python compare_weights.py /tmp2/igor/checkpoints/S_PUR50_epoch_126.pth --from /tmp2/igor/checkpoints/COCO_R50_epoch_1.pth --report_file_path=S_PUR50_COCO.csv
# python compare_weights.py /tmp2/igor/checkpoints/S_R50_epoch_121.pth --from /tmp2/igor/checkpoints/COCO_R50_epoch_1.pth --report_file_path=S_R50_COCO.csv
# python compare_weights.py /tmp2/igor/checkpoints/S_PURS50_epoch_112.pth --from /tmp2/igor/checkpoints/S_R50_epoch_121.pth --report_file_path=S_PURS50_S_R50.csv
# python compare_weights.py /tmp2/igor/checkpoints/S_PUR50_epoch_126.pth --from /tmp2/igor/checkpoints/S_R50_epoch_121.pth --report_file_path=S_PUR50_S_R50.csv
# python compare_weights.py /tmp2/igor/checkpoints/X.pth --from /tmp2/igor/checkpoints/S_R50_epoch_121.pth --report_file_path=S_X_S_R50.csv
# python compare_weights.py /tmp2/igor/checkpoints/S_PUR50_epoch_126.pth --from /tmp2/igor/checkpoints/S_R50_epoch_121.pth --report_file_path=S_PUR50_S_R50.csv

# python compare_weights.py /tmp2/igor/checkpoints/S_PURS50_epoch_112.pth --from /auto/phd/09/igor/UNet_1602_184712_130000.pth --report_file_path=S_PURS50_UNet.csv
# python compare_weights.py /tmp2/igor/checkpoints/S_PUR50_epoch_126.pth --from /auto/phd/09/igor/UNet_1602_184712_130000.pth --report_file_path=S_PUR50_UNet.csv



import argparse
import os
import os.path as op

import torch
import numpy as np
import pandas as pd 

# async def main():
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Use mmdet backbone to extract features')
    parser.add_argument('A', type=str)
    parser.add_argument('--from', dest="B", type=str)
    parser.add_argument('--report_file_path', help="Report file path, must end in csv")
    args = parser.parse_args()
    assert args.report_file_path.endswith(".csv")
    name = op.split(args.report_file_path)[-1].split(".")[0]
    A_d = torch.load(args.A)["state_dict"]
    result = {}
    for layer in A_d.keys():
        if ("weight" not in layer) or ("conv" not in layer):
            continue
        if layer not in B_d.keys():
            result[layer] = None
            continue
        A_w = A_d[layer].cuda().float()
        B_w = B_d[layer].cuda().float()
        if A_w.shape != B_w.shape:
            result[layer] = None
            continue
        result[layer] = torch.pow(torch.sum(torch.pow(A_w-B_w,2)),0.5).detach().cpu().numpy()
    report = pd.DataFrame(data=result, index=[name]).transpose()
    report.to_csv(args.report_file_path)