# python compare_cka.py "/tmp2/igor/LL-tSNE/features_manual/COCO_R50_epoch_1?.pkl" "/tmp2/igor/LL-tSNE/features_manual/S_R50_epoch_121?.pkl"
# python compare_cka.py "/tmp2/igor/LL-tSNE/features_manual/COCO_R50_epoch_1?.pkl" "/tmp2/igor/LL-tSNE/features_manual/S_PUR50_epoch_126?.pkl"
# python compare_cka.py "/tmp2/igor/LL-tSNE/features_manual/COCO_R50_epoch_1?.pkl" "/tmp2/igor/LL-tSNE/features_manual/S_PURS50_epoch_112?.pkl"
import argparse
import os
import os.path as op

import torch
import numpy as np
import pickle
import tqdm
from CKA import kernel_CKA
from skimage.measure import block_reduce as maxpool

REPLACE_SYMBOL = "?"

# async def main():
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Use mmdet backbone to extract features')
    parser.add_argument('pickled_features', nargs=2, help="Pickled features, use ? to indicate the placeholder for layer number.")
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--size', type=int, default=25, help="Size to maxpool to")
    args = parser.parse_args()
    if not op.exists("cache"):
        os.mkdir("cache")
    for pickled_features in args.pickled_features:
        assert REPLACE_SYMBOL in pickled_features
        print(pickled_features)
    results = []
    for l in range(args.layers):
        X = []
        Y = []
        for pickled_features in args.pickled_features:
            cached_file = op.join("cache",f"size{args.size}_"+op.split(pickled_features.replace(REPLACE_SYMBOL, str(l)))[-1])
            if op.exists(cached_file):
                with open(cached_file, "rb") as handle:
                    print(f"Reading {cached_file}...")
                    x, y = pickle.load(handle)
            else:
                curr_file = pickled_features.replace(REPLACE_SYMBOL, str(l))
                print(f"Reading {curr_file}...")
                with open(curr_file, "rb") as handle:
                    data = pickle.load(handle)
                    x = []
                    y = []
                    while data:
                        sample = data.pop()
                        features = sample["features"]
                        C, H, W = features.shape
                        features = maxpool(features, (C, H//args.size, W//args.size))
                        C, H, W = features.shape
                        assert H == W == args.size
                        img_metas = sample["img_metas"]
                        x.append(features)
                        y.append(img_metas["ori_filename"])
                    x = np.array(x)
                    B, C, H, W = x.shape
                    x = x.reshape(B, -1)
                print("Writing cached file...")
                with open(cached_file, "wb") as handle:
                    pickle.dump((x,y),handle)
            X.append(x)
            Y.append(y)
        assert Y[0] == Y[1]
        X1, X2 = X
        print(f"Calculating kernel CKA for layer {l}...")
        results.append(kernel_CKA(X1,X2))
    for l, result in enumerate(results):
        print(f"Layer {l} : {result}")
