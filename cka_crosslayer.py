# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/COCO_R50_epoch_1?.pkl"
# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/S_R50_epoch_121?.pkl"
# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/S_PUR50_epoch_126?.pkl"
# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/S_PURS50_epoch_112?.pkl"

# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/unet_S_PURS50_epoch_112?.pkl"
# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/unet_S_PUR50_epoch_126?.pkl"

# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/unet_S_PUR50_epoch_126?.pkl" --layers=3
# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/unet_S_PURS50_epoch_112?.pkl" --layers=3
# python cka_crosslayer.py "/tmp2/igor/LL-tSNE/features_manual/unet_pretrained?.pkl" --layers=3


import argparse
import os
import os.path as op

import torch
import numpy as np
import pickle
import tqdm
from CKA import kernel_CKA
from skimage.measure import block_reduce as maxpool

import matplotlib.pyplot as plt

REPLACE_SYMBOL = "?"

def load_features(args, l):
    cached_file = op.join("cache",f"size{args.size}_"+op.split(args.pickled_features.replace(REPLACE_SYMBOL, str(l)))[-1])
    if op.exists(cached_file):
        with open(cached_file, "rb") as handle:
            print(f"Reading {cached_file}...")
            x, y = pickle.load(handle)
    else:
        curr_file = args.pickled_features.replace(REPLACE_SYMBOL, str(l))
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
    return x, y

# async def main():
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Use mmdet backbone to extract features')
    parser.add_argument('pickled_features', help="Pickled features, use ? to indicate the placeholder for layer number.")
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--size', type=int, default=25, help="Size to maxpool to")
    args = parser.parse_args()
    if not op.exists("cache"):
        os.mkdir("cache")
    if not op.exists("plots"):
        os.mkdir("plots")
    assert REPLACE_SYMBOL in args.pickled_features
    results = np.ones([args.layers,args.layers]) * -1
    for l1 in range(args.layers):
        X1, Y1 = load_features(args, l1)
        for l2 in range(args.layers):
            if results[l1, l2] != -1:
                continue
            X2, Y2 = load_features(args, l2)
            assert Y1 == Y2
            print(f"Calculating kernel CKA for layers {l1} {l2}...")
            results[l1,l2]=kernel_CKA(X1,X2)
    print("Result: ")
    print(results)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(np.flip(results, axis=1),vmin=0, vmax=1)
    heatmap = "plasma"
    imgplot.set_cmap(heatmap)
    ax.set_title(op.split(args.pickled_features.replace(REPLACE_SYMBOL, ""))[-1].split(".")[0])
    ax.set_xlabel("Blocks")
    ax.set_ylabel("Blocks")
    plt.yticks(list(range(args.layers)))
    plt.xticks(list(range(args.layers)))
    plt.colorbar()
    fig.savefig(op.join("plots",op.split(args.pickled_features.replace(REPLACE_SYMBOL, ""))[-1].split(".")[0]+".png"))