#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prépare le dataset YOLO : copie images + labels de Mutwanga\2 vers train/val."""
import os
import shutil
import random

SRC_DIR = r"D:\GIS\Recherche\Segmentation\Decoupe Raster\Mutwanga\2"
DST_DIR = r"D:\GIS\Recherche\dataset_palmiers"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Collecter les tuiles annotées (qui ont un .txt)
tiles = []
for f in os.listdir(SRC_DIR):
    if f.startswith("mutwanga_tile_") and f.endswith(".txt"):
        base = f.replace(".txt", "")
        tiles.append(base)

tiles.sort(key=lambda x: int(x.split("_")[-1]))
random.shuffle(tiles)

n_train = int(len(tiles) * TRAIN_RATIO)
train_tiles = tiles[:n_train]
val_tiles = tiles[n_train:]

for split in ("train", "val"):
    for sub in ("images", "labels"):
        d = os.path.join(DST_DIR, sub, split)
        if os.path.exists(d):
            shutil.rmtree(d)

for split, tile_list in [("train", train_tiles), ("val", val_tiles)]:
    img_dir = os.path.join(DST_DIR, "images", split)
    lbl_dir = os.path.join(DST_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for tile in tile_list:
        src_img = os.path.join(SRC_DIR, tile + ".tif")
        src_lbl = os.path.join(SRC_DIR, tile + ".txt")
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(img_dir, tile + ".tif"))
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(lbl_dir, tile + ".txt"))

print(f"Train: {len(train_tiles)} tuiles | Val: {len(val_tiles)} tuiles")
