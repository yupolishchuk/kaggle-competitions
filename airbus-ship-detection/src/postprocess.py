import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset import IMAGE_SIZE, read_mask
from metric import f2_image
from rle import IMAGE_SHAPE, rle_decode, rle_encode

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def mask_to_rles(probs, threshold=0.5, min_area=0, connectivity=8):
    # Збільшуємо ймовірності, а не бінарну маску: після порогу межі гладкі, без «сходинок».
    probs = cv2.resize(probs.astype(np.float32), IMAGE_SHAPE[::-1], interpolation=cv2.INTER_LINEAR)
    mask = (probs > threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    # Мітка 0 — фон.
    return [
        rle_encode(labels == k)
        for k in range(1, n_labels)
        if stats[k, cv2.CC_STAT_AREA] >= min_area
    ]


if __name__ == "__main__":
    masks = pd.read_csv(DATA_DIR / "train_ship_segmentations_v2.csv")
    split = pd.read_csv(DATA_DIR / "split.csv")

    # Перевірка на маленькому прикладі: два квадрати, що торкаються кутами.
    probs = np.zeros((384, 384), dtype=np.float32)
    probs[10:20, 10:20] = 1
    probs[20:30, 20:30] = 1
    probs[100:101, 100:101] = 1
    assert len(mask_to_rles(probs, connectivity=8)) == 2
    assert len(mask_to_rles(probs, connectivity=4)) == 3
    assert len(mask_to_rles(probs, connectivity=4, min_area=50)) == 2
    print("ручні приклади: OK")

    # Oracle: замість U-Net — правильні маски з CSV. Знімки не потрібні, тож беремо всю val
    # зі знімками з кораблями. Втрати тут — лише від постобробки та роздільності.
    val_ids = split.loc[(split["split"] == "val") & (split["n_ships"] > 0), "ImageId"]
    ships = masks.dropna(subset=["EncodedPixels"])
    rles = ships[ships["ImageId"].isin(val_ids)].groupby("ImageId")["EncodedPixels"].apply(list)

    variants = {
        "768, 8-зв'язність": dict(size=768, connectivity=8),
        "768, 4-зв'язність": dict(size=768, connectivity=4),
        "384, 8-зв'язність": dict(size=IMAGE_SIZE, connectivity=8),
    }
    start = time.time()
    for name, params in variants.items():
        scores, merged = [], 0
        for image_id, image_rles in rles.items():
            true_masks = [rle_decode(rle) for rle in image_rles]
            probs = read_mask(image_rles, size=params["size"]).astype(np.float32)
            pred_rles = mask_to_rles(probs, connectivity=params["connectivity"])
            merged += len(pred_rles) < len(true_masks)
            scores.append(f2_image(true_masks, [rle_decode(rle) for rle in pred_rles]))
        print(
            f"{name}: F2 на знімках з кораблями {np.mean(scores):.4f} | "
            f"знімків, де компонент менше, ніж кораблів: {merged} з {len(scores)} | "
            f"{time.time() - start:.0f} с"
        )
