from pathlib import Path

import numpy as np
import pandas as pd

from rle import rle_decode

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def f2_image(true_masks, pred_masks):
    if len(true_masks) == 0 and len(pred_masks) == 0:
        return 1.0

    ious = np.zeros((len(true_masks), len(pred_masks)))
    for i, true_mask in enumerate(true_masks):
        for j, pred_mask in enumerate(pred_masks):
            ious[i, j] = iou(true_mask, pred_mask)

    scores = []
    for threshold in IOU_THRESHOLDS:
        # Кораблі не перекриваються, тож при IoU > 0.5 передбачення може збігтися щонайбільше
        # з одним кораблем — окремий алгоритм зіставлення не потрібен.
        matched = ious > threshold
        tp = matched.any(axis=1).sum()
        fn = len(true_masks) - tp
        fp = len(pred_masks) - tp
        scores.append(5 * tp / (5 * tp + 4 * fn + fp))
    return np.mean(scores)


def masks_by_image(df):
    ships = df.dropna(subset=["EncodedPixels"])
    return {
        image_id: [rle_decode(rle) for rle in group["EncodedPixels"]]
        for image_id, group in ships.groupby("ImageId")
    }


def f2_dataset(true_df, pred_df):
    true_masks = masks_by_image(true_df)
    pred_masks = masks_by_image(pred_df)
    scores = [
        f2_image(true_masks.get(image_id, []), pred_masks.get(image_id, []))
        for image_id in true_df["ImageId"].unique()
    ]
    return np.mean(scores)


if __name__ == "__main__":

    def strip(first, last, length=20):
        # Пікселі нумеруються з 1 і включно з last — як у прикладах, порахованих вручну.
        mask = np.zeros((1, length), dtype=np.uint8)
        mask[0, first - 1:last] = 1
        return mask

    def check(name, result, expected):
        assert np.isclose(result, expected), f"{name}: got {result}, expected {expected}"
        print(f"OK  {name}: {result:.4f}")

    check("IoU, one-pixel shift", iou(strip(3, 6), strip(4, 8)), 0.5)
    check("no ships, no predictions", f2_image([], []), 1.0)
    check("no ships, one prediction", f2_image([], [strip(2, 5)]), 0.0)
    check("one ship, no predictions", f2_image([strip(2, 5)], []), 0.0)
    check("one ship, perfect prediction", f2_image([strip(2, 5)], [strip(2, 5)]), 1.0)
    check("one ship, IoU 0.72", f2_image([strip(1, 18, 30)], [strip(1, 25, 30)]), 0.5)

    ship_a, ship_b = strip(2, 5), strip(11, 16)
    check(
        "two ships, three predictions",
        f2_image([ship_a, ship_b], [strip(2, 5), strip(12, 17), strip(8, 9)]),
        7.5 / 11,
    )
    check("two ships merged into one prediction", f2_image([ship_a, ship_b], [strip(2, 16)]), 0.0)

    data_dir = Path(__file__).parent.parent / "data"
    sample_ids = pd.read_csv(data_dir / "sample_ids.csv")["ImageId"]
    all_masks = pd.read_csv(data_dir / "train_ship_segmentations_v2.csv")
    sample_true = all_masks[all_masks["ImageId"].isin(sample_ids)]
    no_ships = pd.DataFrame({"ImageId": sample_ids, "EncodedPixels": np.nan})

    check("train_micro: truth as prediction", f2_dataset(sample_true, sample_true), 1.0)
    check("train_micro: 'no ships anywhere'", f2_dataset(sample_true, no_ships), 0.5)
