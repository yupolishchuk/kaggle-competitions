from pathlib import Path

import numpy as np
import pandas as pd

from metric import f2_dataset

DATA_DIR = Path(__file__).parent.parent / "data"


def empty_predictions(image_ids):
    return pd.DataFrame({"ImageId": image_ids, "EncodedPixels": np.nan})


if __name__ == "__main__":
    masks = pd.read_csv(DATA_DIR / "train_ship_segmentations_v2.csv")
    split = pd.read_csv(DATA_DIR / "split.csv")
    val = split[split["split"] == "val"]

    val_true = masks[masks["ImageId"].isin(val["ImageId"])]
    val_score = f2_dataset(val_true, empty_predictions(val["ImageId"]))
    empty_share = (val["n_ships"] == 0).mean()
    print(f"Val score:                 {val_score:.4f}")
    print(f"Частка порожніх знімків у val: {empty_share:.4f}")
    assert np.isclose(val_score, empty_share), "скор порожнього сабміту має дорівнювати частці порожніх знімків"

    test_ids = pd.read_csv(DATA_DIR / "sample_submission_v2.csv")["ImageId"]
    submission = empty_predictions(test_ids)
    submission.to_csv(DATA_DIR / "submission_empty.csv", index=False)
    print(f"Збережено {DATA_DIR / 'submission_empty.csv'}: {len(submission)} знімків")
