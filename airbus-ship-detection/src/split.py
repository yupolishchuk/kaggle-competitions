from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent.parent / "data"
VAL_SIZE = 0.1
SEED = 42


def ships_per_image(masks_df):
    # count() пропускає порожні EncodedPixels, тож знімок без кораблів дає 0, а не 1.
    n_ships = masks_df.groupby("ImageId")["EncodedPixels"].count()
    return n_ships.rename("n_ships").reset_index()


def ship_group(n_ships):
    # Рідкісні кількості об'єднані, щоб у val у кожній групі були сотні знімків.
    groups = pd.cut(n_ships, bins=[-1, 0, 1, 2, 4, 100], labels=["0", "1", "2", "3-4", "5+"])
    return groups.astype(str)


def make_split(masks_df, val_size=VAL_SIZE, seed=SEED):
    images = ships_per_image(masks_df)
    train_ids, val_ids = train_test_split(
        images["ImageId"],
        test_size=val_size,
        stratify=ship_group(images["n_ships"]),
        random_state=seed,
    )
    images["split"] = "train"
    images.loc[images["ImageId"].isin(val_ids), "split"] = "val"
    return images


if __name__ == "__main__":
    masks = pd.read_csv(DATA_DIR / "train_ship_segmentations_v2.csv")
    split = make_split(masks)

    shares = pd.crosstab(ship_group(split["n_ships"]), split["split"], normalize="columns") * 100
    print("Частка груп, %:")
    print(shares.round(2))
    print()
    print(split["split"].value_counts())
    print()

    n_images = masks["ImageId"].nunique()
    assert len(split) == n_images, "кількість знімків не збігається з CSV"
    assert split["ImageId"].is_unique, "знімок потрапив у split.csv двічі"
    val_share = (split["split"] == "val").mean()
    assert abs(val_share - VAL_SIZE) < 0.001, f"частка val {val_share:.4f}"
    max_diff = (shares["train"] - shares["val"]).abs().max()
    assert max_diff < 0.1, f"частки груп у train і val відрізняються на {max_diff:.3f} п.п."
    assert make_split(masks).equals(split), "повторний запуск з тим самим seed дав інший поділ"
    print("OK  усі перевірки пройдено")

    split.to_csv(DATA_DIR / "split.csv", index=False)
    print(f"Збережено {DATA_DIR / 'split.csv'}")
