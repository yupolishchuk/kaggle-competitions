from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "airbus-ship-detection"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "train_micro"
SEED = 42

GROUP_SIZES = {"0 ships": 150, "1 ship": 75, "2+ ships": 75}


def ship_group(n_ships):
    if n_ships == 0:
        return "0 ships"
    if n_ships == 1:
        return "1 ship"
    return "2+ ships"


def main():
    df = pd.read_csv(DATA_DIR / "train_ship_segmentations_v2.csv")
    ships_per_image = df.groupby("ImageId")["EncodedPixels"].count().rename("n_ships").reset_index()
    ships_per_image["group"] = ships_per_image["n_ships"].map(ship_group)

    sample = pd.concat(
        ships_per_image[ships_per_image["group"] == group].sample(n=size, random_state=SEED)
        for group, size in GROUP_SIZES.items()
    )

    api = KaggleApi()
    api.authenticate()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for i, image_id in enumerate(sample["ImageId"], start=1):
        # Вже завантажені файли пропускаються, тож після обриву мережі скрипт можна безпечно запустити знову.
        if (IMAGES_DIR / image_id).exists():
            continue
        api.competition_download_file(COMPETITION, f"train_v2/{image_id}", path=IMAGES_DIR, quiet=True)
        if i % 25 == 0:
            print(f"{i}/{len(sample)}")

    sample.to_csv(DATA_DIR / "sample_ids.csv", index=False)
    print(sample["group"].value_counts())
    print("images on disk:", len(list(IMAGES_DIR.glob("*.jpg"))))


if __name__ == "__main__":
    main()
