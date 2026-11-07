from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rle import IMAGE_SHAPE, rle_decode

IMAGE_SIZE = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_image(path, size=IMAGE_SIZE):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    # cv2 читає кольори як BGR, а pretrained-моделі вчились на RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # INTER_AREA усереднює пікселі при зменшенні — дрібні кораблі не «зникають між пікселями».
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def augment(image, mask=None):
    # Випадковість з torch, а не numpy: DataLoader дає кожному процесу (worker) свій seed
    # для torch, тож процеси не повторюють однакові аугментації.
    if torch.rand(1).item() < 0.5:
        image = np.flip(image, axis=1)
        mask = None if mask is None else np.flip(mask, axis=1)
    if torch.rand(1).item() < 0.5:
        image = np.flip(image, axis=0)
        mask = None if mask is None else np.flip(mask, axis=0)
    k = torch.randint(0, 4, (1,)).item()
    image = np.rot90(image, k)
    mask = None if mask is None else np.rot90(mask, k)
    return image, mask


def to_tensor(image):
    image = (image.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    # [H, W, C] → [C, H, W]; ascontiguousarray — бо після flip/rot90 numpy лише «перевертає»
    # індекси без копіювання, а torch.from_numpy такі масиви не приймає.
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image)


class ShipClassificationDataset(Dataset):
    def __init__(self, image_ids, labels, image_dir, train):
        self.image_ids = list(image_ids)
        self.labels = list(labels)
        self.image_dir = Path(image_dir)
        self.train = train

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image = read_image(self.image_dir / self.image_ids[i])
        if self.train:
            image, _ = augment(image)
        label = torch.tensor(float(self.labels[i]))
        return to_tensor(image), label


def rles_by_image(masks_df):
    ships = masks_df.dropna(subset=["EncodedPixels"])
    return ships.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()


def read_mask(rles, size=IMAGE_SIZE):
    mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    for rle in rles:
        mask |= rle_decode(rle)
    # INTER_NEAREST, а не INTER_AREA: маска має лишитись з 0 і 1, без проміжних значень на межах.
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


class ShipSegmentationDataset(Dataset):
    def __init__(self, image_ids, rles, image_dir, train, size=IMAGE_SIZE):
        self.image_ids = list(image_ids)
        self.rles = rles
        self.image_dir = Path(image_dir)
        self.train = train
        self.size = size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        image = read_image(self.image_dir / image_id, self.size)
        mask = read_mask(self.rles.get(image_id, []), self.size)
        if self.train:
            image, mask = augment(image, mask)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return to_tensor(image), mask


CROP_SIZE = 256


def random_window_start(start, length, crop_size, total):
    # Вікно [s, s + crop_size) має повністю містити об'єкт [start, start + length) і не виходити за знімок.
    low = max(0, start + length - crop_size)
    high = min(start, total - crop_size)
    if low > high:
        # Об'єкт довший за вікно — вимагаємо лише, щоб вікно містило його центр.
        center = start + length // 2
        low = max(0, center - crop_size + 1)
        high = min(center, total - crop_size)
    return torch.randint(low, high + 1, (1,)).item()


def ship_crop(image, mask, crop_size=CROP_SIZE, ship_prob=0.75):
    height, width = mask.shape
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1 and torch.rand(1).item() < ship_prob:
        # Компонента — рівномірно, а не через випадковий піксель маски: інакше великі кораблі
        # траплялися б частіше, а слабке місце моделі — дрібні.
        k = torch.randint(1, n_labels, (1,)).item()
        x, y, w, h = stats[k, :4]
        top = random_window_start(y, h, crop_size, height)
        left = random_window_start(x, w, crop_size, width)
    else:
        top = torch.randint(0, height - crop_size + 1, (1,)).item()
        left = torch.randint(0, width - crop_size + 1, (1,)).item()
    window = (slice(top, top + crop_size), slice(left, left + crop_size))
    return image[window], mask[window]


class ShipCropDataset(Dataset):
    def __init__(self, image_ids, rles, image_dir, crop_size=CROP_SIZE, ship_prob=0.75):
        self.image_ids = list(image_ids)
        self.rles = rles
        self.image_dir = Path(image_dir)
        self.crop_size = crop_size
        self.ship_prob = ship_prob

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        image = read_image(self.image_dir / image_id, IMAGE_SHAPE[0])
        mask = read_mask(self.rles.get(image_id, []), IMAGE_SHAPE[0])
        image, mask = ship_crop(image, mask, self.crop_size, self.ship_prob)
        image, mask = augment(image, mask)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return to_tensor(image), mask


def balanced_sample(split_df, seed):
    with_ships = split_df[split_df["n_ships"] > 0]
    empty = split_df[split_df["n_ships"] == 0].sample(n=len(with_ships), random_state=seed)
    return pd.concat([with_ships, empty]).sample(frac=1, random_state=seed)


if __name__ == "__main__":
    import time

    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    torch.manual_seed(42)

    # Корабель довший за вікно: вікно має містити його центр.
    mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    mask[100:110, 50:450] = 1
    for _ in range(100):
        _, crop = ship_crop(mask, mask, ship_prob=1.0)
        assert crop.shape == (CROP_SIZE, CROP_SIZE) and crop.sum() > 0
    print("корабель довший за вікно: OK")

    # Великий і дрібний корабель далеко один від одного: в одне вікно не влазять,
    # тож дрібний має траплятися приблизно в половині кропів.
    mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    mask[0:60, 0:60] = 1
    mask[700:704, 700:704] = 1
    small = np.mean([ship_crop(mask, mask, ship_prob=1.0)[1].sum() == 16 for _ in range(1000)])
    print(f"частка кропів з дрібним кораблем: {small:.2f} (очікується ~0.5)")
    assert 0.4 < small < 0.6

    masks = pd.read_csv(DATA_DIR / "train_ship_segmentations_v2.csv")
    rles = rles_by_image(masks)
    image_dir = DATA_DIR / "train_micro"
    ship_ids = sorted(p.name for p in image_dir.iterdir() if p.name in rles)

    dataset = ShipCropDataset(ship_ids, rles, image_dir)
    image, crop_mask = dataset[0]
    assert image.shape == (3, CROP_SIZE, CROP_SIZE) and crop_mask.shape == (1, CROP_SIZE, CROP_SIZE)
    print(f"ShipCropDataset: знімок {tuple(image.shape)}, маска {tuple(crop_mask.shape)}")

    for ship_prob in [1.0, 0.75, 0.0]:
        dataset = ShipCropDataset(ship_ids, rles, image_dir, ship_prob=ship_prob)
        with_ship = np.mean([dataset[i][1].sum() > 0 for i in range(len(dataset)) for _ in range(3)])
        print(f"ship_prob {ship_prob}: кропів з кораблем {with_ship:.0%}")
        if ship_prob == 1.0:
            assert with_ship == 1.0

    # Де в кропі опиняється корабель: центр маски по горизонталі, 4 смуги по 64 пікселі.
    positions = []
    for image_id in ship_ids:
        full_mask = read_mask(rles[image_id], IMAGE_SHAPE[0])
        for _ in range(5):
            _, crop = ship_crop(full_mask, full_mask, ship_prob=1.0)
            positions.append(np.nonzero(crop)[1].mean())
    counts = np.histogram(positions, bins=4, range=(0, CROP_SIZE))[0]
    print("центр корабля по горизонталі, смуги 0–64 / 64–128 / 128–192 / 192–256:", counts.tolist())
    assert counts.min() > 0.1 * counts.sum()

    for name, dataset in [("кроп 256 з 768", ShipCropDataset(ship_ids, rles, image_dir)),
                          ("ціле 384", ShipSegmentationDataset(ship_ids, rles, image_dir, train=True))]:
        start = time.time()
        for i in range(len(dataset)):
            dataset[i]
        print(f"{name}: {(time.time() - start) / len(dataset) * 1000:.1f} мс на знімок")
