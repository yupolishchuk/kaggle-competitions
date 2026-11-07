import numpy as np

IMAGE_SHAPE = (768, 768)


def rle_decode(encoded_pixels, shape=IMAGE_SHAPE):
    numbers = np.array(encoded_pixels.split(), dtype=int)
    # У RLE пікселі нумеруються з 1, а в numpy — з 0.
    starts = numbers[0::2] - 1
    lengths = numbers[1::2]

    flat_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = 1

    # RLE рахує пікселі спершу вниз по стовпцю, тому order='F'.
    return flat_mask.reshape(shape, order="F")


def rle_encode(mask):
    flat_mask = mask.flatten(order="F")
    # Нулі по краях гарантують, що кожна серія одиниць має і точку початку, і точку кінця.
    padded = np.concatenate([[0], flat_mask, [0]])
    change_points = np.where(padded[1:] != padded[:-1])[0] + 1
    starts = change_points[0::2]
    lengths = change_points[1::2] - starts
    return " ".join(f"{start} {length}" for start, length in zip(starts, lengths))
