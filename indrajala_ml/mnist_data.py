import struct
import zlib

import numpy as np

IMAGE_SIZE = 28
RECORD_SIZE = IMAGE_SIZE * IMAGE_SIZE + 1  # IMAGE_SIZE*IMAGE_SIZE pixel bytes + 1 label byte


def _decode_grayscale_png(data: bytes) -> list[int]:
    """
    Decodes the one PNG variant the MNIST parquet files contain (8-bit grayscale, no interlacing, no
    palette) with zlib/struct only. Returns IMAGE_SIZE*IMAGE_SIZE pixel values (0-255), row-major.
    """

    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"

    pos = 8
    idat = b""
    width = height = bit_depth = color_type = None

    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 8 + length + 4  # skip the trailing CRC

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, _interlace = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat += chunk
        elif chunk_type == b"IEND":
            break

    assert width is not None and height is not None, "no IHDR chunk"
    assert bit_depth == 8 and color_type == 0, (
        f"expected 8-bit grayscale; got bit_depth={bit_depth}, color_type={color_type}"
    )

    raw = zlib.decompress(idat)
    stride = width
    pixels: list[int] = []
    previous_row = bytearray(stride)
    pos = 0

    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        row = bytearray(raw[pos : pos + stride])
        pos += stride

        for x in range(stride):
            a = row[x - 1] if x > 0 else 0
            b = previous_row[x]
            c = previous_row[x - 1] if x > 0 else 0

            if filter_type == 1:
                row[x] = (row[x] + a) & 0xFF
            elif filter_type == 2:
                row[x] = (row[x] + b) & 0xFF
            elif filter_type == 3:
                row[x] = (row[x] + (a + b) // 2) & 0xFF
            elif filter_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                row[x] = (row[x] + predictor) & 0xFF
            # filter_type == 0 (None): no change

        pixels.extend(row)
        previous_row = row

    return pixels


def convert_parquet_to_binary(parquet_path: str, binary_path: str, limit: int | None = None) -> None:
    """
    Converts an MNIST parquet file (an `image` struct column of PNG bytes, a `label` int64 column)
    once to a flat binary file: per example, IMAGE_SIZE*IMAGE_SIZE pixel bytes then 1 label byte,
    no header.

    pyarrow is imported inside this function so that importing the module to load data never pulls
    it in: its import footprint, inherited by every forked worker, is several times the training
    data's. It belongs to this offline step only, as scikit-learn does to digits.csv's extraction.
    """

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.slice(0, limit).to_pylist() if limit is not None else table.to_pylist()

    with open(binary_path, "wb") as f:
        for row in rows:
            pixels = _decode_grayscale_png(row["image"]["bytes"])
            assert len(pixels) == IMAGE_SIZE * IMAGE_SIZE, (
                f"expected a {IMAGE_SIZE}x{IMAGE_SIZE} image; got {len(pixels)} pixels"
            )
            f.write(bytes(pixels))
            f.write(bytes([row["label"]]))


def _read_binary_records(path: str, limit: int | None = None) -> bytes:
    """
    Up to limit records (all when None) from the start of the file, checked to be whole records.
    load_mnist_records_at_indices seeks per index instead.
    """

    with open(path, "rb") as f:
        data = f.read(limit * RECORD_SIZE) if limit is not None else f.read()

    assert len(data) % RECORD_SIZE == 0, (
        f"file size is not a multiple of RECORD_SIZE ({RECORD_SIZE}); got {len(data)} bytes"
    )
    return data


def load_mnist_dataset(path: str, limit: int | None = None) -> list[tuple[tuple[float, ...], int]]:
    """
    Loads convert_parquet_to_binary's format, pixels divided by 255 into [0.0, 1.0], as
    (state tuple, label) pairs. limit reads only the first limit records' bytes; the full files
    are 60000/10000 records.
    """

    data = _read_binary_records(path, limit)

    # one float object per pixel value, shared by every row: 60000 rows of freshly boxed floats
    # are about 1.9 GB, too much for several sweep workers at once. The values are unchanged.
    pixel_values = [pixel / 255.0 for pixel in range(256)]

    dataset: list[tuple[tuple[float, ...], int]] = []
    for offset in range(0, len(data), RECORD_SIZE):
        record = data[offset : offset + RECORD_SIZE]
        state = tuple(pixel_values[pixel] for pixel in record[:-1])
        label = record[-1]
        dataset.append((state, label))

    return dataset


def load_mnist_dataset_as_array(path: str, limit: int | None = None) -> np.ndarray:
    """
    load_mnist_dataset's pixels as one (n, 784) array in [0.0, 1.0], via np.frombuffer, instead of
    47 million boxed floats. Pixels only: load_mnist_labels gives the labels.
    """

    data = _read_binary_records(path, limit)

    record_count = len(data) // RECORD_SIZE
    records = np.frombuffer(data, dtype=np.uint8).reshape(record_count, RECORD_SIZE)
    return records[:, :-1].astype(np.float64) / 255.0


def load_mnist_labels(path: str) -> list[int]:
    """
    Only each record's label byte, for deciding which examples each class draws without decoding
    pixels (see load_mnist_records_at_indices). Decoded as tuples, all 60000 examples cost several
    GB once copied into a worker.
    """

    data = _read_binary_records(path)

    return [data[offset + IMAGE_SIZE * IMAGE_SIZE] for offset in range(0, len(data), RECORD_SIZE)]


def load_mnist_records_at_indices(path: str, indices: list[int]) -> list[tuple[tuple[float, ...], int]]:
    """
    Decodes only the records at indices, by seeking, so a worker builds its own class-balanced slice
    without any process holding the decoded dataset. Returns them in the order of indices.
    """

    dataset: list[tuple[tuple[float, ...], int]] = []
    with open(path, "rb") as f:
        for index in indices:
            f.seek(index * RECORD_SIZE)
            record = f.read(RECORD_SIZE)
            state = tuple(pixel / 255.0 for pixel in record[:-1])
            dataset.append((state, record[-1]))

    return dataset
