import struct
import zlib

import numpy as np

IMAGE_SIZE = 28
RECORD_SIZE = IMAGE_SIZE * IMAGE_SIZE + 1  # IMAGE_SIZE*IMAGE_SIZE pixel bytes + 1 label byte


def _decode_grayscale_png(data: bytes) -> list[int]:
    """
    A minimal, hand-rolled decoder for the one PNG variant the bundled MNIST parquet files
    actually contain (confirmed via inspection): 8-bit grayscale, no interlacing, no palette.
    Uses only the stdlib zlib/struct modules - parsing PNG's chunk structure and un-filtering
    its scanlines is a small, well-specified algorithm, unlike parsing parquet itself (see
    convert_parquet_to_binary's docstring).

    Returns a flat, row-major list of IMAGE_SIZE*IMAGE_SIZE pixel values (0-255).
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
            width, height, bit_depth, color_type, _compression, _filter, _interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif chunk_type == b"IDAT":
            idat += chunk
        elif chunk_type == b"IEND":
            break

    assert bit_depth == 8 and color_type == 0, f"expected 8-bit grayscale; got bit_depth={bit_depth}, color_type={color_type}"

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
    One-time conversion from a bundled MNIST parquet file (a HuggingFace-style export: an
    `image` struct column of PNG-encoded bytes, a `label` int64 column) to a flat,
    dependency-free binary format: IMAGE_SIZE*IMAGE_SIZE raw pixel bytes (0-255) followed by 1
    label byte, per example, back to back, no header.

    pyarrow is imported locally, inside this function, not at module level - so merely
    importing this module (to call load_mnist_dataset) never pulls pyarrow into a process that
    doesn't need it. That's not a style preference: it avoids a real multiprocessing
    memory-exhaustion failure - pyarrow's own import footprint, inherited by every forked worker
    process regardless of whether that worker ever touches it, is several times larger than the
    training data itself.
    Mirrors exactly how digits_data.py's bundled CSV was extracted once from scikit-learn
    without scikit-learn ever becoming a runtime dependency - pyarrow only belongs to this one
    offline conversion step, never to actual training.
    """

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.slice(0, limit).to_pylist() if limit is not None else table.to_pylist()

    with open(binary_path, "wb") as f:
        for row in rows:
            pixels = _decode_grayscale_png(row["image"]["bytes"])
            assert len(pixels) == IMAGE_SIZE * IMAGE_SIZE, f"expected a {IMAGE_SIZE}x{IMAGE_SIZE} image; got {len(pixels)} pixels"
            f.write(bytes(pixels))
            f.write(bytes([row["label"]]))


def _read_binary_records(path: str, limit: int | None = None) -> bytes:
    """
    Shared by load_mnist_dataset/load_mnist_dataset_as_array/load_mnist_labels: open the file,
    read up to limit*RECORD_SIZE bytes from the start (or the whole file when limit is None),
    and validate the result is a whole number of RECORD_SIZE-byte records - the same three-line
    sequence each of them repeated independently. load_mnist_records_at_indices is deliberately
    not unified here - it seeks per-index rather than doing one bulk read, a genuinely different
    access pattern, not just a copy of this one.
    """

    with open(path, "rb") as f:
        data = f.read(limit * RECORD_SIZE) if limit is not None else f.read()

    assert len(data) % RECORD_SIZE == 0, f"file size is not a multiple of RECORD_SIZE ({RECORD_SIZE}); got {len(data)} bytes"
    return data


def load_mnist_dataset(path: str, limit: int | None = None) -> list[tuple[tuple[float, ...], int]]:
    """
    Loads the lightweight binary format convert_parquet_to_binary produces - no pyarrow, no PNG
    decoding, just raw bytes read directly off disk and normalized to [0.0, 1.0] (divide by
    255) - the same shape/normalization convention digits_data.load_digits_dataset already
    established for the smaller bundled dataset.

    limit caps how many records are read (from the start of the file) - the real files are
    60000/10000 records, too many to load in every test run. Reads only the needed bytes
    directly (RECORD_SIZE * limit), not the whole file - a true partial read, not just a
    post-hoc slice of everything.
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
    The array-backed counterpart to load_mnist_dataset: decoding into `tuple[float, ...]` per
    example means 60000 x 784 = 47 million individually boxed Python float objects. Decoding raw
    uint8 pixel bytes directly
    via np.frombuffer avoids that entirely - one array, not millions of objects.

    Returns pixels only (shape (n, 784), normalized to [0.0, 1.0]), not labels - use
    load_mnist_labels for those, exactly as load_mnist_dataset's own callers already do when
    they only need labels. A new, additive function alongside load_mnist_dataset, not a
    replacement - nothing currently calling that function needs to change.
    """

    data = _read_binary_records(path, limit)

    record_count = len(data) // RECORD_SIZE
    records = np.frombuffer(data, dtype=np.uint8).reshape(record_count, RECORD_SIZE)
    return records[:, :-1].astype(np.float64) / 255.0


def load_mnist_labels(path: str) -> list[int]:
    """
    Reads only every record's label byte - no pixel decoding, no per-pixel float objects. Used
    to make class-balancing decisions (which examples to draw for which class) cheaply, without
    ever materializing the full dataset's pixel data as Python objects - see
    load_mnist_records_at_indices. Materializing all 60000 examples as decoded
    (tuple-of-784-floats, label) pairs, even just once in the main process, costs several GB
    once duplicated into a multiprocessing worker - not because of any particular library, but
    because 47 million individual boxed Python float objects is simply a lot of memory, however
    they got there.
    """

    data = _read_binary_records(path)

    return [data[offset + IMAGE_SIZE * IMAGE_SIZE] for offset in range(0, len(data), RECORD_SIZE)]


def load_mnist_records_at_indices(path: str, indices: list[int]) -> list[tuple[tuple[float, ...], int]]:
    """
    Reads and decodes only the specific records at indices, via direct seek - not the whole
    file. This is what lets a multiprocessing worker build just its own small, class-balanced
    slice of the data without any process (main or worker) ever holding the full dataset
    decoded in memory at once - see load_mnist_labels. Not necessarily called in index order -
    callers needing a specific order should sort/shuffle the result themselves.
    """

    dataset: list[tuple[tuple[float, ...], int]] = []
    with open(path, "rb") as f:
        for index in indices:
            f.seek(index * RECORD_SIZE)
            record = f.read(RECORD_SIZE)
            state = tuple(pixel / 255.0 for pixel in record[:-1])
            dataset.append((state, record[-1]))

    return dataset
