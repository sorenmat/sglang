"""CPU packed QSA index experiment. No GPU unpacking or timing is implemented.

Rows store offset-binary quantized codes, least-significant bit first, padded
to whole bytes. FP32 scale and outward-rounded residual norm cost 8 B/row.
payload_nbytes reports stored NumPy array bytes, excluding Python metadata,
temporary encoding/decoding buffers and allocator overhead.
"""

import numpy as np


def encode(keys, bits):
    """Encode real keys[N,D] with symmetric 4-, 6- or 8-bit quantization."""
    if type(bits) is not int or bits not in (4, 6, 8):
        raise ValueError("bits must be 4, 6 or 8")
    keys = np.asarray(keys)
    if (keys.ndim != 2 or not keys.size or not np.issubdtype(keys.dtype, np.number)
            or np.issubdtype(keys.dtype, np.complexfloating)):
        raise ValueError("keys must be a nonempty real numeric [N,D] array")
    keys = keys.astype(np.float64)
    if not np.isfinite(keys).all():
        raise ValueError("keys must be finite")
    maximum = 2 ** (bits - 1) - 1
    with np.errstate(over="ignore", under="ignore"):
        scale = np.max(np.abs(keys), axis=1) / maximum
        scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("key range cannot be represented with positive finite FP32 scales")
    quantized = np.rint(keys / scale[:, None]).clip(-maximum, maximum)
    coarse = quantized * scale[:, None]
    with np.errstate(over="ignore", under="ignore"):
        radius = np.linalg.norm(keys - coarse, axis=1)
        radius = np.nextafter(radius.astype(np.float32), np.float32(np.inf))
    if not np.isfinite(radius).all():
        raise ValueError("residual norm cannot be represented with finite FP32 radii")
    codes = (quantized + maximum).astype(np.uint8)
    binary = ((codes[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1)
    packed = np.packbits(binary.reshape(len(keys), -1), axis=1, bitorder="little")
    return {"packed": packed, "scale": scale, "radius": radius,
            "bits": bits, "dimension": keys.shape[1]}


def _validate(index):
    if not isinstance(index, dict):
        raise ValueError("index must be a dictionary")
    bits, dimension = index.get("bits"), index.get("dimension")
    if type(bits) is not int or bits not in (4, 6, 8):
        raise ValueError("index bits must be 4, 6 or 8")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("index dimension must be a positive integer")
    packed, scale, radius = (index.get(name) for name in ("packed", "scale", "radius"))
    if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint8
            or packed.ndim != 2 or len(packed) == 0
            or packed.shape[1] != (dimension * bits + 7) // 8):
        raise ValueError("packed must be uint8 [N,ceil(D*bits/8)] with N>0")
    for name, array in (("scale", scale), ("radius", radius)):
        if (not isinstance(array, np.ndarray) or array.dtype != np.float32
                or array.shape != (len(packed),) or not np.isfinite(array).all()):
            raise ValueError(f"{name} must be finite FP32 [N]")
    if np.any(scale <= 0) or np.any(radius < 0):
        raise ValueError("scales must be positive and radii nonnegative")
    used = (dimension * bits) % 8
    if used and np.any(packed[:, -1] >> used):
        raise ValueError("nonzero row-padding bits")
    return bits, dimension, packed, scale, radius


def decode(index):
    """Decode coarse[N,D] in float64, using the stored rounded FP32 scale."""
    bits, dimension, packed, scale, _ = _validate(index)
    binary = np.unpackbits(packed, axis=1, count=dimension * bits, bitorder="little")
    codes = (binary.reshape(len(packed), dimension, bits)
             << np.arange(bits, dtype=np.uint8)).sum(axis=2, dtype=np.uint16)
    maximum = 2 ** (bits - 1) - 1
    if np.any(codes > 2 * maximum):
        raise ValueError("reserved all-ones quantized code")
    return (codes.astype(np.float64) - maximum) * scale[:, None]


def payload_nbytes(index):
    """Actual array payload bytes, including per-row byte padding and radii."""
    _, _, packed, scale, radius = _validate(index)
    return packed.nbytes + scale.nbytes + radius.nbytes
