import unittest

import numpy as np

from codec import decode, encode, payload_nbytes


def reference(keys, bits):
    # Independent original probe formula; do not call the packed codec here.
    keys = np.asarray(keys, dtype=np.float64)
    maximum = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(keys), axis=1) / maximum
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    coarse = np.rint(keys / scale[:, None]).clip(-maximum, maximum) * scale[:, None]
    radius = np.nextafter(np.linalg.norm(keys - coarse, axis=1).astype(np.float32), np.float32(np.inf))
    return coarse, radius


class CodecTest(unittest.TestCase):
    def test_roundtrip_and_real_storage(self):
        rng = np.random.default_rng(20260925)
        for bits in (4, 6, 8):
            for dimension in (1, 3, 7, 128):
                keys = rng.normal(size=(5, dimension))
                keys[0] = 0
                keys[1] = 16
                keys[2] = -16
                index = encode(keys, bits)
                coarse = decode(index)
                expected, radius = reference(keys, bits)
                np.testing.assert_array_equal(coarse, expected)
                np.testing.assert_array_equal(index["radius"], radius)
                self.assertTrue(np.all(np.linalg.norm(keys - coarse, axis=1) <= index["radius"]))
                self.assertEqual(payload_nbytes(index), len(keys) * ((dimension * bits + 7) // 8 + 8))
                if dimension == 128:
                    self.assertEqual(payload_nbytes(index) // len(keys), {4: 72, 6: 104, 8: 136}[bits])

    def test_invalid_input_and_corrupt_metadata(self):
        for keys in ([], [[np.nan]], [[np.inf]], [[1j]], [["text"]], np.ones((1, 0))):
            with self.subTest(keys=repr(keys)), self.assertRaises(ValueError):
                encode(keys, 6)
        for bits in (True, 5, 6.0):
            with self.assertRaises(ValueError):
                encode([[1]], bits)
        index = encode([[0, 1, -1]], 6)
        bad_values = {
            "bits": 7, "dimension": 0, "packed": np.zeros((1, 2), dtype=np.uint8),
            "scale": np.array([0], dtype=np.float32), "radius": np.array([-1], dtype=np.float32),
        }
        for field, value in bad_values.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                decode(index | {field: value})
        padded = index["packed"].copy()
        padded[:, -1] |= 0x80
        with self.assertRaises(ValueError):
            decode(index | {"packed": padded})
        reserved = encode([[0, 0, 0, 0]], 6)
        reserved["packed"][:] = 0xFF
        with self.assertRaises(ValueError):
            decode(reserved)


if __name__ == "__main__":
    unittest.main()
