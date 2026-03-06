import io

import numpy as np
from PIL import Image


def decode_image_to_chw_fp32(raw_image: bytes) -> np.ndarray:
    """Decode image bytes into CHW float32 tensor normalized to [0, 1]."""
    with Image.open(io.BytesIO(raw_image)) as img:
        rgb = img.convert("RGB")
        hwc = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(hwc, (2, 0, 1))


def encode_chw_fp32_to_png_bytes(chw: np.ndarray) -> tuple[bytes, int, int]:
    """Encode CHW float32 tensor in [0, 1] as PNG bytes."""
    clipped = np.clip(chw, 0.0, 1.0)
    hwc = np.transpose(clipped, (1, 2, 0))
    uint8_img = (hwc * 255.0).round().astype(np.uint8)

    image = Image.fromarray(uint8_img, mode="RGB")
    width, height = image.size

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue(), width, height
