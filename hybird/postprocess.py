import numpy as np
from PIL import Image


def paste_known_region(original, generated, mask):
    original = original.convert("RGB")
    generated = generated.convert("RGB")
    mask = mask.convert("L")

    original_np = np.asarray(original).astype(np.float32) / 255.0
    generated_np = np.asarray(generated).astype(np.float32) / 255.0
    mask_np = np.asarray(mask).astype(np.float32) / 255.0

    # 原项目 mask 白色通常表示保留区域，黑色表示缺失区域。
    known = mask_np[..., None]
    out = original_np * known + generated_np * (1.0 - known)

    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))
