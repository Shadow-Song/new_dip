from pathlib import Path

from PIL import Image


try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


def resize_to_match(image_path, reference_path, output_path):
    image = Image.open(image_path).convert("RGB")
    reference = Image.open(reference_path).convert("RGB")
    resized = image.resize(reference.size, LANCZOS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)
    return output_path


def blend_images(base_path, enhanced_path, output_path, alpha=0.35):
    """Blend an enhanced image into a base image while preserving structure."""
    import numpy as np

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    base = Image.open(base_path).convert("RGB")
    enhanced = Image.open(enhanced_path).convert("RGB").resize(base.size, LANCZOS)

    base_np = np.asarray(base).astype("float32") / 255.0
    enhanced_np = np.asarray(enhanced).astype("float32") / 255.0
    blended_np = base_np * (1.0 - alpha) + enhanced_np * alpha

    output = Image.fromarray((np.clip(blended_np, 0, 1) * 255).astype("uint8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    return output_path


def make_comparison(output_path, image_paths, thumb_height=256):
    loaded = []
    for label, path in image_paths:
        path = Path(path)
        if path.exists():
            loaded.append((label, Image.open(path).convert("RGB")))

    if not loaded:
        return None

    thumb_height = min(thumb_height, max(image.height for _, image in loaded))
    gap = 8
    label_height = 28
    thumbs = []

    for label, image in loaded:
        width = max(1, int(image.width * thumb_height / image.height))
        thumbs.append((label, image.resize((width, thumb_height), LANCZOS)))

    width = sum(image.width for _, image in thumbs) + gap * (len(thumbs) - 1)
    height = thumb_height + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDrawOrNone(canvas)

    x = 0
    for label, image in thumbs:
        canvas.paste(image, (x, label_height))
        if draw is not None:
            draw.text((x + 4, 6), label, fill=(0, 0, 0))
        x += image.width + gap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def ImageDrawOrNone(canvas):
    try:
        from PIL import ImageDraw

        return ImageDraw.Draw(canvas)
    except Exception:
        return None
