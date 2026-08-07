from __future__ import annotations

import io

from PIL import Image, ImageOps

DEFAULT_MAX_SIZE = 1280
DEFAULT_QUALITY = 85
MIN_COMPRESS_BYTES = 1024 * 1024


def prepare_image_bytes(
    data: bytes,
    *,
    max_size: int = DEFAULT_MAX_SIZE,
    quality: int = DEFAULT_QUALITY,
) -> bytes:
    """Resize large images and encode them as JPEG for vision requests."""
    if not data:
        return data

    max_size = max(1, int(max_size))
    quality = min(100, max(1, int(quality)))
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            needs_resize = max(width, height) > max_size
            if not needs_resize and len(data) < MIN_COMPRESS_BYTES:
                return data

            if needs_resize:
                scale = max_size / max(width, height)
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )

            if image.mode not in ("RGB", "L"):
                if "A" in image.getbands():
                    background = Image.new("RGB", image.size, "white")
                    background.paste(image, mask=image.getchannel("A"))
                    image = background
                else:
                    image = image.convert("RGB")

            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            result = output.getvalue()
            if not result:
                return data
            if not needs_resize and len(result) >= len(data):
                return data
            return result
    except Exception:
        return data
