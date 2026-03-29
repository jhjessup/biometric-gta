"""
EXIF Stripping Pre-Processor
Strips all EXIF/metadata from images and returns a clean copy + SHA-256 hash.
Supports JPEG, PNG, and HEIC/HEIF formats.
"""

import hashlib
import io
from pathlib import Path

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()


def strip_exif(input_path: str | Path) -> tuple[bytes, str]:
    """
    Load an image, strip all EXIF metadata, and return clean bytes + SHA-256 hash.

    Returns:
        (image_bytes, sha256_hex) — clean image data and its hash
    """
    input_path = Path(input_path)
    with Image.open(input_path) as img:
        # Convert to RGB to drop any embedded profile/transparency that carries metadata
        clean = img.convert(img.mode if img.mode in ("RGB", "L") else "RGB")

        buf = io.BytesIO()
        clean.save(buf, format="PNG", exif=b"")
        image_bytes = buf.getvalue()

    sha256 = hashlib.sha256(image_bytes).hexdigest()
    return image_bytes, sha256


def strip_exif_to_file(input_path: str | Path, output_path: str | Path) -> str:
    """
    Strip EXIF from input_path, write clean image to output_path.

    Returns:
        SHA-256 hex digest of the stripped image
    """
    image_bytes, sha256 = strip_exif(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return sha256


def verify_no_exif(image_path: str | Path) -> bool:
    """Return True if image has no EXIF data."""
    with Image.open(image_path) as img:
        exif = img.info.get("exif", b"")
        return len(exif) == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python exif_stripper.py <input_image> <output_image>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    sha = strip_exif_to_file(src, dst)
    clean = verify_no_exif(dst)
    print(f"Stripped: {src} -> {dst}")
    print(f"SHA-256:  {sha}")
    print(f"EXIF clean: {clean}")
