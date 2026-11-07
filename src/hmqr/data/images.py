"""Image loading for local datasets and occasional URL-hosted images."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

from ..utils import get_logger

LOG = get_logger("hmqr.images")


def _temporary_path(url: str) -> Path:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return Path(tempfile.gettempdir()) / "hmqr_url_images" / f"{h}.jpg"


def load_image(path_or_url: str) -> Optional[Image.Image]:
    """Return an RGB PIL image whose `.filename` is a real local path.

    All configured datasets resolve to local files. For compatibility with an
    annotation that still contains an HTTP URL, Qwen-VL's required local copy
    is placed in the operating system's temporary directory rather than under
    the persistent project data tree.
    """
    try:
        if str(path_or_url).startswith(("http://", "https://")):
            local = _temporary_path(path_or_url)
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                import requests

                r = requests.get(path_or_url, timeout=30)
                r.raise_for_status()
                local.write_bytes(r.content)
            img = Image.open(local).convert("RGB")
            img.filename = str(local)
            return img
        p = Path(path_or_url)
        if not p.exists():
            return None
        img = Image.open(p).convert("RGB")
        img.filename = str(p)
        return img
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not load image %s: %s", path_or_url, exc)
        return None
