# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""XLA / TPU detection helpers.

Centralises ``torch_xla`` availability checks so the rest of the codebase
can branch on TPU support without scattered try/except blocks.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def is_torch_xla_available() -> bool:
    """Return ``True`` when ``torch_xla`` can be imported.

    The result is cached after the first call so repeated checks are free.
    """
    try:
        import torch_xla  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def is_xla_device(device) -> bool:
    """Return ``True`` when *device* is an XLA device.

    Accepts ``torch.device``, string, or any object with a ``.type`` attribute.
    Returns ``False`` without raising when ``torch_xla`` is not installed.
    """
    if device is None:
        return False
    device_type = getattr(device, "type", str(device)).lower()
    return device_type == "xla"
