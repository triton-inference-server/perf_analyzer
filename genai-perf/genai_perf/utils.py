# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import random
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import genai_perf.logging as logging
import orjson

# Skip type checking to avoid mypy error
# Issue: https://github.com/python/mypy/issues/10632
import yaml  # type: ignore
from PIL import Image

logger = logging.getLogger(__name__)


def encode_image(img: Image, format: str):
    """Encodes an image into base64 encoding."""
    # Lazy import for vision related endpoints
    import base64
    from io import BytesIO

    # JPEG does not support P or RGBA mode (commonly used for PNG) so it needs
    # to be converted to RGB before an image can be saved as JPEG format.
    if format == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")

    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def not_data_sse_field(msg: str) -> bool:
    # (TODO) TPA-829: Add more proper SSE event stream support
    # Check for empty, comment, or event SSE response
    return msg.startswith((":", "event:", "id:", "retry:"))


def sse_error_occurred(msg: str) -> bool:
    return msg.startswith("event:") and "error" in msg.lower()


def remove_sse_prefix(msg: str) -> str:
    prefix = "data:"
    if msg.startswith(prefix):
        return msg[len(prefix) :].strip()
    return msg.strip()


def load_yaml(filepath: Path) -> Dict[str, Any]:
    with open(str(filepath)) as f:
        configs = yaml.safe_load(f)
    return configs


def load_json(filepath: Path) -> Dict[str, Any]:
    with open(str(filepath), encoding="utf-8", errors="ignore") as f:
        content = f.read()
        return load_json_str(content)


def load_json_str(json_str: str, func: Callable = lambda x: x) -> Dict[str, Any]:
    """
    Deserializes JSON encoded string into Python object.

    Args:
      - json_str: string
          JSON encoded string
      - func: callable
          A function that takes deserialized JSON object. This can be used to
          run validation checks on the object. Defaults to identity function.
    """
    try:
        # Note: orjson may not parse JSON the same way as Python's standard json library,
        # notably being stricter on UTF-8 conformance.
        # Refer to https://github.com/ijl/orjson?tab=readme-ov-file#str for details.
        return func(orjson.loads(json_str))
    except orjson.JSONDecodeError:
        snippet = json_str[:200] + ("..." if len(json_str) > 200 else "")
        logger.error("Failed to parse JSON string: '%s'", snippet)
        raise


def remove_file(file: Path) -> None:
    if file.is_file():
        file.unlink()


def get_enum_names(enum: Type[Enum]) -> List[str]:
    names = []
    for e in enum:
        names.append(e.name.lower())
    return names


def scale(value, factor):
    return value * factor


def sample_bounded_normal(
    mean, stddev, lower=float("-inf"), upper=float("inf")
) -> float:
    """Bound random normal sampling to [lower, upper]. Set the final value to
    the boundary value if the value goes below or above the boundaries.
    """
    n = random.gauss(mean, stddev)
    return min(max(lower, n), upper)


def sample_bounded_normal_int(
    mean, stddev, lower=float("-inf"), upper=float("inf")
) -> int:
    return round(sample_bounded_normal(mean, stddev, lower, upper))


def is_power_of_two(n: int) -> bool:
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def split_and_strip_whitespace(input_string: str) -> List[str]:
    """
    Split a string by comma and strip whitespace from each item
    """
    return [item.strip() for item in input_string.split(",")]
