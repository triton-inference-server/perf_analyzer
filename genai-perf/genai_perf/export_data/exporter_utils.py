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

import logging
import textwrap
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def format_metric_name(
    name: str, unit: Optional[str], width: Optional[int] = None
) -> str:
    """
    Formats a metric name into a human-readable string with an optional unit.

    Args:
        name: The raw metric name with underscores.
        unit: The unit of the metric (e.g., 'ms').
        width: The maximum width of the metric name.

    Returns:
        The formatted metric name with the unit if provided.
    """
    metric_str = name.replace("_", " ").title()
    metric_str = f"{metric_str} ({unit})" if unit else metric_str

    # Wrap the string if it's longer than the provided width.
    if width and len(metric_str) > width:
        metric_str = textwrap.fill(metric_str, width=width)
    return metric_str


def format_stat_value(value: Any) -> str:
    """
    Formats a statistic value for human-readable output.

    Args:
        value: The value to format. Supports int and float types.

    Returns:
        The formatted value as a string. If not a number, returns the string representation.
    """
    return f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)


def fetch_stat(
    stats: Dict[str, Dict[str, float]],
    metric_name: str,
    stat: str,
) -> str:
    """
    Fetches a statistic value for a metric.
    Logs warnings for missing metrics or stats and returns 'N/A' if the value is missing.

    Args:
        stats: Dictionary containing statistics for metrics.
        metric_name: The name of the metric.
        stat: The statistic to fetch (e.g., 'avg', 'min', 'max').

    Returns:
        The formatted statistic value or 'N/A' if missing.
    """
    if metric_name not in stats:
        logger.error(f"Metric '{metric_name}' is missing in the provided statistics.")
        return "N/A"

    metric_stats = stats[metric_name]
    if not isinstance(metric_stats, dict):
        logger.error(
            f"Expected statistics for metric '{metric_name}' to be a dictionary. Got: {type(metric_stats).__name__}."
        )
        return "N/A"

    if stat not in metric_stats:
        logger.error(
            f"Statistic '{stat}' for metric '{metric_name}' is missing. "
            f"Available stats: {list(metric_stats.keys())}."
        )
        return "N/A"

    return format_stat_value(metric_stats[stat])
