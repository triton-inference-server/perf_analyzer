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
import logging.config
import os
from typing import Optional

from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

DEFAULT_LOG_FORMAT = "%(message)s"
DEFAULT_DATE_FORMAT = "[%Y-%m-%d %H:%M:%S]"
HANDLER = "rich.logging.RichHandler"
MOCK_HANDLER = "genai_perf.logging.MockRichHandler"


class MockRichHandler(RichHandler):
    """A mock RichHandler that prints the raw message for testing purposes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        # In test mode, just print the raw message
        message = self.format(record)
        print(message)


def init_logging(log_level: Optional[str] = None) -> None:
    """Initialize logging configuration for the genai_perf package.

    Args:
        log_level: Override default log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """
    # Use environment variable or passed parameter to override default log level
    log_level = log_level or os.environ.get("GENAI_PERF_LOG_LEVEL", "DEBUG")

    # Check if we are running pytest
    is_testing = os.environ.get("PYTEST_CURRENT_TEST") is not None

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": DEFAULT_LOG_FORMAT,
                "datefmt": DEFAULT_DATE_FORMAT,
            },
        },
        "handlers": {
            "console": {
                "class": MOCK_HANDLER if is_testing else HANDLER,
                "formatter": "standard",
                "highlighter": NullHighlighter(),
                "omit_repeated_times": False,
                "rich_tracebacks": True,
                "show_path": True,
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False,
            },
            "__main__": {  # if __name__ == '__main__'
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "genai_perf": {  # All modules in genai_perf package
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(logging_config)


def getLogger(name):
    """Get a logger with the specified name.

    Args:
        name (str): Name of the logger, typically __name__

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
