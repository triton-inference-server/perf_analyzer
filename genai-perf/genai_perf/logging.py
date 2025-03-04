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

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M"


def init_logging() -> None:
    LOGGING_CONFIG = {
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
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # Default is stderr
            },
        },
        "loggers": {
            "": {  # root logger - avoid using
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False,
            },
            "__main__": {  # if __name__ == '__main__'
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.parser": {  # must use module name for loggers
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.wrapper": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.plots.plot_config_parser": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.plots.plot_manager": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.export_data.json_exporter": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.export_data.csv_exporter": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.goodput_calculator.llm_goodput_calculator": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.subcommand.analyze": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.subcommand.common": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.inputs.retrievers.synthetic_prompt_generator": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.profile_data_parser": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)


def getLogger(name):
    return logging.getLogger(name)
