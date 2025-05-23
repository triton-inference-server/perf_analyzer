# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Union

from genai_perf.exceptions import GenAIPerfException
from genai_perf.types import RecordValue


@dataclass(frozen=True)
class ReductionFactor:
    NS_TO_MS = 6
    NJ_TO_MJ = 6
    MIB_TO_GB: float = 1.048576e-3
    PERCENTAGE = -2
    NONE = 0
    BYTES_TO_KB: float = 1 / 1024


class RecordType(ABCMeta):
    """
    A metaclass that holds the instantiated Record types
    """

    record_types: Dict[str, "RecordType"] = {}

    def __new__(cls, name, base, namespace):
        """
        This function is called upon declaration of any classes of type
        RecordType
        """

        record_type = super().__new__(cls, name, base, namespace)

        # If record_type.tag is a string, register it here
        if isinstance(record_type.tag, str):
            cls.record_types[record_type.tag] = record_type
        return record_type

    @classmethod
    def get(cls, tag: str) -> "RecordType":
        """
        Parameters
        ----------
        tag : str
            tag that a record type has registered it classname with

        Returns
        -------
        The class of type RecordType corresponding to the tag
        """

        if tag not in cls.record_types:
            try:
                importlib.import_module("genai_perf.record.types.%s" % tag)
            except ImportError as e:
                print(e)
        return cls.record_types[tag]

    @classmethod
    def get_all_record_types(cls) -> Dict[str, "RecordType"]:
        """
        Returns
        -------
        dict
            keys are tags and values are
            all the types that have this as a
            metaclass
        """

        type_module_directory = os.path.join(
            globals()["__spec__"].origin.rsplit("/", 1)[0], "types"
        )
        for filename in os.listdir(type_module_directory):
            if filename != "__init__.py" and filename.endswith(".py"):
                try:
                    importlib.import_module(f"genai_perf.record.types.{filename[:-3]}")
                except AttributeError:
                    raise GenAIPerfException("Error retrieving all record types")
        return cls.record_types


class Record(metaclass=RecordType):
    """
    This class is used for representing
    records
    """

    def __init__(self, value: RecordValue, timestamp: int):
        assert type(value) is float or type(value) is int
        assert type(timestamp) is int

        self._value = value
        self._timestamp = timestamp

    @staticmethod
    def aggregation_function():
        """
        The function that is used to aggregate
        this type of record

        Returns
        -------
        callable()
            [Records] -> Record
        """

        return lambda records: max(records, key=lambda r: r.value())

    @staticmethod
    def value_function() -> float:
        """
        Returns the average value from a list
        """
        return mean  # type: ignore

    @staticmethod
    @abstractmethod
    def header(aggregation_tag=False) -> str:
        """
        Parameters
        ----------
        aggregation_tag : boolean
            An optional tag that may be displayed as part of the header
            indicating that this record has been aggregated using max, min or
            average etc.

        Returns
        -------
        str
            The full name of the
            metric.
        """

    @property
    @abstractmethod
    def tag(self) -> str:
        """
        Returns
        -------
        str
            the name tag of the record type.
        """

    @property
    @abstractmethod
    def reduction_factor(self) -> float:
        """
        Returns
        -------
        float
            the reduction factor of the record type.
        """

    def create_checkpoint_object(self):
        return (self.tag, self.__dict__)

    @classmethod
    def create_class_from_checkpoint(cls, record_dict) -> "Record":
        record = cls(0, 0)
        for key in ["_value", "_timestamp"]:
            if key in record_dict:
                setattr(record, key, record_dict[key])
        return record

    def value(self) -> RecordValue:
        """
        This method returns the value of recorded metric

        Returns
        -------
        float
            value of the metric
        """

        return self._value

    def timestamp(self) -> int:
        """
        This method should return the time at which the record was created.

        Returns
        -------
        float
            timestamp passed in during
            record creation
        """

        return self._timestamp

    def __mul__(self, other) -> "Record":
        """
        Defines left multiplication for records with floats or ints.

        Returns
        -------
        Record
        """

        if isinstance(other, (int, float)):
            return self.__class__(value=(self.value() * other), timestamp=0)
        else:
            raise TypeError

    def __rmul__(self, other) -> "Record":
        """
        Defines right multiplication
        """

        return self.__mul__(other)

    def __truediv__(self, other) -> "Record":
        """
        Defines left multiplication for records with floats or ints

        Returns
        -------
        Record
        """

        if isinstance(other, (int, float)):
            return self.__class__(value=(self.value() / other), timestamp=0)

        else:
            raise TypeError

    @abstractmethod
    def _positive_is_better(self) -> bool:
        """
        Returns a bool indicating if a larger positive value is better
        for a given record type
        """

    def calculate_percentage_gain(self, other: "Record") -> float:
        """
        Calculates percentage gain between records
        """

        # When increasing values are better gain is based on the original value (other):
        # example: 200 vs. 100 is (200 - 100) / 100 = 100%
        # example: 100 vs. 200 is (100 - 200) / 200 = -50%
        if self._positive_is_better():
            return ((self.value() - other.value()) / other.value()) * 100

        # When decreasing values are better gain is based on the new value (self):
        # example: 100 vs. 200 is (200 - 100) / 100 = 100%
        # example: 200 vs. 100 is (100 - 200) / 200 = -50%
        else:
            return ((other.value() - self.value()) / self.value()) * 100

    def is_passing_constraint(self, constraint_value: Union[int | float]) -> bool:
        if self._positive_is_better():
            return self.value() > constraint_value
        else:
            return self.value() < constraint_value


class IncreasingRecord(Record):
    """
    Record where an increasing positive value is better
    """

    def _positive_is_better(self) -> bool:
        return True


class DecreasingRecord(Record):
    """
    Record where an increasing positive value is worse
    """

    def _positive_is_better(self) -> bool:
        return False
