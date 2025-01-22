# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ConfigField:
    """
    A class to represent a configuration field.
    Only the default value is required, all other values are optional.
    Bounds and choices are checked when the value is set.
    """

    def __init__(
        self,
        default: Any,
        required: bool = False,
        add_to_template: bool = True,
        template_comment: Optional[str] = None,
        value: Optional[Any] = None,
        bounds: Optional[Dict[str, Any]] = None,
        choices: Optional[Union[List[Any], Enum]] = None,
    ):
        self.default = default
        self.required = required
        self.add_to_template = add_to_template
        self.template_comment = template_comment
        self.bounds = bounds
        self.choices = choices

        # It is important that this is set last to ensure that anything used to
        # check value (like bounds/choices) is set before value is set
        self.value = value

        self._set_is_set_by_user()
        self._set_default_value()
        self._check_bounds()
        self._check_choices()

    def _set_is_set_by_user(self):
        self.is_set_by_user = not self.value is None

    def _set_default_value(self):
        if self.value is None:
            self.value = self.default

    def _check_bounds(self):
        if isinstance(self.value, (int, float)):
            if self.bounds and "upper" in self.bounds:
                if self.bounds["upper"] < self.value:
                    raise ValueError(
                        f"User Config: {self.value} exceeds upper bounds (f{self.bounds})"
                    )
            if self.bounds and "lower" in self.bounds:
                if self.bounds["lower"] > self.value:
                    raise ValueError(
                        f"User Config: {self.value} exceeds lower bounds (f{self.bounds})"
                    )

    def _check_choices(self):
        if not self.choices or not self.value:
            return

        value_list = (
            list(self.value.keys()) if type(self.value) is dict else [self.value]
        )

        if isinstance(self.choices, list):
            for value in value_list:
                if value not in self.choices:
                    raise ValueError(
                        f"User Config: {value} not in list of choices: f{self.choices}"
                    )
        elif issubclass(self.choices, Enum):
            for value in value_list:
                if value.name not in [e.name for e in self.choices]:
                    raise ValueError(
                        f"User Config: {value.name} not in list of choices: f{self.choices}"
                    )

    def __setattr__(self, name, value):
        self.__dict__[name] = value

        if name == "value":
            self.is_set_by_user = True
            self._check_bounds()
            self._check_choices()

    def __eq__(self, other):
        if not isinstance(other, ConfigField):
            return False
        return self.__dict__ == other.__dict__


class ConfigFields:
    """
    A class to represent a collection of configuration fields (ConfigField).

    The __getattr__ and __setattr__ methods are custom to allow
    the user to access the value of the ConfigField directly.
    """

    def __init__(self):
        self._fields = {}
        self._children = {}

        # This exists just to make looking up values when debugging easier
        self._values = {}

    def get_field(self, name):
        if name not in self._fields:
            raise ValueError(f"{name} not found in ConfigFields")

        return self._fields[name]

    def __setattr__(self, name, value):
        # This prevents recursion failure in __init__
        if name == "_fields" or name == "_values" or name == "_children":
            self.__dict__[name] = value
        else:
            if type(value) is ConfigField:
                self._fields[name] = value
            elif name in self._fields:
                self._fields[name].value = value
            else:
                self._children[name] = value
                self._values[name] = value
                return

            self._values[name] = self._fields[name].value

    def __getattr__(self, name):
        if name == "_fields" or name == "_values":
            return self.__dict__[name]
        elif name in self._children:
            return self._children[name]
        else:
            return self._fields[name].value

    def __deepcopy__(self, memo):
        new_copy = ConfigFields()
        new_copy._fields = deepcopy(self._fields, memo)
        new_copy._values = self._values
        new_copy._children = self._children
        return new_copy

    def __eq__(self, other):
        if not isinstance(other, ConfigFields):
            return False
        return self._fields == other._fields
