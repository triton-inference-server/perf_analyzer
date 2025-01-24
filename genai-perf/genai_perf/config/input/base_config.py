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

from genai_perf.config.input.config_field import ConfigField


class BaseConfig:
    """
    A base class that holds a collection of configuration fields (ConfigField).

    The __getattr__ and __setattr__ methods are custom to allow
    the user to access the value of the ConfigField directly.

    Examples:
      field_a.field_b = 5 sets the value of field_b to 5
      field_a.field_b returns the value of field_b

      field_a.get_field("field_b") returns the ConfigField object of field_b
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

            if self._fields[name].is_set_by_user:
                self._values[name] = self._fields[name].value
            else:
                self._values[name] = self._fields[name].default

    def __getattr__(self, name):
        if name == "_fields" or name == "_values":
            return self.__dict__[name]
        elif name in self._children:
            return self._children[name]
        else:
            if self._fields[name].is_set_by_user:
                return self._fields[name].value
            else:
                return self._fields[name].default

    def __deepcopy__(self, memo):
        new_copy = BaseConfig()
        new_copy._fields = deepcopy(self._fields, memo)
        new_copy._values = self._values
        new_copy._children = self._children
        return new_copy

    def __eq__(self, other):
        if not isinstance(other, BaseConfig):
            return False
        return self._fields == other._fields
