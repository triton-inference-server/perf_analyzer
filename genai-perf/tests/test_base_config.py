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

import unittest
from copy import deepcopy
from unittest.mock import patch

from genai_perf.config.input.base_config import BaseConfig, ConfigField
from genai_perf.config.input.config_defaults import Range
from genai_perf.inputs.input_constants import ModelSelectionStrategy, PromptSource


class TestBaseConfig(unittest.TestCase):

    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        pass

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Basic ConfigField Testing
    ###########################################################################
    def test_basic_config_field(self):
        """
        Test that a basic ConfigField can be written and read from
        """
        test_field = ConfigField(
            default=1,
            required=True,
            add_to_template=False,
            template_comment="test comment",
            value=2,
        )

        self.assertEqual(test_field.default, 1)
        self.assertEqual(test_field.required, True)
        self.assertEqual(test_field.add_to_template, False)
        self.assertEqual(test_field.template_comment, "test comment")
        self.assertEqual(test_field.value, 2)
        self.assertEqual(test_field.bounds, None)
        self.assertEqual(test_field.choices, None)

    def test_is_set_by_user(self):
        """
        Test that is_set_by_user is set correctly
        """
        test_field = ConfigField(
            default=1,
            required=True,
            add_to_template=False,
            template_comment="test comment",
            value=2,
        )

        self.assertEqual(test_field.is_set_by_user, True)

    def test_is_not_set_by_user(self):
        """
        Test that is_set_by_user is not set when no value is specified
        """
        test_field = ConfigField(
            default=1,
            required=True,
            add_to_template=False,
            template_comment="test comment",
        )

        self.assertEqual(test_field.is_set_by_user, False)

    ###########################################################################
    # ConfigField Bounds & Choice Testing
    ###########################################################################
    def test_config_field_bounds(self):
        """
        Test that a ConfigField with bounds can be written and read from
        """
        test_field = ConfigField(
            default=1,
            required=True,
            add_to_template=False,
            template_comment="test comment",
            value=2,
            bounds={"upper": 3, "lower": 1},
        )

        self.assertEqual(test_field.default, 1)
        self.assertEqual(test_field.required, True)
        self.assertEqual(test_field.add_to_template, False)
        self.assertEqual(test_field.template_comment, "test comment")
        self.assertEqual(test_field.value, 2)
        self.assertEqual(test_field.bounds, {"upper": 3, "lower": 1})
        self.assertEqual(test_field.choices, None)

    def test_config_field_choices(self):
        """
        Test that a ConfigField with choices can be written and read from
        """
        test_field = ConfigField(
            default=1,
            required=True,
            add_to_template=False,
            template_comment="test comment",
            value=2,
            choices=[1, 2, 3],
        )

        self.assertEqual(test_field.default, 1)
        self.assertEqual(test_field.required, True)
        self.assertEqual(test_field.add_to_template, False)
        self.assertEqual(test_field.template_comment, "test comment")
        self.assertEqual(test_field.value, 2)
        self.assertEqual(test_field.bounds, None)
        self.assertEqual(test_field.choices, [1, 2, 3])

    def test_config_field_out_of_bounds(self):
        """
        Test that a ConfigField with out of bounds value raises an error
        """
        with self.assertRaises(ValueError):
            _ = ConfigField(
                default=1,
                required=True,
                add_to_template=False,
                template_comment="test comment",
                value=4,
                bounds={"upper": 3, "lower": 1},
            )

    def test_config_field_invalid_choice(self):
        """
        Test that a ConfigField with invalid choice raises an error
        """
        with self.assertRaises(ValueError):
            _ = ConfigField(
                default=1,
                required=True,
                add_to_template=False,
                template_comment="test comment",
                value=4,
                choices=[1, 2, 3],
            )

    def test_config_field_invalid_choice_with_enum(self):
        """
        Test that a ConfigField with invalid choice raises an error when using an enum
        """
        with self.assertRaises(ValueError):
            _ = ConfigField(
                default=ModelSelectionStrategy.RANDOM,
                required=True,
                add_to_template=False,
                template_comment="test comment",
                value=PromptSource.SYNTHETIC,
                choices=ModelSelectionStrategy,
            )

    ###########################################################################
    # Basic BaseConfig Testing
    ###########################################################################
    def test_base_config(self):
        """
        Test that a BaseConfig object can be written and read from
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_base_config.test_field_B = ConfigField(default=3, value=4)

        # Check that just the value is returned when accessing the attribute
        self.assertEqual(test_base_config.test_field_A, 2)
        self.assertEqual(test_base_config.test_field_B, 4)

        # Check the get_field() method
        self.assertEqual(test_base_config.get_field("test_field_A").value, 2)
        self.assertEqual(test_base_config.get_field("test_field_A").default, 1)
        self.assertEqual(
            test_base_config.get_field("test_field_A").template_comment, "test comment"
        )

        self.assertEqual(test_base_config.get_field("test_field_B").value, 4)

    def test_base_config_change_value(self):
        """
        Test that a BaseConfig object can have its values changed
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )

        # Change the value of a field
        test_base_config.test_field_A = 5

        # Check that the value has changed
        self.assertEqual(test_base_config.test_field_A, 5)

    def test_base_config_change_bounds(self):
        """
        Test that a BaseConfig object can have its bounds changed
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment", bounds={"upper": 3}
        )

        # Change the bounds of a field
        test_base_config.get_field("test_field_A").bounds = {"upper": 5}

        # Check that the value has changed
        self.assertEqual(
            test_base_config.get_field("test_field_A").bounds, {"upper": 5}
        )

    def test_base_config_change_choices_using_a_list(self):
        """
        Test that a BaseConfig object can have its choices changed using a list
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment", choices=[1, 2, 3]
        )

        # Change the choices of a field
        test_base_config.get_field("test_field_A").choices = [1, 2, 3, 4]

        # Check that the value has changed
        self.assertEqual(
            test_base_config.get_field("test_field_A").choices, [1, 2, 3, 4]
        )

    def test_base_config_change_choices_using_enum(self):
        """
        Test that a BaseConfig object can have its choices changed using an Enum
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=ModelSelectionStrategy.RANDOM,
            value=ModelSelectionStrategy.ROUND_ROBIN,
            template_comment="test comment",
            choices=ModelSelectionStrategy,
        )

        # Change the choices of a field
        test_base_config.get_field("test_field_A").choices = PromptSource

        # Check that the value has changed
        self.assertEqual(
            test_base_config.get_field("test_field_A").choices, PromptSource
        )

    def test_base_config_out_of_bounds_enum(self):
        """
        Test that a BaseConfig object with an out of bounds enum value raises an error
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=ModelSelectionStrategy.RANDOM,
            value=ModelSelectionStrategy.ROUND_ROBIN,
            template_comment="test comment",
            choices=ModelSelectionStrategy,
        )

        # Change the value of a field to an out of bounds value
        with self.assertRaises(ValueError):
            test_base_config.test_field_A = PromptSource.SYNTHETIC

    ###########################################################################
    # Utility Testing
    ###########################################################################
    def test_base_config_deepcopy(self):
        """
        Test that a BaseConfig object can be deepcopied
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        child_base_config = BaseConfig()
        child_base_config.test_field_B = ConfigField(default=3, value=4)
        test_base_config.child_element = child_base_config

        test_base_config_copy = deepcopy(test_base_config)

        # Check that the copied object is not the same object
        self.assertNotEqual(id(test_base_config), id(test_base_config_copy))

        # Check that the copied object is equal to the original object
        self.assertEqual(
            test_base_config.test_field_A, test_base_config_copy.test_field_A
        )

        # Modify the copied object
        test_base_config_copy.child_element.test_field_B = 6

        # Check that the copied object is not equal to the original object
        self.assertNotEqual(
            test_base_config.child_element.test_field_B,
            test_base_config_copy.child_element.test_field_B,
        )

    def test_to_dict(self):
        """
        Test that a BaseConfig object can be converted to a dictionary
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_base_config.test_field_B = ConfigField(default=3, value=4)

        # Convert the object to a dictionary
        test_dict = test_base_config.to_json_dict()

        # Check that the dictionary is correct
        self.assertEqual(test_dict["test_field_A"], 2)
        self.assertEqual(test_dict["test_field_B"], 4)

    def test_to_dict_nested(self):
        """
        Test that a BaseConfig object with nested objects can be converted to a dictionary
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_base_config.test_field_B = BaseConfig()
        test_base_config.test_field_B.test_field_C = ConfigField(default=3, value=4)

        # Convert the object to a dictionary
        test_dict = test_base_config.to_json_dict()

        # Check that the dictionary is correct
        self.assertEqual(test_dict["test_field_A"], 2)
        self.assertEqual(test_dict["test_field_B"]["test_field_C"], 4)

    def test_to_dict_enum(self):
        """
        Test that a BaseConfig object with an Enum can be converted to a dictionary
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=ModelSelectionStrategy.RANDOM,
            value=ModelSelectionStrategy.ROUND_ROBIN,
            template_comment="test comment",
            choices=ModelSelectionStrategy,
        )

        # Convert the object to a dictionary
        test_dict = test_base_config.to_json_dict()

        # Check that the dictionary is correct
        self.assertEqual(test_dict["test_field_A"], "round_robin")

    def test_to_dict_range(self):
        """
        Test that a BaseConfig object with a Range can be converted to a dictionary
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=Range(0, 100), value=Range(10, 20), template_comment="test comment"
        )

        # Convert the object to a dictionary
        test_dict = test_base_config.to_json_dict()

        # Check that the dictionary is correct
        self.assertEqual(test_dict["test_field_A"], {"min": 10, "max": 20})

    ###########################################################################
    # Template Testing
    ###########################################################################
    def test_template(self):
        """
        Test that a BaseConfig object can be converted to a template
        """

        test_base_config = BaseConfig()
        test_base_config.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_base_config.test_field_B = ConfigField(default=3, value=4)

        # Create the template
        template = test_base_config.create_template(header="test")
        expected_template = (
            "  test:\n"
            + "    # test comment\n"
            + "    test_field_A: 2\n"
            + "    test_field_B: 4\n\n"
        )

        # Check that the template is correct
        self.assertEqual(
            template,
            expected_template,
        )


if __name__ == "__main__":
    unittest.main()
