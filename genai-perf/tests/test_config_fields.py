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

from genai_perf.config.input.config_fields import ConfigField, ConfigFields
from genai_perf.inputs.input_constants import ModelSelectionStrategy, PromptSource


class TestConfigFields(unittest.TestCase):

    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        pass

    def tearDown(self):
        patch.stopall()

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
            test_field = ConfigField(
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
            test_field = ConfigField(
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
            test_field = ConfigField(
                default=ModelSelectionStrategy.RANDOM,
                required=True,
                add_to_template=False,
                template_comment="test comment",
                value=PromptSource.SYNTHETIC,
                choices=ModelSelectionStrategy,
            )

    def test_config_fields(self):
        """
        Test that a ConfigFields object can be written and read from
        """

        test_fields = ConfigFields()
        test_fields.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )
        test_fields.test_field_B = ConfigField(default=3, value=4)

        # Check that just the value is returned when accessing the attribute
        self.assertEqual(test_fields.test_field_A, 2)
        self.assertEqual(test_fields.test_field_B, 4)

        # Check the get_field() method
        self.assertEqual(test_fields.get_field("test_field_A").value, 2)
        self.assertEqual(test_fields.get_field("test_field_A").default, 1)
        self.assertEqual(
            test_fields.get_field("test_field_A").template_comment, "test comment"
        )

        self.assertEqual(test_fields.get_field("test_field_B").value, 4)

    def test_config_fields_change_value(self):
        """
        Test that a ConfigFields object can have its values changed
        """

        test_fields = ConfigFields()
        test_fields.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )

        # Change the value of a field
        test_fields.test_field_A = 5

        # Check that the value has changed
        self.assertEqual(test_fields.test_field_A, 5)

    def test_config_fields_change_bounds(self):
        """
        Test that a ConfigFields object can have its bounds changed
        """

        test_fields = ConfigFields()
        test_fields.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment", bounds={"upper": 3}
        )

        # Change the bounds of a field
        test_fields.get_field("test_field_A").bounds = {"upper": 5}

        # Check that the value has changed
        self.assertEqual(test_fields.get_field("test_field_A").bounds, {"upper": 5})

    def test_config_fields_change_choices_using_a_list(self):
        """
        Test that a ConfigFields object can have its choices changed using a list
        """

        test_fields = ConfigFields()
        test_fields.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment", choices=[1, 2, 3]
        )

        # Change the choices of a field
        test_fields.get_field("test_field_A").choices = [1, 2, 3, 4]

        # Check that the value has changed
        self.assertEqual(test_fields.get_field("test_field_A").choices, [1, 2, 3, 4])

    def test_config_fields_change_choices_using_enum(self):
        """
        Test that a ConfigFields object can have its choices changed using an Enum
        """

        test_fields = ConfigFields()
        test_fields.test_field_A = ConfigField(
            default=ModelSelectionStrategy.RANDOM,
            value=ModelSelectionStrategy.ROUND_ROBIN,
            template_comment="test comment",
            choices=ModelSelectionStrategy,
        )

        # Change the choices of a field
        test_fields.get_field("test_field_A").choices = PromptSource

        # Check that the value has changed
        self.assertEqual(test_fields.get_field("test_field_A").choices, PromptSource)

    def test_config_fields_out_of_bounds_enum(self):
        """
        Test that a ConfigFields object with an out of bounds enum value raises an error
        """

        test_fields = ConfigFields()
        test_fields.test_field_A = ConfigField(
            default=ModelSelectionStrategy.RANDOM,
            value=ModelSelectionStrategy.ROUND_ROBIN,
            template_comment="test comment",
            choices=ModelSelectionStrategy,
        )

        # Change the value of a field to an out of bounds value
        with self.assertRaises(ValueError):
            test_fields.test_field_A = PromptSource.SYNTHETIC

    def test_config_fields_deepcopy(self):
        """
        Test that a ConfigFields object can be deepcopied
        """

        test_fields = ConfigFields()
        test_fields.test_field_A = ConfigField(
            default=1, value=2, template_comment="test comment"
        )

        test_fields_copy = deepcopy(test_fields)

        # Check that the copied object is not the same object
        self.assertNotEqual(id(test_fields), id(test_fields_copy))

        # Check that the copied object is equal to the original object
        self.assertEqual(test_fields.test_field_A, test_fields_copy.test_field_A)


if __name__ == "__main__":
    unittest.main()
