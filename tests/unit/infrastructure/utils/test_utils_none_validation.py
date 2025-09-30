"""
Tests for utils None validation and type safety.

This module tests the critical bug fix for None handling
in the preprocess_text utility function.
"""

import pytest

from local_rag_backend.utils import preprocess_text


class TestUtilsNoneValidation:
    """Test None validation and type safety in utils."""

    def test_preprocess_text_with_none(self):
        """Test that None input returns empty string."""
        result = preprocess_text(None)
        assert result == ""

    @pytest.mark.parametrize("non_string_input,expected_result", [
        (123, "123"),  # Integer
        (45.67, "45.67"),  # Float
        (True, "true"),  # Boolean
        ([], "[]"),  # List
        ({}, "{}"),  # Dict
        ((1, 2), "(1, 2)"),  # Tuple
    ])
    def test_preprocess_text_with_non_string_types(self, non_string_input, expected_result):
        """Test that non-string types are converted to strings."""
        result = preprocess_text(non_string_input)
        assert result == expected_result

    def test_preprocess_text_with_unconvertible_object(self):
        """Test handling of objects that can't be converted to string."""
        class UnconvertibleObject:
            def __str__(self):
                raise Exception("Cannot convert to string")

        obj = UnconvertibleObject()
        result = preprocess_text(obj)
        assert result == ""

    @pytest.mark.parametrize("valid_text,expected_result", [
        ("Hello World", "hello world"),
        ("  TRIM ME  ", "trim me"),
        ("HTML<b>bold</b>text", "html bold text"),
        ("Multiple   spaces", "multiple spaces"),
        ("", ""),  # Empty string
        ("   ", ""),  # Whitespace only
    ])
    def test_preprocess_text_with_valid_strings(self, valid_text, expected_result):
        """Test that valid strings are processed correctly."""
        result = preprocess_text(valid_text)
        assert result == expected_result

    def test_preprocess_text_preserves_original_functionality(self):
        """Test that original functionality is preserved."""
        # Test HTML tag removal
        html_text = "This is <b>bold</b> and <i>italic</i> text"
        result = preprocess_text(html_text)
        assert result == "this is bold and italic text"

        # Test whitespace normalization
        spaced_text = "Multiple    spaces   and\ttabs\nand\nnewlines"
        result = preprocess_text(spaced_text)
        assert result == "multiple spaces and tabs and newlines"

        # Test case conversion
        mixed_case = "MiXeD CaSe TeXt"
        result = preprocess_text(mixed_case)
        assert result == "mixed case text"

    def test_preprocess_text_with_unicode(self):
        """Test handling of unicode characters."""
        unicode_texts = [
            ("Café résumé", "café résumé"),
            ("机器学习", "机器学习"),
            ("Здравствуй мир", "здравствуй мир"),
            ("🤖 AI Robot", "🤖 ai robot"),
        ]

        for input_text, expected in unicode_texts:
            result = preprocess_text(input_text)
            assert result == expected

    def test_preprocess_text_with_special_characters(self):
        """Test handling of special characters and symbols."""
        special_texts = [
            ("Hello@World.com", "hello@world.com"),
            ("Price: $19.99!", "price: $19.99!"),
            ("Math: 2+2=4", "math: 2+2=4"),
            ("Symbols: !@#$%^&*()", "symbols: !@#$%^&*()"),
        ]

        for input_text, expected in special_texts:
            result = preprocess_text(input_text)
            assert result == expected

    def test_preprocess_text_defensive_behavior(self):
        """Test that function never crashes with any input."""
        # These should all return strings without crashing
        test_inputs = [
            None,
            "",
            "normal text",
            123,
            [],
            {},
            object(),
            lambda x: x,  # Function object
        ]

        for test_input in test_inputs:
            result = preprocess_text(test_input)
            assert isinstance(result, str), f"Failed for input: {test_input}"

    def test_preprocess_text_empty_and_whitespace_handling(self):
        """Test specific handling of empty and whitespace inputs."""
        empty_cases = [
            ("", ""),
            ("   ", ""),
            ("\t", ""),
            ("\n", ""),
            ("\r\n", ""),
            ("  \t\n\r  ", ""),
        ]

        for input_text, expected in empty_cases:
            result = preprocess_text(input_text)
            assert result == expected

    def test_preprocess_text_complex_html_structures(self):
        """Test handling of complex HTML structures."""
        complex_html_cases = [
            ("<div><p>Nested</p></div>", "nested"),
            ("<a href='link'>Text</a>", "text"),
            ("Before<br/>After", "before after"),
            ("<script>alert('xss')</script>Clean", "alert('xss') clean"),
            ("<!-- comment -->Visible", "visible"),
        ]

        for input_html, expected in complex_html_cases:
            result = preprocess_text(input_html)
            assert result == expected

    def test_preprocess_text_performance_with_large_input(self):
        """Test performance doesn't degrade significantly with large inputs."""
        # Create a large text input
        large_text = "This is a test sentence. " * 1000  # ~25KB of text

        result = preprocess_text(large_text)

        # Should still process correctly
        assert result.startswith("this is a test sentence.")
        assert len(result) > 0

    def test_preprocess_text_maintains_word_boundaries(self):
        """Test that HTML tag removal maintains word boundaries."""
        # This was a specific issue where tags were removed without spaces
        boundary_cases = [
            ("word<tag>word", "word word"),  # Should have space between words
            ("start<b></b>end", "start end"),  # Empty tags should create space
            ("multiple<i>tags</i>here", "multiple tags here"),
        ]

        for input_text, expected in boundary_cases:
            result = preprocess_text(input_text)
            assert result == expected

    def test_preprocess_text_type_annotation_compatibility(self):
        """Test that the function accepts both str and None as documented."""
        # This test ensures the type annotation is correct
        string_result = preprocess_text("test string")
        none_result = preprocess_text(None)

        assert isinstance(string_result, str)
        assert isinstance(none_result, str)
        assert string_result == "test string"
        assert none_result == ""

    def test_preprocess_text_regression_prevention(self):
        """Test specific cases that could cause regressions."""
        # These are edge cases that might break in future changes
        regression_cases = [
            (None, ""),  # The original bug
            ("", ""),    # Empty string handling
            ("   ", ""), # Whitespace-only handling
            ("<br/>", ""),  # Self-closing HTML tags
            ("a<b>c", "a c"),  # Unclosed tags
        ]

        for input_text, expected in regression_cases:
            result = preprocess_text(input_text)
            assert result == expected, f"Regression detected for input: {input_text!r}"
