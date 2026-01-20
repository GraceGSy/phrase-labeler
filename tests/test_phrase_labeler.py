import ast
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, api_key=None):
            """Create a minimal stub that mimics the OpenAI client shape."""
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )

    fake_openai.api_key = None
    fake_openai.OpenAI = DummyOpenAI
    sys.modules["openai"] = fake_openai

import phrase_labeler.phrase_labeler as pl


class FakeMessage:
    def __init__(self, content):
        """Hold message content for fake API responses."""
        self.content = content


class FakeChoice:
    def __init__(self, content):
        """Wrap a fake message to mimic the OpenAI choice object."""
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, contents):
        """Provide a list of fake choice objects for response extraction."""
        self.choices = [FakeChoice(content) for content in contents]


class PromptTemplateTests(unittest.TestCase):
    """Tests for prompt templating behavior."""

    def test_fill_and_is_concrete(self):
        """Ensure filling templates tracks concreteness and formatting."""
        template = pl.PromptTemplate("Hello ${name} from ${place}")
        partial = template.fill({"name": "Alice"})
        self.assertFalse(partial.is_concrete())
        full = partial.fill({"place": "Paris"})
        self.assertTrue(full.is_concrete())
        self.assertEqual(str(full), "Hello Alice from Paris")


class PromptPermutationGeneratorTests(unittest.TestCase):
    """Tests for prompt permutation generation."""

    def test_generates_all_permutations(self):
        """Generate prompts for all combinations of parameters."""
        generator = pl.PromptPermutationGenerator("Hi ${x} ${y}")
        prompts = list(generator({"x": ["A", "B"], "y": ["1", "2"]}))
        rendered = {str(p) for p in prompts}
        expected = {
            "Hi A 1",
            "Hi A 2",
            "Hi B 1",
            "Hi B 2",
        }
        self.assertEqual(rendered, expected)


class PromptBuildTests(unittest.TestCase):
    """Tests for prompt construction from sentences and categories."""

    def test_build_prompt_includes_categories_and_sentence(self):
        """Ensure prompt includes the provided categories and segments."""
        sentence = ["This paper", "proposes a method"]
        categories = ["Background", "Method"]
        prompt = pl.build_prompt(sentence, categories)
        self.assertIn('"This paper"', prompt)
        self.assertIn("0 Background", prompt)
        self.assertIn("1 Method", prompt)
        self.assertIn("Categories:", prompt)


class ResponseExtractionTests(unittest.TestCase):
    """Tests for response extraction utilities."""

    def test_extract_responses_chatgpt(self):
        """Extract message content from a fake ChatGPT response."""
        response = {"response": FakeResponse(["first", "second"])}
        extracted = pl.extract_responses(response, pl.LLM.ChatGPT)
        self.assertEqual(extracted, ["first", "second"])


class FileValidationTests(unittest.TestCase):
    """Tests for file validation helpers."""

    def test_is_valid_filepath_creates_file(self):
        """Create a cache file when the path does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.json")
            self.assertTrue(pl.is_valid_filepath(path))
            self.assertTrue(os.path.exists(path))


class CLITests(unittest.TestCase):
    """Tests for CLI argument handling."""

    def test_main_uses_custom_categories_file(self):
        """Use the supplied categories file when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cats.json")
            with open(path, "w") as handle:
                json.dump(["A", "B"], handle)

            argv = ["label-phrase", '["seg"]', "test-key", path]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(pl, "find_labels") as mock_find:
                    pl.main()

            mock_find.assert_called_once()
            called_sentence, called_key, called_categories = mock_find.call_args[0]
            self.assertEqual(called_sentence, ["seg"])
            self.assertEqual(called_key, "test-key")
            self.assertEqual(called_categories, ["A", "B"])

    def test_main_extends_default_categories(self):
        """Append user categories when --extend-categories is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cats.json")
            with open(path, "w") as handle:
                json.dump(["Extra"], handle)

            argv = ["label-phrase", '["seg"]', "test-key", path, "--extend-categories"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(pl, "find_labels") as mock_find:
                    pl.main()

            mock_find.assert_called_once()
            called_categories = mock_find.call_args[0][2]
            self.assertEqual(called_categories, pl.DEFAULT_CATEGORIES + ["Extra"])


class FindLabelsTests(unittest.TestCase):
    """Tests for label generation output formatting."""

    def test_find_labels_outputs_labels(self):
        """Print labeled segments based on the model's numeric output."""
        fake_iter = iter([{"response": FakeResponse(["[0, 1]"])}])
        with mock.patch.object(pl.Phrase_TaggerPromptPipeline, "gen_responses", return_value=fake_iter):
            with mock.patch.object(pl.Phrase_TaggerPromptPipeline, "clear_cached_responses", return_value=None):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    pl.find_labels(["alpha", "beta"], "test-key", ["Cat1", "Cat2"])

        output = ast.literal_eval(buffer.getvalue().strip())
        expected = [
            {"text": "alpha", "label": 0},
            {"text": "beta", "label": 1},
        ]
        self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
