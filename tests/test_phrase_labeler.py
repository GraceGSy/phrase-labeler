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

import phrase_labeler.cli as cli
import phrase_labeler.pipeline as pipeline
import phrase_labeler.prompting as prompting
from phrase_labeler.categories import load_categories, parse_categories_payload


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
        template = prompting.PromptTemplate("Hello ${name} from ${place}")
        partial = template.fill({"name": "Alice"})
        self.assertFalse(partial.is_concrete())
        full = partial.fill({"place": "Paris"})
        self.assertTrue(full.is_concrete())
        self.assertEqual(str(full), "Hello Alice from Paris")


class PromptPermutationGeneratorTests(unittest.TestCase):
    """Tests for prompt permutation generation."""

    def test_generates_all_permutations(self):
        """Generate prompts for all combinations of parameters."""
        generator = prompting.PromptPermutationGenerator("Hi ${x} ${y}")
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
        prompt = prompting.build_prompt(sentence, categories)
        self.assertIn('"This paper"', prompt)
        self.assertIn("0 Background", prompt)
        self.assertIn("1 Method", prompt)
        self.assertIn("Categories:", prompt)

    def test_build_prompt_injects_description(self):
        """Ensure ${description} placeholder is filled when description is provided."""
        sentence = ["seg"]
        categories = ["Cat"]
        template = "Desc: ${description}\n${sentence}\n${categories}"
        prompt = prompting.build_prompt(sentence, categories, template, description="My description")
        self.assertIn("Desc: My description", prompt)

    def test_build_prompt_empty_description_renders_blank(self):
        """Ensure ${description} renders as empty string when not provided."""
        sentence = ["seg"]
        categories = ["Cat"]
        template = "Desc: ${description}\n${sentence}\n${categories}"
        prompt = prompting.build_prompt(sentence, categories, template)
        self.assertIn("Desc: \n", prompt)


class CategoryParsingTests(unittest.TestCase):
    """Tests for category JSON parsing and description extraction."""

    def test_parse_list_format_no_description(self):
        """List format produces empty description."""
        label_map, desc = parse_categories_payload(["A", "B"])
        self.assertEqual(label_map, {0: "A", 1: "B"})
        self.assertEqual(desc, "")

    def test_parse_labels_object_with_description(self):
        """Object with labels key and description field returns both."""
        payload = {"description": "My category set", "labels": {"0": "A", "1": "B"}}
        label_map, desc = parse_categories_payload(payload)
        self.assertEqual(label_map, {0: "A", 1: "B"})
        self.assertEqual(desc, "My category set")

    def test_parse_labels_object_without_description(self):
        """Object with labels key but no description returns empty string."""
        payload = {"labels": {"0": "A", "1": "B"}}
        label_map, desc = parse_categories_payload(payload)
        self.assertEqual(label_map, {0: "A", 1: "B"})
        self.assertEqual(desc, "")

    def test_parse_legacy_object_no_description(self):
        """Legacy object without labels key returns empty description."""
        payload = {"0": "A", "1": "B"}
        label_map, desc = parse_categories_payload(payload)
        self.assertEqual(label_map, {0: "A", 1: "B"})
        self.assertEqual(desc, "")

    def test_load_categories_defaults_when_no_file(self):
        """Returns defaults and empty description when no file is given."""
        cats, desc = load_categories(None, defaults=["X", "Y"])
        self.assertEqual(cats, ["X", "Y"])
        self.assertEqual(desc, "")

    def test_load_categories_from_file_with_description(self):
        """Loads categories and description from a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cats.json")
            with open(path, "w") as handle:
                json.dump({"description": "Test desc", "labels": {"0": "A", "1": "B"}}, handle)
            cats, desc = load_categories(path, defaults=["X", "Y"])
        self.assertEqual(cats, ["A", "B"])
        self.assertEqual(desc, "Test desc")

    def test_load_categories_from_file_without_description(self):
        """Loads categories from a file with no description field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cats.json")
            with open(path, "w") as handle:
                json.dump({"labels": {"0": "A", "1": "B"}}, handle)
            cats, desc = load_categories(path, defaults=["X", "Y"])
        self.assertEqual(cats, ["A", "B"])
        self.assertEqual(desc, "")


class ResponseExtractionTests(unittest.TestCase):
    """Tests for response extraction utilities."""

    def test_extract_responses_chatgpt(self):
        """Extract message content from a fake ChatGPT response."""
        response = {"response": FakeResponse(["first", "second"])}
        extracted = pipeline.extract_responses(response, pipeline.LLM.ChatGPT)
        self.assertEqual(extracted, ["first", "second"])


class FileValidationTests(unittest.TestCase):
    """Tests for file validation helpers."""

    def test_is_valid_filepath_creates_file(self):
        """Create a cache file when the path does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.json")
            self.assertTrue(pipeline.is_valid_filepath(path))
            self.assertTrue(os.path.exists(path))


class CLITests(unittest.TestCase):
    """Tests for CLI argument handling."""

    def test_main_uses_override_categories_file(self):
        """Use the supplied categories file when --override-categories is passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cats.json")
            with open(path, "w") as handle:
                json.dump({"labels": {"0": "A", "1": "B"}}, handle)

            argv = ["label-phrase", '["seg"]', "test-key", "--override-categories", path]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(cli, "find_labels") as mock_find:
                    cli.main()

            mock_find.assert_called_once()
            called_sentence, called_key, called_categories, called_prompt, called_desc = mock_find.call_args[0]
            self.assertEqual(called_sentence, ["seg"])
            self.assertEqual(called_key, "test-key")
            self.assertEqual(called_categories, ["A", "B"])
            self.assertEqual(called_prompt, prompting.DEFAULT_PROMPT_TEMPLATE)
            self.assertEqual(called_desc, "")

    def test_main_uses_defaults_when_no_flag(self):
        """Use default categories when --override-categories is not provided."""
        argv = ["label-phrase", '["seg"]', "test-key"]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(cli, "find_labels") as mock_find:
                cli.main()

        mock_find.assert_called_once()
        called_categories = mock_find.call_args[0][2]
        self.assertEqual(called_categories, prompting.DEFAULT_CATEGORIES)

    def test_main_passes_description_from_file(self):
        """Description from the categories file is propagated to find_labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cats.json")
            with open(path, "w") as handle:
                json.dump({"description": "Test desc", "labels": {"0": "A"}}, handle)

            argv = ["label-phrase", '["seg"]', "test-key", "--override-categories", path]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(cli, "find_labels") as mock_find:
                    cli.main()

        called_desc = mock_find.call_args[0][4]
        self.assertEqual(called_desc, "Test desc")

    def test_main_uses_prompt_file(self):
        """Load the prompt template from the provided file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cats_path = os.path.join(tmpdir, "cats.json")
            prompt_path = os.path.join(tmpdir, "prompt.txt")
            with open(cats_path, "w") as handle:
                json.dump({"labels": {"0": "A"}}, handle)
            with open(prompt_path, "w") as handle:
                handle.write("Prompt: ${sentence}\n${categories}")

            argv = ["label-phrase", '["seg"]', "test-key", "--override-categories", cats_path, "--prompt-file", prompt_path]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(cli, "find_labels") as mock_find:
                    cli.main()

            mock_find.assert_called_once()
            called_prompt = mock_find.call_args[0][3]
            self.assertEqual(called_prompt, "Prompt: ${sentence}\n${categories}")


class FindLabelsTests(unittest.TestCase):
    """Tests for label generation output formatting."""

    def test_find_labels_outputs_labels(self):
        """Print labeled segments based on the model's numeric output."""
        fake_iter = iter([{"response": FakeResponse(["[0, 1]"])}])
        with mock.patch.object(cli.Phrase_TaggerPromptPipeline, "gen_responses", return_value=fake_iter):
            with mock.patch.object(cli.Phrase_TaggerPromptPipeline, "clear_cached_responses", return_value=None):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    cli.find_labels(["alpha", "beta"], "test-key", ["Cat1", "Cat2"])

        output = ast.literal_eval(buffer.getvalue().strip())
        expected = [
            {"text": "alpha", "label": 0},
            {"text": "beta", "label": 1},
        ]
        self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
