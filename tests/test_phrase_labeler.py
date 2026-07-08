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

# Load OPENAI_API_KEY from the repo .env if not already in the environment
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _fh:
        for _line in _fh:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_requires_openai = unittest.skipUnless(_OPENAI_API_KEY, "OPENAI_API_KEY not set")

_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_requires_anthropic = unittest.skipUnless(_ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY not set")

try:
    import openai  # load the real package before the stub guard runs
except ImportError:
    pass

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


class NegativeExamplesTests(unittest.TestCase):
    """Tests for formatting and wiring negative examples into prompts."""

    def test_format_negative_examples_empty_returns_blank(self):
        """No examples should produce an empty section so the prompt stays clean."""
        self.assertEqual(prompting.format_negative_examples(None), "")
        self.assertEqual(prompting.format_negative_examples([]), "")

    def test_format_negative_examples_renders_each_triple(self):
        """Each example is rendered with its sentence, segment, and forbidden label."""
        examples = [
            {
                "sentence": "When editing visualizations users face difficulty.",
                "segment_text": "When editing visualizations",
                "do_not_label_as": "Setting",
            },
            {
                "sentence": "DynaVis proposes a hybrid solution.",
                "segment_text": "DynaVis proposes a hybrid solution",
                "do_not_label_as": "Contribution",
            },
        ]
        rendered = prompting.format_negative_examples(examples)
        self.assertIn("User corrections", rendered)
        self.assertIn('Segment: "When editing visualizations"', rendered)
        self.assertIn("Do NOT label this segment as: Setting", rendered)
        self.assertIn('Segment: "DynaVis proposes a hybrid solution"', rendered)
        self.assertIn("Do NOT label this segment as: Contribution", rendered)

    def test_format_negative_examples_skips_malformed(self):
        """Examples missing segment_text or do_not_label_as should be skipped silently."""
        examples = [
            {"sentence": "A", "segment_text": "", "do_not_label_as": "Goal"},
            {"sentence": "B", "segment_text": "something", "do_not_label_as": ""},
            {"sentence": "C", "segment_text": "good", "do_not_label_as": "Setting"},
        ]
        rendered = prompting.format_negative_examples(examples)
        self.assertIn('Segment: "good"', rendered)
        self.assertIn("Do NOT label this segment as: Setting", rendered)
        # Malformed entries should not produce stray "Segment:" lines
        self.assertEqual(rendered.count("Segment:"), 1)

    def test_build_multi_label_prompt_includes_negative_examples(self):
        """The ${negative_examples} placeholder should be populated when examples are given."""
        examples = [
            {
                "sentence": "S",
                "segment_text": "seg",
                "do_not_label_as": "LabelA",
            }
        ]
        template = "SENTENCE: ${sentence}\nCATS:\n${categories}\nNEG:\n${negative_examples}"
        prompt = prompting.build_multi_label_prompt(
            "S", ["LabelA", "LabelB"], template, negative_examples=examples,
        )
        self.assertIn("SENTENCE: S", prompt)
        self.assertIn('Segment: "seg"', prompt)
        self.assertIn("Do NOT label this segment as: LabelA", prompt)

    def test_build_multi_label_prompt_default_template_when_none(self):
        """Passing prompt_template=None should fall back to the bundled multi-label default."""
        prompt = prompting.build_multi_label_prompt("S", ["A", "B"], None)
        self.assertIn('Sentence: "S"', prompt)
        self.assertIn("0 A", prompt)
        self.assertIn("1 B", prompt)

    def test_build_prompt_negative_examples_via_template(self):
        """Single-label build_prompt should also fill ${negative_examples} if present in template."""
        template = "${sentence}\n${categories}\nNEG:${negative_examples}"
        examples = [{"sentence": "S", "segment_text": "x", "do_not_label_as": "Y"}]
        prompt = prompting.build_prompt(["seg"], ["Cat"], template, negative_examples=examples)
        self.assertIn('Segment: "x"', prompt)
        self.assertIn("Do NOT label this segment as: Y", prompt)


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

    def test_main_loads_negative_examples_file(self):
        """--negative-examples should load a JSON file and pass its contents through."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cats_path = os.path.join(tmpdir, "cats.json")
            neg_path = os.path.join(tmpdir, "neg.json")
            with open(cats_path, "w") as handle:
                json.dump({"labels": {"0": "A"}}, handle)
            with open(neg_path, "w") as handle:
                json.dump(
                    [
                        {
                            "sentence": "S",
                            "segment_text": "seg",
                            "do_not_label_as": "A",
                        }
                    ],
                    handle,
                )

            argv = [
                "label-phrase",
                '["seg"]',
                "test-key",
                "--override-categories", cats_path,
                "--negative-examples", neg_path,
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(cli, "find_labels") as mock_find:
                    cli.main()

        kwargs = mock_find.call_args.kwargs
        self.assertIn("negative_examples", kwargs)
        self.assertEqual(len(kwargs["negative_examples"]), 1)
        self.assertEqual(kwargs["negative_examples"][0]["do_not_label_as"], "A")

    def test_main_multi_label_passes_negative_examples(self):
        """--multi-label + --negative-examples should pipe through to find_labels_multi."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cats_path = os.path.join(tmpdir, "cats.json")
            neg_path = os.path.join(tmpdir, "neg.json")
            with open(cats_path, "w") as handle:
                json.dump({"labels": {"0": "A"}}, handle)
            with open(neg_path, "w") as handle:
                json.dump(
                    [
                        {
                            "sentence": "S",
                            "segment_text": "seg",
                            "do_not_label_as": "A",
                        }
                    ],
                    handle,
                )

            argv = [
                "label-phrase",
                "The raw sentence.",
                "test-key",
                "--multi-label",
                "--override-categories", cats_path,
                "--negative-examples", neg_path,
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(cli, "find_labels_multi", return_value=[]) as mock_multi:
                    cli.main()

        kwargs = mock_multi.call_args.kwargs
        self.assertEqual(len(kwargs["negative_examples"]), 1)
        self.assertEqual(kwargs["negative_examples"][0]["segment_text"], "seg")

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


_SUGGEST_CATEGORIES_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "suggest-categories", "system_prompt.txt"
)
with open(_SUGGEST_CATEGORIES_PROMPT_PATH) as _fh:
    SUGGEST_CATEGORIES_SYSTEM_PROMPT = _fh.read().strip()


def _build_suggest_categories_user_message(segments: list[str]) -> str:
    lines = "\n".join(f"- {seg}" for seg in segments)
    return f"Text segments:\n{lines}\n\nReturn a JSON array of category label strings."


def _call_suggest_categories(segments: list[str], api_key: str) -> list[str]:
    """Send the category-suggestion prompt to OpenAI and return the parsed label list."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUGGEST_CATEGORIES_SYSTEM_PROMPT},
            {"role": "user", "content": _build_suggest_categories_user_message(segments)},
        ],
    )
    content = response.choices[0].message.content.strip()
    # Strip optional markdown code fences the model sometimes adds
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content.strip())


class SuggestCategoriesPromptTests(unittest.TestCase):
    """Tests for a prompt that suggests meaningful category labels from text segments."""

    _SAMPLE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "examples", "sample_negative_examples.json"
    )

    def _load_field(self, field: str) -> list[str]:
        with open(self._SAMPLE_PATH) as fh:
            return [ex[field] for ex in json.load(fh)]

    def test_system_prompt_contains_expert_framing(self):
        """System prompt must include the expert role and category-suggestion task."""
        self.assertIn("expert in linguistics", SUGGEST_CATEGORIES_SYSTEM_PROMPT)
        self.assertIn("category labels", SUGGEST_CATEGORIES_SYSTEM_PROMPT)
        self.assertIn("shared/repeated between text segments", SUGGEST_CATEGORIES_SYSTEM_PROMPT)

    def test_user_message_contains_all_segment_texts(self):
        """User message must include every segment_text from the sample file."""
        segments = self._load_field("segment_text")
        msg = _build_suggest_categories_user_message(segments)
        for seg in segments:
            self.assertIn(seg, msg)

    def test_user_message_contains_all_sentences(self):
        """User message built from full sentences must contain each sentence."""
        sentences = self._load_field("sentence")
        msg = _build_suggest_categories_user_message(sentences)
        for sent in sentences:
            self.assertIn(sent, msg)

    @_requires_openai
    def test_real_api_suggests_categories_from_segment_texts(self):
        """Real OpenAI call with segment_text inputs should return a non-empty list of dicts."""
        segments = self._load_field("segment_text")
        result = _call_suggest_categories(segments, _OPENAI_API_KEY)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertIn("label", item)
            self.assertIn("description", item)
            self.assertIsInstance(item["label"], str)
            self.assertIsInstance(item["description"], str)
            self.assertTrue(item["label"].strip())
            self.assertTrue(item["description"].strip())

    @_requires_openai
    def test_real_api_suggests_categories_from_full_sentences(self):
        """Real OpenAI call with full sentences should return a non-empty list of dicts."""
        sentences = self._load_field("sentence")
        result = _call_suggest_categories(sentences, _OPENAI_API_KEY)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertIn("label", item)
            self.assertIn("description", item)
            self.assertIsInstance(item["label"], str)
            self.assertIsInstance(item["description"], str)
            self.assertTrue(item["label"].strip())
            self.assertTrue(item["description"].strip())
        print("\nSentences:")
        for s in sentences:
            print(f"  - {s}")
        print("\nSuggested categories:")
        for item in result:
            print(f"  - {item['label']}: {item['description']}")


class ClaudeBatchLabelingTests(unittest.TestCase):
    """Integration tests for find_labels_multi_batch() using the Claude provider."""

    _SAMPLE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "examples", "sample_negative_examples.json"
    )
    _CATEGORIES = ["Contribution", "Methodology", "Findings", "Background", "Obstacle"]

    def _load_sentences(self) -> list[str]:
        with open(self._SAMPLE_PATH) as fh:
            return [ex["sentence"] for ex in json.load(fh)]

    @_requires_anthropic
    def test_claude_batch_labels_sample_sentences(self):
        """find_labels_multi_batch() with provider='anthropic' returns spans for each sentence."""
        sentences = self._load_sentences()
        results = cli.find_labels_multi_batch(
            sentences=sentences,
            api_key="",
            categories=self._CATEGORIES,
            provider="anthropic",
            anthropic_api_key=_ANTHROPIC_API_KEY,
        )

        print("\n--- Claude batch labeling results ---")
        for i, (sentence, spans) in enumerate(zip(sentences, results)):
            print(f"\n[{i}] {sentence}")
            for span in spans:
                label_name = self._CATEGORIES[span['label']] if span['label'] < len(self._CATEGORIES) else span['label']
                print(f"      [{span['start']}:{span['end']}] \"{span['text']}\" → {label_name}")
        print("--- end ---\n")

        self.assertEqual(len(results), len(sentences))
        for spans in results:
            self.assertIsInstance(spans, list)
            for span in spans:
                self.assertIn("text", span)
                self.assertIn("label", span)
                self.assertIn("start", span)
                self.assertIn("end", span)
                self.assertIsInstance(span["label"], int)
                self.assertGreaterEqual(span["start"], 0)
                self.assertGreater(span["end"], span["start"])


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
