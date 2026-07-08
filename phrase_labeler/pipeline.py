import json
import os
from abc import abstractmethod
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import openai
from openai import OpenAI

from .prompting import PromptPermutationGenerator, PromptTemplate


def to_serializable(obj):
    """Recursively convert OpenAI SDK response to a JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if hasattr(obj, "to_dict"):
        return to_serializable(obj.to_dict())
    if hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())
    return obj  # base case


"""Supported LLM coding assistants."""
class LLM(Enum):
    ChatGPT = 0
    Claude = 1


def call_claude(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    temperature: float | None = None,
    api_key: str | None = None,
    system: str | None = None,
) -> Tuple[Dict, object]:
    """Send a prompt to the Claude API and return the query and response objects."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    kwargs: dict = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system is not None:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.messages.create(**kwargs)
    return dict(kwargs), response


def call_chatgpt(
    prompt: str,
    n: int = 1,
    temperature: float | None = None,
    model: str = "gpt-5-mini",
    reasoning_effort: str | None = None,
) -> Tuple[Dict, Dict]:
    """Send a prompt to the ChatGPT API and return the query and response objects."""
    query = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "n": n,
    }
    # Only set temperature when explicitly requested; some models (e.g. gpt-5-mini)
    # reject any value other than the default (1). Reasoning models ignore it entirely.
    if reasoning_effort is not None:
        query["reasoning_effort"] = reasoning_effort
    elif temperature is not None:
        query["temperature"] = temperature

    client = OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(**query)
    return query, response


def _extract_chatgpt_responses(response: dict) -> List[dict]:
    """
        Extracts the text part of a response JSON from ChatGPT. If there are more
        than 1 response (e.g., asking the LLM to generate multiple responses),
        this produces a list of all returned responses.
    """
    choices = response['response'].choices
    return [
        c.message.content
        for c in choices
    ]


def _extract_claude_responses(response: dict) -> List[str]:
    """Extract text content from a Claude API response."""
    api_response = response['response']
    return [block.text for block in api_response.content if hasattr(block, 'text')]


def extract_responses(response: dict, llm: LLM) -> List[dict]:
    """
        Given a LLM and a response object from its API, extract the
        text response(s) part of the response object.
    """
    if llm is LLM.ChatGPT or llm == LLM.ChatGPT.name:
        return _extract_chatgpt_responses(response)
    elif llm is LLM.Claude or llm == LLM.Claude.name:
        return _extract_claude_responses(response)
    else:
        raise ValueError(f"LLM {llm} is unsupported.")


def is_valid_filepath(filepath: str) -> bool:
    """Check that a file can be opened or created for caching responses."""
    try:
        with open(filepath, 'r'):
            return True
    except IOError:
        try:
            # Create the file if it doesn't exist, and write an empty json string to it
            with open(filepath, 'w+', encoding="utf-8") as f:
                f.write("{}")
            return True
        except IOError:
            return False


def is_valid_json(json_dict: dict) -> bool:
    """Return True if the input is a JSON-serializable dict."""
    if isinstance(json_dict, dict):
        try:
            json.dumps(json_dict)
            return True
        except (TypeError, ValueError):
            return False
    return False


class PromptPipeline:
    def __init__(self, storageFile: str):
        """Initialize the pipeline with a response cache file."""
        if not is_valid_filepath(storageFile):
            raise IOError(f"Filepath {storageFile} is invalid, or you do not have write access.")

        self._filepath = storageFile

    @abstractmethod
    def gen_prompts(self, properties) -> List[PromptTemplate]:
        raise NotImplementedError("Please Implement the gen_prompts method")

    @abstractmethod
    def analyze_response(self, response) -> bool:
        """
            Analyze the response and return True if the response is valid.
        """
        raise NotImplementedError("Please Implement the analyze_response method")

    def gen_responses(
        self,
        properties,
        llm: LLM,
        n: int = 1,
        temperature: float | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        anthropic_api_key: Optional[str] = None,
    ) -> Iterator[Dict]:
        """
            Calls LLM 'llm' with all prompts, and yields responses as dicts in format {prompt, query, response, llm, info}.

            By default, for each response, this also saves reponses to disk as JSON at the filepath given during init.
            (Very useful for saving money in case something goes awry!)
            To clear the cached responses, call clear_cached_responses().

            Do not override this function.
        """
        # Double-check that properties is the correct type (JSON dict):
        if not is_valid_json(properties):
            raise ValueError(f"Properties argument is not valid JSON.")

        # Generate concrete prompts using properties dict
        prompts = self.gen_prompts(properties)

        # Load any cache'd responses
        responses = self._load_cached_responses()

        # Query LLM with each prompt, yield + cache the responses
        for prompt in prompts:
            if isinstance(prompt, PromptTemplate) and not prompt.is_concrete():
                raise Exception(f"Cannot send a prompt '{prompt}' to LLM: Prompt is a template.")

            # Each prompt has a history of what was filled in from its base template.
            # This data --like, "class", "language", "library" etc --can be useful when parsing responses.
            info = prompt.fill_history
            prompt_str = str(prompt)

            # First check if there is already a response for this item. If so, we can save an LLM call:
            if prompt_str in responses:
                print(f"   - Found cache'd response for prompt {prompt_str}. Using...")
                yield {
                    "prompt": prompt_str,
                    "query": responses[prompt_str]["query"],
                    "response": responses[prompt_str]["response"],
                    "llm": responses[prompt_str]["llm"] if "llm" in responses[prompt_str] else LLM.ChatGPT.name,
                    "info": responses[prompt_str]["info"],
                }
                continue

            # Call the LLM to generate a response
            query, response = self._prompt_llm(
                llm, prompt_str, n, temperature,
                model=model, reasoning_effort=reasoning_effort,
                anthropic_api_key=anthropic_api_key,
            )

            # Save the response to a JSON file
            # NOTE: We do this to save money --in case something breaks between calls, can ensure we got the data!
            responses[prompt_str] = {
                "query": query,
                "response": response,
                "llm": llm.name,
                "info": info,
            }
            # self._cache_responses(responses)

            yield {
                "prompt":prompt_str,
                "query":query,
                "response":response,
                "llm": llm.name,
                "info": info,
            }

    def _load_cached_responses(self) -> Dict:
        """
            Loads saved responses of JSON at self._filepath.
            Useful for continuing if computation was interrupted halfway through.
        """
        if os.path.isfile(self._filepath):
            with open(self._filepath, encoding="utf-8") as f:
                responses = json.load(f)
            return responses
        else:
            return {}

    def _cache_responses(self, responses) -> None:
        """Persist responses to disk for caching."""
        with open(self._filepath, "w") as f:
            json.dump(to_serializable(responses), f)

    def clear_cached_responses(self) -> None:
        """Clear any cached responses stored on disk."""
        self._cache_responses({})

    def _prompt_llm(
        self,
        llm: LLM,
        prompt: str,
        n: int = 1,
        temperature: float | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        anthropic_api_key: Optional[str] = None,
    ) -> Tuple[Dict, Dict]:
        """Dispatch a prompt to the configured language model."""
        if llm is LLM.ChatGPT:
            kwargs = {"n": n, "temperature": temperature}
            if model is not None:
                kwargs["model"] = model
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            return call_chatgpt(prompt, **kwargs)
        elif llm is LLM.Claude:
            return call_claude(
                prompt,
                model=model or "claude-sonnet-4-6",
                temperature=temperature,
                api_key=anthropic_api_key,
            )
        else:
            raise Exception(f"Language model {llm} is not supported.")


class Phrase_TaggerPromptPipeline(PromptPipeline):
    def __init__(self, prompt_template: str):
        """Initialize the pipeline with a concrete prompt template."""
        self._template = PromptTemplate(prompt_template)
        storageFile = 'phrase_tagger_responses.json'
        super().__init__(storageFile)

    def gen_prompts(self, properties):
        """Generate concrete prompts from sentence properties."""
        gen_prompts = PromptPermutationGenerator(self._template)
        return list(gen_prompts({
            "sentence": properties["sentence"]
        }))
