import json
from string import Template
from typing import Dict, List, Union


class PromptTemplate:
    """
    Wrapper around string.Template. Use to generate prompts fast.

    Example usage:
        prompt_temp = PromptTemplate('Can you list all the cities in the country ${country} by the cheapest ${domain} prices?')
        concrete_prompt = prompt_temp.fill({
            "country": "France",
            "domain": "rent"
        });
        print(concrete_prompt)

        # Fill can also fill the prompt only partially, which gives us a new prompt template:
        partial_prompt = prompt_temp.fill({
            "domain": "rent"
        });
        print(partial_prompt)
    """
    def __init__(self, templateStr):
        """
            Initialize a PromptTemplate with a string in string.Template format.
            (See https://docs.python.org/3/library/string.html#template-strings for more details.)
        """
        try:
            Template(templateStr)
        except:
            raise Exception("Invalid template formatting for string:", templateStr)
        self.template = templateStr
        self.fill_history = {}

    def __str__(self) -> str:
        """Return the underlying template string."""
        return self.template

    def __repr__(self) -> str:
        """Return a debug-friendly string representation."""
        return self.__str__()

    def is_concrete(self) -> bool:
        """Returns True if no template variables are left in template string."""
        try:
            Template(self.template).substitute({})
            return True # no exception raised means there was nothing to substitute...
        except KeyError:
            return False

    def fill(self, paramDict: Dict[str, str]) -> 'PromptTemplate':
        """
            Formats the template string with the given parameters, returning a new PromptTemplate.
            Can return a partial completion.

            Example usage:
                prompt = prompt_template.fill({
                    "className": className,
                    "library": "Kivy",
                    "PL": "Python"
                });
        """
        filled_pt = PromptTemplate(
            Template(self.template).safe_substitute(paramDict)
        )

        # Deep copy prior fill history from this version over to new one
        filled_pt.fill_history = { key: val for (key, val) in self.fill_history.items() }

        # Add the new fill history using the passed parameters that we just filled in
        for key, val in paramDict.items():
            if key in filled_pt.fill_history:
                print(f"Warning: PromptTemplate already has fill history for key {key}.")
            filled_pt.fill_history[key] = val

        return filled_pt


class PromptPermutationGenerator:
    """
    Given a PromptTemplate and a parameter dict that includes arrays of items,
    generate all the permutations of the prompt for all permutations of the items.

    Example usage:
        prompt_gen = PromptPermutationGenerator('Can you list all the cities in the country ${country} by the cheapest ${domain} prices?')
        for prompt in prompt_gen({"country":["Canada", "South Africa", "China"],
                                  "domain": ["rent", "food", "energy"]}):
            print(prompt)
    """
    def __init__(self, template: Union[PromptTemplate, str]):
        """Initialize with a template or raw template string."""
        if isinstance(template, str):
            template = PromptTemplate(template)
        self.template = template

    def _gen_perm(self, template, params_to_fill, paramDict):
        """Recursively expand the prompt template over all parameter values."""
        if len(params_to_fill) == 0:
            return []

        # Peel off first element
        param = params_to_fill[0]
        params_left = params_to_fill[1:]

        # Generate new prompts by filling in its value(s) into the PromptTemplate
        val = paramDict[param]
        if isinstance(val, list):
            new_prompt_temps = [template.fill({param: v}) for v in val]
        elif isinstance(val, str):
            new_prompt_temps = [template.fill({param: val})]
        else:
            raise ValueError("Value of prompt template parameter is not a list or a string, but of type " + str(type(val)))

        # Recurse
        if len(params_left) == 0:
            return new_prompt_temps
        else:
            res = []
            for p in new_prompt_temps:
                res.extend(self._gen_perm(p, params_to_fill[1:], paramDict))
            return res

    def __call__(self, paramDict: Dict[str, Union[str, List[str]]]):
        """Yield PromptTemplate instances for every parameter permutation."""
        for p in self._gen_perm(self.template, list(paramDict.keys()), paramDict):
            yield p


DEFAULT_CATEGORIES = [
    "Status Quo/Context (the particular context or existing work)",
    "Challenge/Problem/Obstacle (often starts with 'however', gaps in prior work)",
    "Contribution (what the authors did)",
    "Purpose/Goal/Focus (why the work was done)",
    "Methodology (how the work was done)",
    "Participants (who were involved)",
    "System Description (of a system the authors developed or proposed)",
    "Findings",
    "Example"
]


DEFAULT_PROMPT_TEMPLATE = """A sentence (from a paper abstract) was splitted into several segments, put into the following list. For each list element, please classify it into one of the ${category_count} categories below, based on what it describes.
${description}
${sentence}

Categories:
${categories}

Please return a python list of the Category numbers only. The length of that list must be the same as that of the input list. If the task is impossible, return an empty list."""


DEFAULT_MULTI_LABEL_PROMPT_TEMPLATE = """You are an expert in linguistics and natural language processing with decades of experience analyzing academic writing.

Given a sentence from a paper abstract and a set of category labels, identify all meaningful text fragments and classify each fragment into exactly one category. Fragments may overlap — the same words can appear in multiple fragments with different labels. This captures how a single word or phrase can serve different roles depending on the surrounding context.

For each fragment, return a JSON object with:
- "text": the exact substring from the sentence being labeled
- "context": a longer surrounding substring that uniquely identifies where this fragment appears in the sentence (used for disambiguation when the same text appears multiple times)
- "label": the category number (integer) that best describes this fragment

If a portion of the sentence does not fit any category, include it with label -1.

Return a JSON array of these objects. Every word in the sentence should be covered by at least one fragment.

${description}

## Example 1: Overlapping labels

Sentence: "DynaVis addresses this challenge by blending natural language input with dynamically generated widgets that allow users to refine edits."

Categories:
0 Artifact
1 Method
2 Contribution
3 Interaction

Output:
```json
[
  {"text": "DynaVis", "context": "DynaVis addresses this challenge", "label": 0},
  {"text": "addresses this challenge", "context": "DynaVis addresses this challenge by", "label": 2},
  {"text": "by blending natural language input", "context": "challenge by blending natural language input with", "label": 1},
  {"text": "with dynamically generated widgets", "context": "input with dynamically generated widgets that", "label": 0},
  {"text": "that allow users to refine edits", "context": "widgets that allow users to refine edits", "label": 3},
  {"text": "DynaVis addresses this challenge by blending", "context": "DynaVis addresses this challenge by blending natural", "label": 1}
]
```

Notice how "addresses" appears in two fragments: once as part of "addresses this challenge" (Contribution) and once as part of the larger phrase "DynaVis addresses this challenge by blending" (Method). This is correct — the same words can serve different roles.

## Example 2: Unlabeled content

Sentence: "In this paper, we present a novel clustering algorithm for large-scale datasets."

Categories:
0 Artifact
1 Method

Output:
```json
[
  {"text": "In this paper,", "context": "In this paper, we present", "label": -1},
  {"text": "we present", "context": "this paper, we present a novel", "label": -1},
  {"text": "a novel clustering algorithm", "context": "present a novel clustering algorithm for", "label": 0},
  {"text": "for large-scale datasets", "context": "algorithm for large-scale datasets", "label": -1}
]
```

Here, "In this paper," and "we present" are rhetorical scaffolding that do not fit Artifact or Method, so they receive label -1.

${negative_examples}
---

Now classify the following:

Sentence: "${sentence}"

Categories:
${categories}

Output:"""


def format_categories(
    categories: list[str],
    descriptions: list[str] | None = None,
) -> str:
    """Format categories into a numbered list suitable for prompts.

    If descriptions are provided, non-empty descriptions are appended
    after the category name (e.g. "0 Artifact — the tool or system built").
    """
    lines = []
    for i, name in enumerate(categories):
        desc = descriptions[i] if descriptions and i < len(descriptions) else ""
        if desc:
            lines.append(f"{i} {name} — {desc}")
        else:
            lines.append(f"{i} {name}")
    return "\n".join(lines)


def format_sentence(sentence_list: list[str]) -> str:
    """Format sentence segments as pretty-printed JSON."""
    return json.dumps(sentence_list, ensure_ascii=False, indent=2)


def format_negative_examples(negative_examples: list[dict] | None) -> str:
    """Format a list of negative examples into a prompt section.

    Each example must be a dict with keys:
      - "sentence": the full sentence that contains the segment
      - "segment_text": the exact fragment the user marked as wrong
      - "do_not_label_as": the label name (string) the user wants to forbid

    Returns an empty string when no examples are given so the section
    cleanly disappears from the prompt.
    """
    if not negative_examples:
        return ""

    lines = [
        "## User corrections (avoid these mistakes)",
        "",
        "A human reviewer marked the following (sentence, segment) pairs as misclassified. "
        "Use these as calibration signal: for each identical sentence+segment pair, the forbidden label must not be applied; "
        "for similar segments elsewhere, treat them as hints about how you should resolve category boundaries in this taxonomy. "
        "Negative examples should take priority over your default judgement.",
        "",
    ]
    for ex in negative_examples:
        sentence = str(ex.get("sentence", "")).strip()
        segment = str(ex.get("segment_text", "")).strip()
        forbidden = str(ex.get("do_not_label_as", "")).strip()
        if not segment or not forbidden:
            continue
        lines.append(
            f'- Sentence: "{sentence}"\n'
            f'  Segment: "{segment}"\n'
            f'  Do NOT label this segment as: {forbidden}'
        )
    lines.append("")
    return "\n".join(lines)


def build_prompt(
    sentence_list: list[str],
    categories: list[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    description: str = "",
    negative_examples: list[dict] | None = None,
) -> str:
    """Construct a classification prompt by filling a template with segments and categories.

    The template may use ${description} to include an optional category-set description,
    and ${negative_examples} to include user-supplied do-not-label-as hints.
    If either is empty/missing, its placeholder renders as empty.
    """
    category_text = format_categories(categories)
    formatted_sentence = format_sentence(sentence_list)
    negative_text = format_negative_examples(negative_examples)

    return Template(prompt_template).safe_substitute({
        "sentence": formatted_sentence,
        "categories": category_text,
        "category_count": len(categories),
        "description": description,
        "negative_examples": negative_text,
    })


def build_multi_label_prompt(
    sentence: str,
    categories: list[str],
    prompt_template: str | None = None,
    description: str = "",
    category_descriptions: list[str] | None = None,
    negative_examples: list[dict] | None = None,
) -> str:
    """Construct a multi-label classification prompt for a raw sentence.

    Unlike build_prompt which takes pre-split segments, this takes
    the original sentence text and lets the LLM identify fragments.

    When prompt_template is None, the bundled DEFAULT_MULTI_LABEL_PROMPT_TEMPLATE
    is used so end-users do not need to ship any prompt files alongside the package.
    """
    if prompt_template is None:
        prompt_template = DEFAULT_MULTI_LABEL_PROMPT_TEMPLATE

    category_text = format_categories(categories, category_descriptions)
    negative_text = format_negative_examples(negative_examples)

    return Template(prompt_template).safe_substitute({
        "sentence": sentence,
        "categories": category_text,
        "category_count": len(categories),
        "description": description,
        "negative_examples": negative_text,
    })
