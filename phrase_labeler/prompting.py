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
${sentence}

Categories:
${categories}

Please return a python list of the Category numbers only. The length of that list must be the same as that of the input list. If the task is impossible, return an empty list."""


def format_categories(categories: list[str]) -> str:
    """Format categories into a numbered list suitable for prompts."""
    return "\n".join([f"{i} {desc}" for i, desc in enumerate(categories)])


def format_sentence(sentence_list: list[str]) -> str:
    """Format sentence segments as pretty-printed JSON."""
    return json.dumps(sentence_list, ensure_ascii=False, indent=2)


def build_prompt(
    sentence_list: list[str],
    categories: list[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    """Construct a classification prompt by filling a template with segments and categories."""
    category_text = format_categories(categories)
    formatted_sentence = format_sentence(sentence_list)

    return Template(prompt_template).safe_substitute({
        "sentence": formatted_sentence,
        "categories": category_text,
        "category_count": len(categories),
    })
