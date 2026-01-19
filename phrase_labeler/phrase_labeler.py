import openai
import string
import argparse
from collections import Counter
from typing import Dict, Tuple, List, Union, Iterator
from enum import Enum
from openai import OpenAI
import json
from string import Template
from abc import abstractmethod
import os
import ast

def to_serializable(obj):
    """Recursively convert OpenAI SDK response to a JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif hasattr(obj, "to_dict"):
        return to_serializable(obj.to_dict())
    elif hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())
    else:
        return obj  # base case


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
        return self.template
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def is_concrete(self) -> bool:
        """ Returns True if no template variables are left in template string.
        """
        try:
            Template(self.template).substitute({})
            return True # no exception raised means there was nothing to substitute...
        except KeyError as e:
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
        if isinstance(template, str):
            template = PromptTemplate(template)
        self.template = template
    
    def _gen_perm(self, template, params_to_fill, paramDict):
        if len(params_to_fill) == 0: return []

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
        for p in self._gen_perm(self.template, list(paramDict.keys()), paramDict):
            yield p




""" Supported LLM coding assistants """
class LLM(Enum):
    ChatGPT = 0

def call_chatgpt(prompt: str, n: int = 1, temperature: float = 1.0) -> Tuple[Dict, Dict]:
    query = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "n": n,
        "temperature": temperature,
    }
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
        for i, c in enumerate(choices)
    ]

def extract_responses(response: dict, llm: LLM) -> List[dict]:
    """
        Given a LLM and a response object from its API, extract the
        text response(s) part of the response object.
    """
    if llm is LLM.ChatGPT or llm == LLM.ChatGPT.name:
        return _extract_chatgpt_responses(response)
    else:
        raise ValueError(f"LLM {llm} is unsupported.")

def is_valid_filepath(filepath: str) -> bool:
    try:
        with open(filepath, 'r'):
            pass
    except IOError:
        try:
            # Create the file if it doesn't exist, and write an empty json string to it
            with open(filepath, 'w+') as f:
                f.write("{}")
                pass
        except IOError:
            return False
    return True

def is_valid_json(json_dict: dict) -> bool:
    if isinstance(json_dict, dict):
        try:
            json.dumps(json_dict)
            return True
        except:
            pass
    return False


class PromptPipeline:
    def __init__(self, storageFile: str):
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
    
    def gen_responses(self, properties, llm: LLM, n: int = 1, temperature: float = 1.0) -> Iterator[Dict]:
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
            query, response = self._prompt_llm(llm, prompt_str, n, temperature)

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
        with open(self._filepath, "w") as f:
            json.dump(to_serializable(responses), f)
    
    def clear_cached_responses(self) -> None:
        self._cache_responses({})

    def _prompt_llm(self, llm: LLM, prompt: str, n: int = 1, temperature: float = 1.0) -> Tuple[Dict, Dict]:
        if llm is LLM.ChatGPT:
            return call_chatgpt(prompt, n=n, temperature=temperature)
        else:
            raise Exception(f"Language model {llm} is not supported.")


TEMPERATURE = 0.2 #The temperature for ChatGPT calls

global PHRASE_TAGGER_PROMPT_TEMPLATE = \
"""A sentence (from a paper abstract) was splitted into several segments, put into the following list. For each list element, please classify it into one of the 9 categories below, based on what it describes.
"${sentence}"

Categories: 
0 Status Quo/Context (the particular context or existing work)
1 Challenge/Problem/Obstacle (often starts with 'however', gaps in prior work)
2 Contribution (what the authors did)
3 Purpose/Goal/Focus (why the work was done)
4 Methodology (how the work was done)
5 Participants (who were involved)
6 System Description (of a system the authors developed or proposed)
7 Findings
8 Example

Please return a python list of the category numbers only. The length of that list must be the same as that of the input list. If the task is impossible, return an empty list."""


class Phrase_TaggerPromptPipeline(PromptPipeline):
    def __init__(self):
        self._template = PromptTemplate(PHRASE_TAGGER_PROMPT_TEMPLATE)
        storageFile = 'phrase_tagger_responses.json'
        super().__init__(storageFile)
    def gen_prompts(self, properties):
        gen_prompts = PromptPermutationGenerator(self._template)
        return list(gen_prompts({
            "sentence": properties["sentence"]
        }))


def build_prompt(sentence_list: list[str], categories: list[str]) -> str:
    category_text = "\n".join([f"{i} {desc}" for i, desc in enumerate(categories)])
    formatted_sentence = json.dumps(sentence_list, ensure_ascii=False, indent=2)
    
    return f"""A sentence (from a paper abstract) was splitted into several segments, put into the following list. For each list element, please classify it into one of the {len(categories)} categories below, based on what it describes.
        {formatted_sentence}

        Categories: 
        {category_text}

        Please return a python list of the Category numbers only. The length of that list must be the same as that of the input list. If the task is impossible, return an empty list."""


# the function to invoke if you're working within a Python environment
def find_labels(segmented_sent, k):
    output = []
    openai.api_key = k
    phrase_tagger = Phrase_TaggerPromptPipeline()
    tmp = []
    phrase_tagger.clear_cached_responses()
    for res in phrase_tagger.gen_responses({"sentence": str(segmented_sent)}, LLM.ChatGPT, n=1, temperature=TEMPERATURE):
        tmp.extend(extract_responses(res, llm=LLM.ChatGPT))
    color_list = ast.literal_eval(tmp[0])
    if len(color_list) == len(segmented_sent):
        for j, segment in enumerate(segmented_sent):
            output.append({'text': segment, 'label': color_list[j]})
    else:
        buggy_case_counter += 1
        print(segmented_sent)
        print(color_list)
        for j, segment in enumerate(segmented_sent):
            output.append({'text': segment, 'label': 0})
    print(output)


# for commandline invocation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=json.loads,
                        help="The sentence segments as a JSON list of strings (e.g. '[\"This paper\", \"proposes a method\"]')")
    parser.add_argument("api_key", help="Your OpenAI API key")
    parser.add_argument("categories_file", type=str,
                        help="Path to JSON file containing category list")

    args = parser.parse_args()

    # Load categories from file or use default
    if args.categories_file:
        if not os.path.exists(args.categories_file):
            raise FileNotFoundError(f"File not found: {args.categories_file}")
        with open(args.categories_file, "r") as f:
            categories = json.load(f)
        if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
            raise ValueError("The categories file must contain a JSON list of strings.")
    else:
        categories = [
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

    # Generate prompt
    PHRASE_TAGGER_PROMPT_TEMPLATE = build_prompt(args.sentence, categories)
    find_labels(args.sentence, args.api_key)
