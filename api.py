import os
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from phrase_labeler.categories import load_categories
from phrase_labeler.cli import find_labels, find_labels_multi, find_labels_multi_batch, suggest_categories
from phrase_labeler.prompting import DEFAULT_CATEGORIES

app = FastAPI(title="Phrase Labeler API", version="0.2.0")


def _get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")
    return key


def _get_anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured on the server.")
    return key


def _resolve_categories(categories: list[str] | None, description: str) -> tuple[list[str], str]:
    if categories:
        return categories, description
    defaults, default_desc = load_categories(None, defaults=DEFAULT_CATEGORIES)
    return defaults, description or default_desc


class LabelRequest(BaseModel):
    segments: list[str]
    categories: Optional[list[str]] = None
    description: str = ""
    model: Optional[str] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    negative_examples: Optional[list[dict]] = None


class MultiLabelRequest(BaseModel):
    sentence: str
    categories: Optional[list[str]] = None
    description: str = ""
    model: Optional[str] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    category_descriptions: Optional[list[str]] = None
    negative_examples: Optional[list[dict]] = None
    provider: Optional[str] = None


class BatchMultiLabelRequest(BaseModel):
    sentences: list[str]
    categories: Optional[list[str]] = None
    description: str = ""
    model: Optional[str] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    category_descriptions: Optional[list[str]] = None
    negative_examples: Optional[list[dict]] = None
    provider: Optional[str] = None


class SuggestCategoriesRequest(BaseModel):
    segments: list[str]
    model: Optional[str] = None
    provider: Optional[str] = None


def _resolve_provider(requested: Optional[str], model: Optional[str]) -> str:
    """Return 'anthropic' or 'openai', auto-detecting from model name if not explicit."""
    if requested:
        return requested.lower()
    if model and not model.startswith("claude-"):
        return "openai"
    return "anthropic"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/label")
def label(req: LabelRequest):
    """Classify pre-split segments into single labels (legacy mode)."""
    api_key = _get_api_key()
    cats, desc = _resolve_categories(req.categories, req.description)
    result = find_labels(
        req.segments,
        api_key,
        cats,
        description=desc,
        model=req.model,
        temperature=req.temperature,
        reasoning_effort=req.reasoning_effort,
        negative_examples=req.negative_examples or [],
    )
    return {"labels": result}


@app.post("/label-multi")
def label_multi(req: MultiLabelRequest):
    """Classify a raw sentence into overlapping labeled spans (multi-label mode)."""
    provider = _resolve_provider(req.provider, req.model)
    api_key = _get_api_key() if provider != "anthropic" else ""
    anthropic_api_key = _get_anthropic_api_key() if provider == "anthropic" else None
    cats, desc = _resolve_categories(req.categories, req.description)
    result = find_labels_multi(
        req.sentence,
        api_key,
        cats,
        description=desc,
        model=req.model,
        temperature=req.temperature,
        reasoning_effort=req.reasoning_effort,
        category_descriptions=req.category_descriptions,
        negative_examples=req.negative_examples or [],
        provider=provider,
        anthropic_api_key=anthropic_api_key,
    )
    return {"spans": result}


@app.post("/label-multi-batch")
def label_multi_batch(req: BatchMultiLabelRequest):
    """Classify multiple sentences into overlapping spans in a single LLM call."""
    provider = _resolve_provider(req.provider, req.model)
    api_key = _get_api_key() if provider != "anthropic" else ""
    anthropic_api_key = _get_anthropic_api_key() if provider == "anthropic" else None
    cats, desc = _resolve_categories(req.categories, req.description)
    try:
        results = find_labels_multi_batch(
            req.sentences,
            api_key,
            cats,
            description=desc,
            model=req.model,
            temperature=req.temperature,
            reasoning_effort=req.reasoning_effort,
            category_descriptions=req.category_descriptions,
            negative_examples=req.negative_examples or [],
            provider=provider,
            anthropic_api_key=anthropic_api_key,
        )
        return {"results": [{"spans": spans} for spans in results]}
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n\n{tb}")


@app.post("/suggest-categories")
def suggest_categories_endpoint(req: SuggestCategoriesRequest):
    """Suggest category labels for a list of text segments."""
    provider = _resolve_provider(req.provider, req.model)
    api_key = _get_api_key() if provider != "anthropic" else ""
    anthropic_api_key = _get_anthropic_api_key() if provider == "anthropic" else None
    try:
        result = suggest_categories(
            req.segments,
            api_key=api_key,
            model=req.model,
            provider=provider,
            anthropic_api_key=anthropic_api_key,
        )
        return {"categories": result}
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n\n{tb}")
