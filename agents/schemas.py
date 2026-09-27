"""Pydantic schemas for LLM agent JSON. Explainer has no numeric fields."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NewsModel(BaseModel):
    """Structured news. Sources are attached by code from RSS, not by the model."""

    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["positive", "neutral", "negative"]
    summary: str = Field(min_length=1, max_length=800)


class EventItem(BaseModel):
    """One corporate date. Code decides inside_window / severity."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["earnings", "results", "agm", "dividend"]
    date: date

    @field_validator("date", mode="before")
    @classmethod
    def _parse_date(cls, value):
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            text = value.strip()
            if "T" in text:
                text = text.split("T", 1)[0]
            return date.fromisoformat(text)
        raise ValueError("date must be YYYY-MM-DD")


class EventsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventItem] = Field(default_factory=list)


class ExplainerModel(BaseModel):
    """Plain-words explanation. No prices, percents, or dates as schema fields."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)


NEWS_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "summary": {"type": "string"},
    },
    "required": ["sentiment", "summary"],
}

EVENTS_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["earnings", "results", "agm", "dividend"],
                    },
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["kind", "date"],
            },
        }
    },
    "required": ["events"],
}

EXPLAINER_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
}
