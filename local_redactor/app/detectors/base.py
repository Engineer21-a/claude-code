"""Shared detection types.

A `Detection` is a half-open character span `[start, end)` over a reconstructed
page string, plus metadata. Detectors never compute pixel/PDF boxes themselves;
`span_mapper` translates spans into boxes. This keeps every text detector
uniform and testable without any rendering backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable


@dataclass(frozen=True)
class Detection:
    start: int
    end: int
    label: str            # e.g. "user_word", "regex:iban", "gliner:person"
    text: str             # the matched substring (kept in-memory only, never logged)
    score: float = 1.0    # 1.0 for deterministic detectors; model confidence otherwise
    source: str = ""      # detector name, for review-UI colouring / audit counts
    #: When True the box should span the full text line, not just the tokens.
    full_line: bool = False

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Detection end {self.end} < start {self.start}")


@dataclass
class Token:
    """One OCR/text token: its substring and where it sits in the page string."""

    text: str
    start: int            # char offset (inclusive) into the page string
    end: int              # char offset (exclusive)
    box: "Box"            # bounding box in the token's native coordinate space
    line_id: int = 0      # tokens sharing a line_id form one text line
    confidence: float = 1.0


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle. Coordinate space is whatever produced it
    (image device pixels for OCR/Mode A, PDF points for born-digital/Mode B).
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def union(self, other: "Box") -> "Box":
        return Box(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def expand(self, margin: float) -> "Box":
        return Box(self.x0 - margin, self.y0 - margin, self.x1 + margin, self.y1 + margin)

    def intersects(self, other: "Box") -> bool:
        return not (
            self.x1 < other.x0
            or other.x1 < self.x0
            or self.y1 < other.y0
            or other.y1 < self.y0
        )

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class PageText:
    """A reconstructed page string with the tokens that back every character."""

    text: str
    tokens: List[Token] = field(default_factory=list)


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, page: PageText) -> List[Detection]:
        ...
