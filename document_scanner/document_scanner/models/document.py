from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FilterSettings:
    preset: str = "original"
    brightness: float = 0.0      # -1.0 to +1.0
    contrast: float = 1.0        # 0.01 to 3.0
    saturation: float = 1.0      # 0.0 to 3.0
    sharpness: float = 0.0       # 0.0 to 2.0
    shadow: float = 0.0          # -1.0 to +1.0
    highlight: float = 0.0       # -1.0 to +1.0


@dataclass
class DocumentImage:
    path: Path
    original: np.ndarray                          # BGR uint8
    corners: Optional[List[Tuple[int, int]]] = None  # [TL, TR, BR, BL]
    filter_settings: FilterSettings = field(default_factory=FilterSettings)
    processed: Optional[np.ndarray] = None
    thumbnail: Optional[np.ndarray] = None
    dewarp_enabled: bool = True
    sharpen_enabled: bool = True
    needs_reprocess: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.filter_settings, FilterSettings):
            self.filter_settings = FilterSettings(**self.filter_settings)
        if self.corners is None:
            h, w = self.original.shape[:2]
            self.corners = [(0, 0), (w, 0), (w, h), (0, h)]

    @property
    def display_name(self) -> str:
        return self.path.name
