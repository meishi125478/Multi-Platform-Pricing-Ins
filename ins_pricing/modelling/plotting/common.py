from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-8


@dataclass(frozen=True)
class PlotStyle:
    figsize: Tuple[float, float] = (8.0, 4.5)
    dpi: int = 300
    palette: Sequence[str] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    grid: bool = True
    grid_alpha: float = 0.3
    grid_style: str = "--"
    title_size: int = 10
    label_size: int = 8
    tick_size: int = 7
    legend_size: int = 7
    weight_color: str = "seagreen"


def ensure_parent_dir(path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)


def finalize_figure(
    fig: plt.Figure,
    *,
    save_path: Optional[str] = None,
    show: bool = False,
    close: bool = True,
    style: Optional[PlotStyle] = None,
) -> None:
    if save_path:
        ensure_parent_dir(save_path)
        dpi = style.dpi if style else 300
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)
