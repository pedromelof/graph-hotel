from __future__ import annotations
import math
from datetime import datetime

import torch

from graph.graph import CAMPOS_NUMERICOS

CAMPOS = sorted(CAMPOS_NUMERICOS)


def _cyclical_day_of_year(data_str: str | None) -> tuple[float, float]:
    if not data_str:
        return 0.0, 1.0
    dt = datetime.fromisoformat(str(data_str)[:10])
    angle = 2 * math.pi * dt.timetuple().tm_yday / 366.0
    return math.sin(angle), math.cos(angle)


def build_evento_features(
    nodes: list[dict],
    periodos: list[str],
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    rows = []
    for node in nodes:
        numeric = [float(node.get(campo)) if node.get(campo) is not None else 0.0 for campo in CAMPOS]
        day_sin, day_cos = _cyclical_day_of_year(node.get("data"))
        periodo_onehot = [1.0 if node.get("periodo") == p else 0.0 for p in periodos]
        rows.append(numeric + [day_sin, day_cos] + periodo_onehot)

    x = torch.tensor(rows, dtype=torch.float32)
    numeric_width = len(CAMPOS)

    if mean is None or std is None:
        mean = x[:, :numeric_width].mean(dim=0, keepdim=True)
        std = x[:, :numeric_width].std(dim=0, keepdim=True).clamp_min(1e-6)

    x[:, :numeric_width] = (x[:, :numeric_width] - mean) / std

    feature_names = CAMPOS + ["day_sin", "day_cos"] + [f"periodo_{p}" for p in periodos]
    return x, feature_names, mean, std
