from __future__ import annotations
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from IA.graphsage.features import build_evento_features

EXPORT_DIR = Path("data/graph_export")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_nodes() -> list[dict]:
    nodes = _load_jsonl(EXPORT_DIR / "nodes_Evento.jsonl")
    nodes.sort(key=lambda r: r["idx"])
    return nodes


def load_edges() -> list[dict]:
    path = EXPORT_DIR / "edges_Evento__PROXIMO_DIA__Evento.jsonl"
    return _load_jsonl(path) if path.exists() else []


def _symmetrize(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def build_data(mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
    nodes = load_nodes()
    edges = load_edges()

    periodos = sorted({n.get("periodo") for n in nodes if n.get("periodo")})
    x, feature_names, mean, std = build_evento_features(nodes, periodos, mean=mean, std=std)

    if edges:
        src = torch.tensor([e["src"] for e in edges], dtype=torch.long)
        dst = torch.tensor([e["dst"] for e in edges], dtype=torch.long)
        edge_index = _symmetrize(torch.stack([src, dst], dim=0))
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    event_ids = [n["id"] for n in nodes]
    return data, event_ids, feature_names, periodos, mean, std
