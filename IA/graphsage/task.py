from __future__ import annotations
import random
from dataclasses import dataclass

import torch

from IA.graphsage.dataset import load_edges, load_nodes

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
NEG_PER_POS = 1


@dataclass
class LinkPredictionSplit:
    edge_index: torch.Tensor
    edge_label_index: torch.Tensor
    edge_label: torch.Tensor


def _pair_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _to_edge_index(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _symmetrize(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def build_link_prediction_splits(seed: int = 42) -> dict[str, LinkPredictionSplit]:
    nodes = load_nodes()
    dates = [n.get("data") or "1970-01-01" for n in nodes]

    edges = load_edges()
    pairs = [(e["src"], e["dst"]) for e in edges]
    timestamps = [dates[dst] for _src, dst in pairs]

    order = sorted(range(len(pairs)), key=lambda i: timestamps[i])
    n = len(order)
    train_cut = int(n * TRAIN_FRAC)
    val_cut = int(n * (TRAIN_FRAC + VAL_FRAC))

    train_idx, val_idx, test_idx = order[:train_cut], order[train_cut:val_cut], order[val_cut:]
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]

    existing = {_pair_key(u, v) for u, v in pairs}
    num_nodes = len(nodes)
    rng = random.Random(seed)

    def _sample_negatives(count: int) -> list[tuple[int, int]]:
        negs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        while len(negs) < count and num_nodes > 1:
            u, v = rng.randrange(num_nodes), rng.randrange(num_nodes)
            if u == v:
                continue
            key = _pair_key(u, v)
            if key in existing or key in seen:
                continue
            seen.add(key)
            negs.append((u, v))
        return negs

    def _make_split(mp_pairs: list[tuple[int, int]], label_pairs: list[tuple[int, int]]) -> LinkPredictionSplit:
        negs = _sample_negatives(len(label_pairs) * NEG_PER_POS)
        pos_t = _to_edge_index(label_pairs)
        neg_t = _to_edge_index(negs)
        edge_label_index = torch.cat([pos_t, neg_t], dim=1)
        edge_label = torch.cat([torch.ones(pos_t.shape[1]), torch.zeros(neg_t.shape[1])])
        return LinkPredictionSplit(
            edge_index=_symmetrize(_to_edge_index(mp_pairs)),
            edge_label_index=edge_label_index,
            edge_label=edge_label,
        )

    return {
        "train": _make_split(train_pairs, train_pairs),
        "val": _make_split(train_pairs, val_pairs),
        "test": _make_split(train_pairs + val_pairs, test_pairs),
    }
