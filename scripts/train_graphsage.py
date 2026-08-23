from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from IA.graphsage.dataset import build_data
from IA.graphsage.model import GraphSAGELinkPredictionModel
from IA.graphsage.task import build_link_prediction_splits
from utils.config import settings

MODEL_DIR = Path(settings.graphsage_model_path)


def _run_eval(model: GraphSAGELinkPredictionModel, x: torch.Tensor, split) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x, split.edge_index, split.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, split.edge_label).item()
        probs = torch.sigmoid(logits).numpy()
        labels = split.edge_label.numpy()
    return {
        "loss": loss,
        "auc": roc_auc_score(labels, probs),
        "ap": average_precision_score(labels, probs),
    }


def main(
    epochs: int | None = None,
    hidden_channels: int | None = None,
    out_channels: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    lr: float | None = None,
    patience: int | None = None,
    seed: int = 42,
) -> None:
    epochs = epochs if epochs is not None else settings.graphsage_epochs
    hidden_channels = hidden_channels if hidden_channels is not None else settings.graphsage_hidden_dim
    out_channels = out_channels if out_channels is not None else settings.graphsage_dim
    num_layers = num_layers if num_layers is not None else settings.graphsage_num_layers
    dropout = dropout if dropout is not None else settings.graphsage_dropout
    lr = lr if lr is not None else settings.graphsage_lr
    patience = patience if patience is not None else settings.graphsage_patience

    torch.manual_seed(seed)

    data, event_ids, feature_names, periodos, mean, std = build_data()
    print(f"[train] {data.x.shape[0]} Eventos | {data.edge_index.shape[1]} arestas PROXIMO_DIA (já simetrizadas) | {data.x.shape[1]} features")

    splits = build_link_prediction_splits(seed=seed)
    for name, split in splits.items():
        num_pos = int(split.edge_label.sum().item())
        num_neg = int((split.edge_label == 0).sum().item())
        print(f"[train] split={name}: message_passing_edges={split.edge_index.shape[1]} label_pos={num_pos} label_neg={num_neg}")

    model = GraphSAGELinkPredictionModel(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = -1.0
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, splits["train"].edge_index, splits["train"].edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, splits["train"].edge_label)
        loss.backward()
        optimizer.step()

        val_metrics = _run_eval(model, data.x, splits["val"])
        print(f"  epoch {epoch:3d} | train_loss={loss.item():.4f} | val_auc={val_metrics['auc']:.4f} | val_ap={val_metrics['ap']:.4f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stopping na epoch {epoch} (sem melhora em {patience} epochs)")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    test_metrics = _run_eval(model, data.x, splits["test"])
    print(f"\nOK: treino concluído — best_val_auc={best_val_auc:.4f} test_auc={test_metrics['auc']:.4f} test_ap={test_metrics['ap']:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_DIR / "model.pt")
    with (MODEL_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "in_channels": data.x.size(1),
                "hidden_channels": hidden_channels,
                "out_channels": out_channels,
                "num_layers": num_layers,
                "dropout": dropout,
                "feature_names": feature_names,
                "periodos": periodos,
                "feature_mean": mean.squeeze(0).tolist(),
                "feature_std": std.squeeze(0).tolist(),
                "best_val_auc": best_val_auc,
                "test_auc": test_metrics["auc"],
                "test_ap": test_metrics["ap"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (MODEL_DIR / "node_index_map.json").open("w", encoding="utf-8") as f:
        json.dump(event_ids, f, ensure_ascii=False, indent=2)

    print(f"OK: modelo salvo em {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina o encoder GraphSAGE (homogêneo, Evento) por link prediction não supervisionado sobre PROXIMO_DIA"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--out-channels", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
