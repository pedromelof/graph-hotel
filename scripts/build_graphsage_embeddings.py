from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from graph.graph import GraphManager
from IA.graphsage.dataset import build_data
from IA.graphsage.model import GraphSAGELinkPredictionModel
from utils.config import settings

MODEL_DIR = Path(settings.graphsage_model_path)
BATCH_SIZE = 200


def _load_config() -> dict:
    with (MODEL_DIR / "config.json").open(encoding="utf-8") as f:
        return json.load(f)


def main(force: bool = False) -> None:
    config = _load_config()
    mean = torch.tensor(config["feature_mean"]).unsqueeze(0)
    std = torch.tensor(config["feature_std"]).unsqueeze(0)

    data, event_ids, feature_names, periodos, _mean, _std = build_data(mean=mean, std=std)

    if feature_names != config["feature_names"] or periodos != config["periodos"]:
        raise RuntimeError(
            "O schema de features mudou desde o treino — rode scripts/train_graphsage.py de novo antes de gerar embeddings.\n"
            f"  treino: features={config['feature_names']} periodos={config['periodos']}\n"
            f"  atual:  features={feature_names} periodos={periodos}"
        )

    model = GraphSAGELinkPredictionModel(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        out_channels=config["out_channels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
    model.eval()

    print(f"[build_graphsage_embeddings] Rodando o encoder (eval) sobre {data.x.shape[0]} Eventos — checkpoint val_auc={config['best_val_auc']:.4f}")
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)

    gm = GraphManager(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    gm.setup_structural_vector_index(z.shape[1])

    already_done: set[str] = set() if force else gm.fetch_eventos_com_embedding_estrutural()

    items = [
        {"id": event_id, "embedding": z[i].tolist(), "updated_at": now}
        for i, event_id in enumerate(event_ids)
        if event_id not in already_done
    ]

    if not items:
        print("[build_graphsage_embeddings] Nenhum evento novo para gravar (use --force para regerar todos).")
        gm.close()
        return

    print(f"[build_graphsage_embeddings] Gravando {len(items)} embeddings estruturais (e.embedding_estrutural) no Neo4j...")
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        gm.persist_structural_embeddings(batch)
        print(f"  {min(i + BATCH_SIZE, len(items))}/{len(items)} gravados...")

    gm.close()
    print("OK: embeddings estruturais (GraphSAGE) gravados no Neo4j")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roda o encoder GraphSAGE treinado (eval) e grava e.embedding_estrutural no Neo4j")
    parser.add_argument("--force", action="store_true", help="Regrava embeddings mesmo para eventos que já têm embedding_estrutural")
    args = parser.parse_args()
    main(force=args.force)
