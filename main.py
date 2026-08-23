import sys

from scripts.sync_hotel import main as sync_hotel
from scripts.create_embedding import main as create_embedding
from scripts.build_temporal_edges import main as build_temporal_edges
from scripts.export_graphsage import main as export_graphsage
from scripts.train_graphsage import main as train_graphsage
from scripts.build_graphsage_embeddings import main as build_graphsage_embeddings


def run_step(name: str, fn, optional: bool = False) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")

    try:
        fn()
        return True
    except Exception as exc:
        print(f"\n❌ Erro em '{name}': {exc}", file=sys.stderr)
        if optional:
            print("  ⚠️  Passo opcional — continuando pipeline sem este dado.", file=sys.stderr)
            return False
        raise


def _graphsage_step(force: bool, retrain: bool) -> None:
    build_temporal_edges()
    export_graphsage()
    if retrain:
        train_graphsage()
    build_graphsage_embeddings(force=force)


def main(force: bool = False, retrain_graphsage: bool = False):
    run_step("1/3 — Sincronização PostgreSQL → Neo4j + extract_sql.jsonl", sync_hotel)
    run_step("2/3 — Geração de embeddings dos Eventos", lambda: create_embedding(force=force))
    run_step(
        "3/3 — GraphSAGE (arestas temporais + embedding estrutural)",
        lambda: _graphsage_step(force=force, retrain=retrain_graphsage),
        optional=True,
    )
    print("\n✅ Pipeline completo!")


if __name__ == "__main__":
    main(
        force="--force" in sys.argv,
        retrain_graphsage="--retrain-graphsage" in sys.argv,
    )