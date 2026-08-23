import sys

from graph.graph import GraphManager
from IA.embeddings import build_text, get_dimensions, get_embeddings
from utils.config import settings

BATCH_SIZE = 100


def main(force: bool = False):
    gm = GraphManager(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )

    try:
        gm.setup_vector_index(get_dimensions())
        records = gm.fetch_eventos(force=force)
        print(f"\nEventos para processar ({'force' if force else 'apenas sem embedding'}): {len(records)}\n")

        for i in range(0, len(records), BATCH_SIZE):
            chunk = records[i:i + BATCH_SIZE]
            texts = [build_text(r["props"]) for r in chunk]
            vectors = get_embeddings(texts)

            rows = [
                {
                    "id": chunk[j]["id"],
                    "embedding": vectors[j],
                    "embedding_text": texts[j],
                }
                for j in range(len(chunk))
            ]
            gm.persist_embeddings(rows)
            print(f"  {min(i + BATCH_SIZE, len(records))}/{len(records)} embeddings gravados...")

        print("OK: embeddings gerados e gravados no Neo4j")
    finally:
        gm.close()


if __name__ == "__main__":
    main(force="--force" in sys.argv)