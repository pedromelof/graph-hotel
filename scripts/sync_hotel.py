import json
from database.extract_sql import main as extract_sql
from utils.config import settings
from graph.graph import GraphManager

SRC = "data/sql_extract.jsonl"


def main():
    # extract_sql()

    with open(SRC, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"\nTotal de eventos extraídos: {len(records)}\n")

    gm = GraphManager(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    gm.setup_constraints()
    gm.persist_eventos(records)
    gm.close()

    print("OK: grafo atualizado no Neo4j")


if __name__ == "__main__":
    main()