from __future__ import annotations

from graph.graph import GraphManager
from utils.config import settings


def main() -> None:
    gm = GraphManager(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    total = gm.build_proximo_dia_edges()
    gm.close()
    print(f"OK: {total} arestas :PROXIMO_DIA criadas/confirmadas (mesmo Periodo, um dia consecutivo).")


if __name__ == "__main__":
    main()
