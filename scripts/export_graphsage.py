from __future__ import annotations
import json

from graph.graph import CAMPOS_NUMERICOS, GraphManager
from IA.graphsage.dataset import EXPORT_DIR
from utils.config import settings


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    gm = GraphManager(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )

    rows = gm.fetch_eventos_export()

    eid_to_idx: dict[str, int] = {}
    nodes_path = EXPORT_DIR / "nodes_Evento.jsonl"
    with nodes_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            props = dict(row["props"])
            record = {
                "idx": idx,
                "id": row["id"],
                "data": row["data"],
                "periodo": row["periodo"],
                **{campo: props.get(campo) for campo in CAMPOS_NUMERICOS},
            }
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            eid_to_idx[row["eid"]] = idx

    print(f"[export_graphsage] Evento: {len(rows)} nós -> {nodes_path}")

    edge_rows = gm.fetch_proximo_dia_edges()
    edges_path = EXPORT_DIR / "edges_Evento__PROXIMO_DIA__Evento.jsonl"
    skipped = 0
    count = 0
    with edges_path.open("w", encoding="utf-8") as f:
        for row in edge_rows:
            src_idx = eid_to_idx.get(row["src_eid"])
            dst_idx = eid_to_idx.get(row["dst_eid"])
            if src_idx is None or dst_idx is None:
                skipped += 1
                continue
            f.write(json.dumps({"src": src_idx, "dst": dst_idx}, ensure_ascii=False) + "\n")
            count += 1

    if skipped:
        print(f"[export_graphsage] PROXIMO_DIA: {skipped} arestas ignoradas (nó fora do export)")
    print(f"[export_graphsage] Evento-[PROXIMO_DIA]->Evento: {count} arestas -> {edges_path}")

    gm.close()

    manifest = {"nodes": {"Evento": len(rows)}, "edges": {"Evento__PROXIMO_DIA__Evento": count}}
    manifest_path = EXPORT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[export_graphsage] Concluído. Manifesto em {manifest_path}")


if __name__ == "__main__":
    main()
