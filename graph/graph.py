from neo4j import GraphDatabase

CONSTRAINTS = [
    "CREATE CONSTRAINT data_valor IF NOT EXISTS FOR (d:Data) REQUIRE d.data IS UNIQUE",
    "CREATE CONSTRAINT evento_id IF NOT EXISTS FOR (e:Evento) REQUIRE e.id IS UNIQUE",
]

MERGE_EVENTOS = """
UNWIND $rows AS row
MERGE (d:Data {data: row.data})
MERGE (e:Evento {id: row.id})
SET e += row.props
MERGE (e)-[:NA_DATA]->(d)
"""

FETCH_EVENTOS_SEM_EMBEDDING = """
MATCH (e:Evento)
WHERE e.embedding IS NULL
RETURN e.id AS id, e AS props
"""

FETCH_TODOS_EVENTOS = """
MATCH (e:Evento)
RETURN e.id AS id, e AS props
"""

SET_EMBEDDING = """
UNWIND $rows AS row
MATCH (e:Evento {id: row.id})
SET e.embedding = row.embedding,
    e.embedding_text = row.embedding_text
"""

SEARCH_SEMANTICO = """
MATCH (e:Evento)-[:NA_DATA]->(d:Data)
WHERE e.embedding IS NOT NULL
  AND ($data_inicio IS NULL OR d.data >= $data_inicio)
  AND ($data_fim IS NULL OR d.data <= $data_fim)
  AND ($periodo IS NULL OR e.periodo = $periodo)
WITH e, vector.similarity.cosine(e.embedding, $embedding) AS score
ORDER BY score DESC
LIMIT $top_k
RETURN e AS props, score
"""

CAMPOS_NUMERICOS = {
    "ocupados", "totuh", "pct_uh_disp", "pct_uh_total", "entradas", "saidas",
    "cort", "uso_casa", "permutas", "indisponiveis", "disponiveis", "pax",
    "chd", "receita", "diaria_media", "acrescimo", "revpar", "desconto", "bloqueada",
}

OPERADORES_CYPHER = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "=": "="}

def _to_float(valor):
    if valor is None or isinstance(valor, float):
        return valor
    return float(valor)

BUILD_PROXIMO_DIA = """
MATCH (e1:Evento)-[:NA_DATA]->(d1:Data)
MATCH (e2:Evento)-[:NA_DATA]->(d2:Data)
WHERE e1.periodo = e2.periodo
  AND date(d2.data) = date(d1.data) + duration('P1D')
MERGE (e1)-[:PROXIMO_DIA]->(e2)
RETURN count(*) AS total
"""

# Evento não guarda data/periodo como propriedade própria (só via NA_DATA/NO_PERIODO
# acima) — por isso o join com Data/Periodo aqui.
FETCH_EVENTOS_EXPORT = """
MATCH (e:Evento)-[:NA_DATA]->(d:Data)
RETURN elementId(e) AS eid, e.id AS id, d.data AS data, e.periodo AS periodo, properties(e) AS props
ORDER BY d.data
"""

FETCH_PROXIMO_DIA = """
MATCH (e1:Evento)-[:PROXIMO_DIA]->(e2:Evento)
RETURN elementId(e1) AS src_eid, elementId(e2) AS dst_eid
"""

SET_EMBEDDING_ESTRUTURAL = """
UNWIND $rows AS row
MATCH (e:Evento {id: row.id})
SET e.embedding_estrutural = row.embedding,
    e.embedding_estrutural_updated_at = row.updated_at
"""


class GraphManager:
    def __init__(self, uri, user, password, database=None):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self):
        self._driver.close()

    def setup_constraints(self):
        with self._driver.session(database=self._database) as session:
            for stmt in CONSTRAINTS:
                session.run(stmt)

    def persist_eventos(self, records, batch_size=1000):
        rows = [self._build_row(r) for r in records]
        with self._driver.session(database=self._database) as session:
            for i in range(0, len(rows), batch_size):
                session.run(MERGE_EVENTOS, rows=rows[i:i + batch_size])

    @staticmethod
    def _build_row(record):
        data = record["Data"]
        periodo = record["Periodo"]
        props = {
            "data": data,
            "periodo": periodo,
            "pct_uh_disp": _to_float(record.get("% / UH Disp")),
            "pct_uh_total": _to_float(record.get("% / UH Total")),
            "totuh": record.get("Total"),
            "ocupados": record.get("Ocupados"),
            "entradas": record.get("Entrada"),
            "saidas": record.get("Saidas"),
            "cort": record.get("Cort"),
            "uso_casa": record.get("Uso da Casa"),
            "permutas": record.get("Permutas"),
            "indisponiveis": record.get("Indisponiveis"),
            "disponiveis": record.get("Disponiveis"),
            "pax": record.get("# Pax"),
            "chd": record.get("# CHD"),
            "receita": record.get("Receita de Hospedagem"),
            "diaria_media": _to_float(record.get("Diaria Media")),
            "acrescimo": record.get("Acrescimo"),
            "revpar": _to_float(record.get("RevPar")),
            "desconto": record.get("Desconto"),
            "bloqueada": record.get("Bloqueada"),
        }
        return {
            "id": data,
            "data": data,
            "periodo": periodo,
            "props": props,
        }

    def setup_vector_index(self, dims: int):
        stmt = f"""
        CREATE VECTOR INDEX evento_embedding IF NOT EXISTS
        FOR (e:Evento) ON (e.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {dims},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        with self._driver.session(database=self._database) as session:
            session.run(stmt)

    def fetch_eventos(self, force: bool = False):
        query = FETCH_TODOS_EVENTOS if force else FETCH_EVENTOS_SEM_EMBEDDING
        with self._driver.session(database=self._database) as session:
            return [dict(r) for r in session.run(query)]

    def persist_embeddings(self, rows):
        with self._driver.session(database=self._database) as session:
            session.run(SET_EMBEDDING, rows=rows)

    def search_semantico(self, embedding, top_k=10, data_inicio=None, data_fim=None, periodo=None):
        with self._driver.session(database=self._database) as session:
            result = session.run(
                SEARCH_SEMANTICO,
                embedding=embedding,
                top_k=top_k,
                data_inicio=data_inicio,
                data_fim=data_fim,
                periodo=periodo,
            )
            return [{**dict(record["props"]), "score": record["score"]} for record in result]

    def search_eventos(
        self,
        data_inicio=None,
        data_fim=None,
        periodo=None,
        condicoes=None,
        embedding=None,
        top_k=10,
    ):
        where_partes = [
            "($data_inicio IS NULL OR d.data >= $data_inicio)",
            "($data_fim IS NULL OR d.data <= $data_fim)",
            "($periodo IS NULL OR e.periodo = $periodo)",
        ]
        params = {
            "data_inicio": data_inicio,
            "data_fim": data_fim,
            "periodo": periodo,
            "top_k": top_k,
        }

        for i, cond in enumerate(condicoes or []):
            campo = cond.get("campo")
            operador = cond.get("operador")
            valor = cond.get("valor")
            if campo not in CAMPOS_NUMERICOS:
                raise ValueError(f"Campo inválido: {campo}")
            if operador not in OPERADORES_CYPHER:
                raise ValueError(f"Operador inválido: {operador}")
            pname = f"valor{i}"
            where_partes.append(f"e.{campo} {OPERADORES_CYPHER[operador]} ${pname}")
            params[pname] = valor

        where_clause = " AND ".join(where_partes)

        with self._driver.session(database=self._database) as session:
            if embedding is not None:
                params["embedding"] = embedding
                cypher = f"""
                MATCH (e:Evento)-[:NA_DATA]->(d:Data)
                WHERE e.embedding IS NOT NULL AND {where_clause}
                WITH e, vector.similarity.cosine(e.embedding, $embedding) AS score
                ORDER BY score DESC
                LIMIT $top_k
                RETURN e AS props, score
                """
                result = session.run(cypher, **params)
                return [{**dict(record["props"]), "score": record["score"]} for record in result]

            cypher = f"""
            MATCH (e:Evento)-[:NA_DATA]->(d:Data)
            WHERE {where_clause}
            RETURN e AS props
            ORDER BY d.data DESC
            LIMIT $top_k
            """
            result = session.run(cypher, **params)
            return [dict(record["props"]) for record in result]

    def build_proximo_dia_edges(self) -> int:
        with self._driver.session(database=self._database) as session:
            return session.run(BUILD_PROXIMO_DIA).single()["total"]

    def fetch_eventos_export(self):
        with self._driver.session(database=self._database) as session:
            return [dict(r) for r in session.run(FETCH_EVENTOS_EXPORT)]

    def fetch_proximo_dia_edges(self):
        with self._driver.session(database=self._database) as session:
            return [dict(r) for r in session.run(FETCH_PROXIMO_DIA)]

    def setup_structural_vector_index(self, dims: int):
        stmt = f"""
        CREATE VECTOR INDEX evento_embedding_estrutural IF NOT EXISTS
        FOR (e:Evento) ON (e.embedding_estrutural)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {dims},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        with self._driver.session(database=self._database) as session:
            session.run(stmt)

    def fetch_eventos_com_embedding_estrutural(self):
        with self._driver.session(database=self._database) as session:
            return {
                r["id"] for r in session.run(
                    "MATCH (e:Evento) WHERE e.embedding_estrutural IS NOT NULL RETURN e.id AS id"
                )
            }

    def persist_structural_embeddings(self, rows):
        with self._driver.session(database=self._database) as session:
            session.run(SET_EMBEDDING_ESTRUTURAL, rows=rows)
