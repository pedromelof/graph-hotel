from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import AnswerOut, QueryIn
from graph.graph import GraphManager
from IA.embeddings import get_embeddings
from IA.intent import escolher_retriever, resolver_parametros
from IA.llm import gerar_resposta
from utils.audit import registrar_auditoria
from utils.config import settings

router = APIRouter()


def _strip_embedding(evento: dict) -> dict:
    return {k: v for k, v in evento.items() if k != "embedding"}


def _run_temporal(gm: GraphManager, params: dict, top_k: int) -> list[dict]:
    return gm.search_eventos(
        data_inicio=params["data_inicio"],
        data_fim=params["data_fim"],
        periodo=params["periodo"],
        top_k=top_k,
    )


def _run_estrutural(gm: GraphManager, params: dict, top_k: int) -> list[dict]:
    return gm.search_eventos(condicoes=params["condicoes"], top_k=top_k)


def _run_vector(gm: GraphManager, params: dict, query: str, top_k: int) -> list[dict]:
    texto = params["semantic_query"] or query
    embedding = get_embeddings([texto])[0]
    return gm.search_eventos(embedding=embedding, top_k=top_k)


def _run_hibrido(gm: GraphManager, params: dict, query: str, top_k: int) -> list[dict]:
    texto = params["semantic_query"] or query
    embedding = get_embeddings([texto])[0]
    return gm.search_eventos(
        data_inicio=params["data_inicio"],
        data_fim=params["data_fim"],
        periodo=params["periodo"],
        condicoes=params["condicoes"],
        embedding=embedding,
        top_k=top_k,
    )


def _run_agregado(gm: GraphManager, params: dict) -> list[dict]:
    return gm.agregar_eventos(
        agregacoes=params["agregacoes"],
        data_inicio=params["data_inicio"],
        data_fim=params["data_fim"],
        periodo=params["periodo"],
        condicoes=params["condicoes"],
    )


@router.post("/query", response_model=AnswerOut)
def query_eventos(payload: QueryIn) -> AnswerOut:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query não pode ser vazia")

    retriever = escolher_retriever(payload.query)
    params = resolver_parametros(payload.query, retriever)

    print(f'retriever: {retriever}')
    print(f'params: {params}')

    gm = GraphManager(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )

    try:
        if retriever == "cypher_temporal":
            dados = _run_temporal(gm, params, payload.top_k)
        elif retriever == "cypher_estrutural":
            dados = _run_estrutural(gm, params, payload.top_k)
        elif retriever == "vector_retriever":
            dados = _run_vector(gm, params, payload.query, payload.top_k)
        elif retriever == "cypher_agregado":
            dados = _run_agregado(gm, params)
        else:  # cypher_hibrido_com_embedding
            dados = _run_hibrido(gm, params, payload.query, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        gm.close()

    resultados = [_strip_embedding(e) for e in dados]

    registrar_auditoria({
        "query": payload.query,
        "retriever": retriever,
        "data_inicio": params["data_inicio"],
        "data_fim": params["data_fim"],
        "periodo": params["periodo"],
        "condicoes": params["condicoes"],
        "agregacoes": params["agregacoes"],
        "semantic_query": params["semantic_query"],
        "resultados": resultados,
    })

    resposta = gerar_resposta(payload.query, resultados)

    return AnswerOut(query=payload.query, resposta=resposta)