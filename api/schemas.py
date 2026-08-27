from __future__ import annotations

from pydantic import BaseModel


class QueryIn(BaseModel):
    query: str
    top_k: int = 10


class Condicao(BaseModel):
    campo: str
    operador: str
    valor: float


class QueryOut(BaseModel):
    query: str
    tipo: str
    retriever: str
    data_inicio: str | None
    data_fim: str | None
    periodo: str | None
    condicoes: list[Condicao]
    resultados: list[dict]


class AnswerOut(BaseModel):
    query: str
    resposta: str