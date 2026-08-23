from __future__ import annotations

_OPENAI_DIMS: dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def get_dimensions() -> int:
    from utils.config import settings
    dims = _OPENAI_DIMS.get(settings.embedding_model)
    if dims:
        return dims
    return len(get_embeddings(["test"])[0])


def get_embeddings(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    from utils.config import settings
    client = OpenAI(api_key=settings.openai_api_key)  # timeout=<segundos> aqui para limitar a chamada de embeddings
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def make_embedder():
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
    from utils.config import settings
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def build_text(props: dict) -> str:
    return (
        f"Data {props.get('data')}, periodo {props.get('periodo')}: "
        f"ocupacao {props.get('ocupados')} de {props.get('totuh')} UHs "
        f"({props.get('pct_uh_total')}% do total, {props.get('pct_uh_disp')}% das disponiveis). "
        f"Entradas {props.get('entradas')}, saidas {props.get('saidas')}, "
        f"indisponiveis {props.get('indisponiveis')}, disponiveis {props.get('disponiveis')}, "
        f"bloqueadas {props.get('bloqueada')}. "
        f"Cortesias {props.get('cort')}, uso da casa {props.get('uso_casa')}, permutas {props.get('permutas')}. "
        f"Hospedes {props.get('pax')} (sendo {props.get('chd')} criancas). "
        f"Receita de hospedagem R$ {props.get('receita')}, diaria media R$ {props.get('diaria_media')}, "
        f"RevPar R$ {props.get('revpar')}, acrescimo R$ {props.get('acrescimo')}, desconto R$ {props.get('desconto')}."
    )