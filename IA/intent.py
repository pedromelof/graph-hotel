from __future__ import annotations

import json
import re
from enum import Enum


class IntentType(str, Enum):
    TEMPORAL = "temporal"
    ESTRUTURAL = "estrutural"
    SEMANTICA = "semantica"
    HIBRIDA = "hibrida"


# ─── Etapa 1: classificação por regex (sem LLM) ────────────────────────────────

_SINAIS_TEMPORAIS = [
    r"\bhoje\b", r"\bontem\b", r"\bsemana\b", r"\bm[êe]s\b", r"\bano\b",
    r"\bjaneiro\b", r"\bfevereiro\b", r"\bmar[çc]o\b", r"\babril\b",
    r"\bmaio\b", r"\bjunho\b", r"\bjulho\b", r"\bagosto\b",
    r"\bsetembro\b", r"\boutubro\b", r"\bnovembro\b", r"\bdezembro\b",
    r"\b20\d{2}\b", r"\bentre\b", r"\b[uú]ltimos\b", r"\bdesde\b", r"\bat[ée]\b",
    r"\bmanh[ãa]\b", r"\btarde\b", r"\bnoite\b", r"\bper[íi]odo\b", r"\bturno\b",
]

_SINAIS_ESTRUTURAIS = [
    r"\bmaior\s+que\b", r"\bmenor\s+que\b", r"\bmaior\s+ou\s+igual\b",
    r"\bmenor\s+ou\s+igual\b", r"\bacima\s+de\b", r"\babaixo\s+de\b",
    r"\bsuperior\s+a\b", r"\binferior\s+a\b", r"\bpelo\s+menos\b",
    r"\bno\s+m[íi]nimo\b", r"\bno\s+m[áa]ximo\b", r"\bmais\s+de\b",
    r"\bmenos\s+de\b", r">=", r"<=", r">", r"<",
]

_SINAIS_SEMANTICOS = [
    "similar", "parecido", "parecida", "relacionado", "como foi",
    "compare", "comparar", "tendência", "tendencia", "padrão", "padrao",
    "por que", "porque", "explique", "resuma", "analise", "análise",
]

_CAMPO_PATTERNS: list[tuple[str, str]] = [
    ("pct_uh_total", r"taxa\s+de\s+ocupa[çc][ãa]o|percentual\s+de\s+ocupa[çc][ãa]o|%\s*(de\s*)?ocupa[çc][ãa]o(\s+total)?"),
    ("pct_uh_disp", r"taxa\s+de\s+dispon[íi]ve(l|is)?|percentual\s+de\s+dispon[íi]ve(l|is)?|%\s*(de\s*)?dispon[íi]ve(l|is)?"),
    ("diaria_media", r"di[áa]ria\s+m[ée]dia"),
    ("revpar", r"\brevpar\b"),
    ("receita", r"\breceita\b|\bfaturamento\b"),
    ("entradas", r"\bcheck-?in\b|\bentradas?\b"),
    ("saidas", r"\bcheck-?out\b|\bsa[íi]das?\b"),
    ("bloqueada", r"\bbloquead[ao]s?\b"),
    ("indisponiveis", r"\bindispon[íi]ve(l|is)?\b"),
    ("disponiveis", r"\bdispon[íi]ve(l|is)?\b"),
    ("pax", r"\bh[óo]spedes?\b|\bpax\b|\badultos?\b"),
    ("chd", r"\bcrian[çc]as?\b|\bchd\b"),
    ("cort", r"\bcortesias?\b"),
    ("uso_casa", r"\buso\s+da\s+casa\b|\binternos?\b"),
    ("permutas", r"\bpermutas?\b"),
    ("acrescimo", r"\bacr[ée]scimos?\b"),
    ("desconto", r"\bdescontos?\b"),
    ("totuh", r"\btotal\s+de\s+(uhs?|apartamentos?)\b"),
    ("ocupados", r"\bocupados\b|\bocupa[çc][ãa]o\b"),
]

_AGREGACAO_PATTERNS: list[tuple[str, str]] = [
    ("avg", r"m[ée]dia|m[ée]dio"),
    ("sum", r"\bsoma\b|\bsomat[óo]rio\b|\btotal\b"),
    ("count", r"\bquantos\b|\bquantas\b|\bquantidade\b|\bn[úu]mero\s+de\b"),
    ("max", r"\bm[áa]ximo\b|\bmaior\s+valor\b|\bpico\b"),
    ("min", r"\bm[íi]nimo\b|\bmenor\s+valor\b"),
]


def detectar_agregacoes(texto: str) -> list[str]:
    q = texto.lower()
    funcoes = []
    for funcao, pattern in _AGREGACAO_PATTERNS:
        if re.search(pattern, q) and funcao not in funcoes:
            funcoes.append(funcao)
    return funcoes


def detectar_campos(texto: str) -> list[str]:
    q = texto.lower()
    campos = []
    for campo, pattern in _CAMPO_PATTERNS:
        if re.search(pattern, q) and campo not in campos:
            campos.append(campo)
    return campos


def analisar_intencao(pergunta: str) -> dict:
    q = pergunta.lower()

    tem_temporal = any(re.search(p, q) for p in _SINAIS_TEMPORAIS)
    tem_estrutural = any(re.search(p, q) for p in _SINAIS_ESTRUTURAIS)
    tem_semantica = any(t in q for t in _SINAIS_SEMANTICOS)

    if sum([tem_temporal, tem_estrutural, tem_semantica]) >= 2:
        tipo = IntentType.HIBRIDA
    elif tem_temporal:
        tipo = IntentType.TEMPORAL
    elif tem_estrutural:
        tipo = IntentType.ESTRUTURAL
    else:
        tipo = IntentType.SEMANTICA
        tem_semantica = True

    return {
        "tipo": tipo,
        "tem_temporal": tem_temporal,
        "tem_estrutural": tem_estrutural,
        "tem_semantica": tem_semantica,
    }


def escolher_retriever(pergunta: str) -> str:
    if detectar_agregacoes(pergunta):
        return "cypher_agregado"

    intent = analisar_intencao(pergunta)
    if intent["tipo"] == IntentType.TEMPORAL:
        return "cypher_temporal"
    if intent["tipo"] == IntentType.ESTRUTURAL:
        return "cypher_estrutural"
    if intent["tipo"] == IntentType.SEMANTICA:
        return "vector_retriever"
    return "cypher_hibrido_com_embedding"


# ─── Etapa 2: extração de parâmetros via LLM, já sabendo o retriever ───────────

EXTRACTION_PROMPT = """\
Você extrai parâmetros de busca de uma pergunta sobre ocupação hoteleira, \
para consultar um grafo Neo4j com nós `Evento` ligados a `Data` (propriedade `data`, \
YYYY-MM-DD). `periodo` NÃO é um nó, é uma propriedade simples de `Evento` \
(e.periodo) — não é um campo prioritário do domínio, então só preencha se o \
usuário citar um valor de período explicitamente; na dúvida, deixe null.

`Evento` tem estes campos numéricos (use EXATAMENTE este nome no campo "campo"):
- ocupados: quantidade de UHs ocupadas
- totuh: total de UHs
- pct_uh_disp: % de ocupação sobre UHs disponíveis
- pct_uh_total: % de ocupação sobre o total de UHs
- entradas: quantidade de check-ins
- saidas: quantidade de check-outs
- cort: quantidade de cortesias
- uso_casa: quantidade de UHs em uso da casa
- permutas: quantidade de permutas
- indisponiveis: quantidade de UHs indisponíveis
- disponiveis: quantidade de UHs disponíveis
- pax: quantidade de hóspedes
- chd: quantidade de crianças
- receita: receita de hospedagem (R$)
- diaria_media: diária média (R$)
- acrescimo: acréscimo (R$)
- revpar: RevPar (R$)
- desconto: desconto (R$)
- bloqueada: quantidade de UHs bloqueadas

Data de referência para resolver expressões relativas ("hoje", "essa semana", "mês passado"): {data_referencia}

Tipo de busca já identificado (por regex, antes desta etapa): **{retriever_type}**

Campo(s) numérico(s) já identificado(s) por regex nesta pergunta (se houver): {campo_sugerido}. \
Se a pergunta tiver comparação(ões) numérica(s), use ESSES nomes de campo exatamente como vieram \
(um por condição), não invente outro nem traduza.

Função(ões) de agregação já identificada(s) por regex nesta pergunta (se houver): {funcao_sugerida}. \
Se a pergunta pedir média/soma/contagem/máximo/mínimo de algum campo, preencha "agregacoes" \
usando ESSAS funções (nomes exatamente "avg", "sum", "count", "max" ou "min") combinadas com \
o campo numérico correspondente da lista acima.

Extraia:
- data_inicio / data_fim (YYYY-MM-DD, resolvendo expressões relativas pela data de referência; null se não houver recorte de data)
- periodo (valor exato de periodo mencionado pelo usuário; null se não mencionado — não é prioritário, não invente)
- condicoes: lista de comparações numéricas encontradas na pergunta, cada uma como \
{{"campo": "<um dos nomes acima>", "operador": "> | >= | < | <= | =", "valor": <número>}}. \
A pergunta pode ter MAIS DE UMA comparação (ex: "receita maior que 50000 e ocupados maior \
que 70" gera DUAS condições na lista). Lista vazia se não houver comparação numérica. \
Todas as condições da lista são combinadas com E (AND).
- agregacoes: lista de agregações pedidas, cada uma como \
{{"funcao": "avg | sum | count | max | min", "campo": "<um dos nomes acima>"}}. \
Lista vazia se a pergunta não pedir nenhuma agregação.
- semantic_query (texto curto em português com os termos descritivos da pergunta, para busca \
por similaridade; preencha quando a pergunta pedir algo descritivo/comparativo/de tendência; senão null)

Preencha só o que fizer sentido pra pergunta — não invente data, período, comparação \
numérica ou agregação que não estejam implícitos no texto.

Retorne APENAS o JSON abaixo, sem markdown, sem texto fora das chaves:
{{
  "data_inicio": "YYYY-MM-DD ou null",
  "data_fim": "YYYY-MM-DD ou null",
  "periodo": "string ou null",
  "condicoes": [
    {{"campo": "string", "operador": "> | >= | < | <= | =", "valor": 0}}
  ],
  "agregacoes": [
    {{"funcao": "avg | sum | count | max | min", "campo": "string"}}
  ],
  "semantic_query": "string ou null"
}}

Pergunta do usuário:
"{query}"
"""


def resolver_parametros(query: str, retriever_type: str) -> dict:
    from datetime import date

    from IA.llm import extract_json, llm_generate

    campos_sugeridos = detectar_campos(query)
    funcoes_sugeridas = detectar_agregacoes(query)

    prompt = EXTRACTION_PROMPT.format(
        query=query,
        retriever_type=retriever_type,
        data_referencia=date.today().isoformat(),
        campo_sugerido=", ".join(campos_sugeridos) if campos_sugeridos else "nenhum campo óbvio identificado",
        funcao_sugerida=", ".join(funcoes_sugeridas) if funcoes_sugeridas else "nenhuma agregação óbvia identificada",
    )
    raw = llm_generate(prompt)
    json_str = extract_json(raw)

    vazio = {
        "data_inicio": None, "data_fim": None, "periodo": None,
        "condicoes": [], "agregacoes": [], "semantic_query": query,
    }
    if not json_str:
        return vazio

    parsed = json.loads(json_str)

    condicoes = []
    for cond in parsed.get("condicoes") or []:
        if not isinstance(cond, dict):
            continue
        campo, operador, valor = cond.get("campo"), cond.get("operador"), cond.get("valor")
        if campo and operador and valor is not None:
            condicoes.append({"campo": campo, "operador": operador, "valor": valor})

    agregacoes = []
    for agg in parsed.get("agregacoes") or []:
        if not isinstance(agg, dict):
            continue
        funcao, campo = agg.get("funcao"), agg.get("campo")
        if funcao and campo:
            agregacoes.append({"funcao": funcao, "campo": campo})

    return {
        "data_inicio": parsed.get("data_inicio") or None,
        "data_fim": parsed.get("data_fim") or None,
        "periodo": parsed.get("periodo") or None,
        "condicoes": condicoes,
        "agregacoes": agregacoes,
        "semantic_query": parsed.get("semantic_query") or None,
    }