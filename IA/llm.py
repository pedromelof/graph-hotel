from __future__ import annotations

import json
import re

_SYSTEM_JSON = "Você só responde com JSON válido, sem markdown e sem texto fora das chaves."

_SYSTEM_RESPOSTA = """\
Você é um assistente que responde em português perguntas sobre ocupação hoteleira, \
a partir de dados já consultados e calculados em um banco de grafos (Neo4j).

Responda em Markdown, escolhendo o formato mais claro para os dados recebidos: texto \
corrido para uma resposta simples/direta, tópicos para listas, tabela para séries de \
valores comparáveis. Se o usuário pedir um formato específico na pergunta (ex: "em \
tabela", "em lista"), respeite esse pedido.

Baseie-se ESTRITAMENTE nos dados fornecidos — não invente números, datas ou fatos que \
não estejam neles. Se os dados vierem vazios, diga isso claramente em vez de supor um \
resultado. Seja direto e conciso."""


def llm_generate(prompt: str, system: str | None = None) -> str:
    from openai import OpenAI
    from utils.config import settings

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada.")

    client = OpenAI(api_key=settings.openai_api_key)  # timeout=<segundos> aqui para limitar a chamada
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system or _SYSTEM_JSON},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def extract_json(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def gerar_resposta(query: str, dados) -> str:
    dados_json = json.dumps(dados, ensure_ascii=False, default=str, indent=2)

    prompt = f"""\
Pergunta do usuário:
"{query}"

Dados retornados da consulta (já filtrados/calculados, prontos para uso):
{dados_json}

Responda a pergunta do usuário em português usando esses dados.
"""
    return llm_generate(prompt, system=_SYSTEM_RESPOSTA)