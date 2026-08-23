from __future__ import annotations

import re


def llm_generate(prompt: str) -> str:
    from openai import OpenAI
    from utils.config import settings

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada.")

    client = OpenAI(api_key=settings.openai_api_key)  # timeout=<segundos> aqui para limitar a chamada
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "Você só responde com JSON válido, sem markdown e sem texto fora das chaves."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def extract_json(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None