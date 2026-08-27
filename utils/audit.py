from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def registrar_auditoria(registro: dict) -> None:
    from utils.config import settings

    caminho = settings.audit_log_path
    diretorio = os.path.dirname(caminho)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)

    linha = {"timestamp": datetime.now(timezone.utc).isoformat(), **registro}
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(linha, ensure_ascii=False, default=str) + "\n")
