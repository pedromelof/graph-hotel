import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Banco PostgreSQL
    pg_host: str = os.getenv("PG_HOST", "")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_user: str = os.getenv("PG_USER", "")
    pg_password: str = os.getenv("PG_PASSWORD", "")
    pg_database: str = os.getenv("PG_DATABASE", "")

    # Neo4j for Graph Database
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str | None = os.getenv("NEO4J_DATABASE") or None

    # Embeddings (OpenAI)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    # LLM (OpenAI)
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    # Audit
    audit_log_path: str = os.getenv("AUDIT_LOG_PATH", "data/audit_log.jsonl")

    # GraphSAGE (docs/graphsage-roadmap.md)
    graphsage_model_path:  str   = os.getenv("GRAPHSAGE_MODEL_PATH", "data/models/graphsage")
    graphsage_hidden_dim:  int   = int(os.getenv("GRAPHSAGE_HIDDEN_DIM", "128"))
    graphsage_dim:         int   = int(os.getenv("GRAPHSAGE_DIM", "64"))  # dimensão do embedding final (out_channels)
    graphsage_num_layers:  int   = int(os.getenv("GRAPHSAGE_NUM_LAYERS", "2"))
    graphsage_dropout:     float = float(os.getenv("GRAPHSAGE_DROPOUT", "0.2"))
    graphsage_epochs:      int   = int(os.getenv("GRAPHSAGE_EPOCHS", "30"))
    graphsage_lr:          float = float(os.getenv("GRAPHSAGE_LR", "0.005"))
    graphsage_batch_size:  int   = int(os.getenv("GRAPHSAGE_BATCH_SIZE", "256"))
    graphsage_patience:    int   = int(os.getenv("GRAPHSAGE_PATIENCE", "5"))
    graphsage_top_k:       int   = int(os.getenv("GRAPHSAGE_TOP_K", "10"))
    graphsage_min_score:   float = float(os.getenv("GRAPHSAGE_MIN_SCORE", "0.5"))


settings = Settings()
