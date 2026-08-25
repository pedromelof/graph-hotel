from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from utils.config import settings

app = FastAPI(
    title="Grafo Hotel API",
    description="Busca temporal e semântica sobre eventos de ocupação hoteleira no Neo4j.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1", tags=["Eventos"])