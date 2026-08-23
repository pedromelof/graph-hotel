from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Grafo Hotel API",
    description="Busca temporal e semântica sobre eventos de ocupação hoteleira no Neo4j.",
    version="1.0.0",
)

app.include_router(router, prefix="/v1", tags=["Eventos"])