import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.agentic_engine import AgenticCodeEngine


# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------

app = FastAPI(
    title="PRISM Agentic Code Intelligence",
    version="1.0.0",
    description=(
        "Local agentic code retrieval engine with semantic, "
        "structural, exact-usage and evolutionary retrieval."
    ),
)


# ------------------------------------------------------------------
# CORS
#
# Kept permissive for local hackathon UI development.
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# ENGINE
#
# AgenticCodeEngine itself is lightweight because individual
# retrieval models are lazy-loaded only when their route is used.
# ------------------------------------------------------------------

VERSION_INDEX_ROOT = os.environ.get(
    "PRISM_VERSION_INDEX",
    "runtime_index/versioned_semantic_test",
)


engine = AgenticCodeEngine(
    index_dir="runtime_index",
    device="mps",
    enable_reranker=True,
    version_index_root=VERSION_INDEX_ROOT,
)


SERVER_START = time.time()


# ------------------------------------------------------------------
# REQUEST / RESPONSE
# ------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=20000,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )


# ------------------------------------------------------------------
# DEMO UI
# ------------------------------------------------------------------

@app.get("/")
def demo_ui():
    return FileResponse(
        "web/index.html"
    )



# ------------------------------------------------------------------
# HEALTH
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "PRISM Agentic Code Intelligence",
        "version": "1.0.0",
        "uptime_seconds": (
            time.time() - SERVER_START
        ),
        "routes": [
            "semantic",
            "exact_usage",
            "structural",
            "evolution",
        ],
        "models_loaded": {
            "semantic":
                engine.semantic_engine
                is not None,

            "structural":
                engine.structural_engine
                is not None,

            "evolution":
                engine.version_engine
                is not None,
        },
    }


# ------------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------------

@app.post("/search")
def search(request: SearchRequest):
    try:
        return engine.search(
            request.query,
            top_k=request.top_k,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
