"""The HTTP app (D7): one FastAPI app with a router per API dialect — OpenAI chat and text completions (`openai.py`),
Anthropic Messages (`anthropic.py`), OpenAI Responses (`responses.py`) — over one chat pipeline (`pipeline.py`),
plus /v1/models, /health and /metrics."""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from holon.api import EngineClient, FrontendBundle

from . import anthropic, openai, responses
from .web import BadRequest


def build_app(client: EngineClient, bundle: FrontendBundle, drop_billing_header: bool = True) -> FastAPI:
    """`drop_billing_header`: drop Claude Code's per-request billing-header system block (D7; counted in /metrics)."""
    app = FastAPI(title="Holon")
    started = int(time.time())
    counters = {"billing_headers_dropped": 0}

    def error(request: Request, status: int, message: str) -> JSONResponse:
        if request.url.path.startswith("/v1/messages"):
            return JSONResponse(anthropic.error_body(status, message), status)
        kind = "invalid_request_error" if status == 400 else "server_error"
        return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, e: RequestValidationError) -> JSONResponse:
        return error(request, 400, "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()))

    @app.exception_handler(BadRequest)
    async def _bad(request: Request, e: BadRequest) -> JSONResponse:
        return error(request, e.status, str(e))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {}

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return await client.stats() | counters

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": bundle.model_name, "object": "model", "created": started,
                                            "owned_by": "holon"}]}

    app.include_router(openai.router(client, bundle))
    app.include_router(anthropic.router(client, bundle, drop_billing_header, counters))
    app.include_router(responses.router(client, bundle))
    return app
