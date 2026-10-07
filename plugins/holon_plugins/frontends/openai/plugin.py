"""The `frontend.openai` plugin: the app served by uvicorn in the frontend process."""
from __future__ import annotations

import uvicorn
from holon.api import EngineClient, FrontendBundle, Registries

from .app import build_app


class Plugin:
    def register(self, regs: Registries) -> None:
        pass   # a frontend registers nothing: the core calls `serve`

    async def serve(self, client: EngineClient, bundle: FrontendBundle, host: str, port: int) -> None:
        config = uvicorn.Config(build_app(client, bundle), host=host, port=port, lifespan="off", log_level="info")
        await uvicorn.Server(config).serve()
