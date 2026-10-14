"""The `frontends.http` plugin: the app served by uvicorn in the frontend process."""
from __future__ import annotations

import uvicorn
from holon.api import EngineClient, FrontendBundle, Registries

from .app import build_app


class Plugin:
    def register(self, regs: Registries) -> None:
        pass   # a frontend registers nothing: the core calls `serve`

    async def serve(self, client: EngineClient, bundle: FrontendBundle, host: str, port: int,
                    options: dict[str, object]) -> None:
        """Option `http.drop_billing_header` (default true; `false`, `0`, `no`, `off` keep the block, D7)."""
        drop = str(options.get("http.drop_billing_header", True)).lower() not in ("false", "0", "no", "off")
        config = uvicorn.Config(build_app(client, bundle, drop), host=host, port=port, lifespan="off",
                                log_level="info")
        await uvicorn.Server(config).serve()
