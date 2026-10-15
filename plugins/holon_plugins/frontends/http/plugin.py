"""The `frontends.http` plugin: the app served by uvicorn in the frontend process."""
from __future__ import annotations

import shlex
import sys

import uvicorn
from holon.api import EngineClient, FrontendBundle, Registries

from .app import build_app, served_id


class Plugin:
    def register(self, regs: Registries) -> None:
        pass   # a frontend registers nothing: the core calls `serve`

    async def serve(self, client: EngineClient, bundle: FrontendBundle, host: str, port: int,
                    options: dict[str, object]) -> None:
        """Options (declared in the manifest): `http.drop_billing_header` (default true; `false`, `0`, `no`, `off`
        keep the block, D7) and `http.model_name` (the served id; default `served_id` of the checkpoint's name)."""
        drop = str(options.get("http.drop_billing_header", True)).lower() not in ("false", "0", "no", "off")
        model_id = str(options.get("http.model_name") or served_id(bundle.model_name))
        config = uvicorn.Config(build_app(client, bundle, drop, model_id), host=host, port=port, lifespan="off",
                                log_level="info")
        await _Server(config, model_id).serve()


class _Server(uvicorn.Server):
    """Prints how to reach the server once it listens (release design §14.6)."""

    def __init__(self, config: uvicorn.Config, model_id: str) -> None:
        super().__init__(config)
        self.model_id = model_id

    async def startup(self, sockets=None) -> None:
        await super().startup(sockets)
        if self.started:              # the port bound (`--port 0` picks one)
            port = self.servers[0].sockets[0].getsockname()[1] if self.servers else self.config.port
            print(hint(self.config.host, port, self.model_id), file=sys.stderr, flush=True)


def hint(host: str, port: int, model_id: str) -> str:
    """The lines that point clients, and Claude Code, at this server."""
    host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else f"[{host}]" if ":" in host else host
    url = f"http://{host}:{port}"
    model = shlex.quote(model_id)
    return (f"holon: serving {model_id} at {url} (OpenAI: /v1; Anthropic: /v1/messages)\n"
            f"holon: Claude Code:\n  ANTHROPIC_BASE_URL={url} ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL={model} "
            f"ANTHROPIC_DEFAULT_HAIKU_MODEL={model} CLAUDE_CODE_SUBAGENT_MODEL={model} claude")
