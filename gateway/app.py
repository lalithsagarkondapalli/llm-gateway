"""HTTP layer: OpenAI-compatible endpoints + admin/ops endpoints."""
from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .config import ApiKeyConfig, GatewayConfig, load_config
from .providers import MockProvider
from .ratelimit import RateLimiter
from .schemas import ChatCompletionRequest
from .service import GatewayService, NoHealthyModelError

log = logging.getLogger("llm_gateway")
ANONYMOUS = ApiKeyConfig(key="anonymous-no-auth", name="anonymous", rate_limit_rpm=600, admin=True)


class ChaosRequest(BaseModel):
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float | None = Field(default=None, ge=0)


def create_app(config: GatewayConfig | None = None, service: GatewayService | None = None) -> FastAPI:
    config = config or load_config()
    service = service or GatewayService(config)
    limiter = RateLimiter()
    keys = {k.key: k for k in config.auth.keys}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await service.aclose()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan,
                  description="OpenAI-compatible gateway with routing, fallback, semantic caching and metrics.")
    app.state.service = service

    def _error(status: int, message: str, err_type: str, headers: dict | None = None) -> HTTPException:
        return HTTPException(status, detail={"error": {"message": message, "type": err_type}}, headers=headers)

    async def authenticate(authorization: str | None = Header(default=None)) -> ApiKeyConfig:
        if not config.auth.enabled:
            return ANONYMOUS
        if not authorization or not authorization.lower().startswith("bearer "):
            raise _error(401, "Missing bearer token", "authentication_error")
        key = keys.get(authorization[7:].strip())
        if key is None:
            raise _error(401, "Invalid API key", "authentication_error")
        return key

    async def rate_limited(key: ApiKeyConfig = Depends(authenticate)) -> ApiKeyConfig:
        allowed, retry_after = limiter.check(key.name, key.rate_limit_rpm)
        if not allowed:
            service.metrics.rate_limited.labels(key.name).inc()
            raise _error(429, "Rate limit exceeded", "rate_limit_error",
                         headers={"Retry-After": str(math.ceil(retry_after))})
        return key

    async def admin_only(key: ApiKeyConfig = Depends(authenticate)) -> ApiKeyConfig:
        if not key.admin:
            raise _error(403, "Admin key required", "permission_error")
        return key

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        body = exc.detail if isinstance(exc.detail, dict) else {"error": {"message": str(exc.detail)}}
        return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, key: ApiKeyConfig = Depends(rate_limited),
                               x_cache_bypass: str | None = Header(default=None),
                               x_no_fallback: str | None = Header(default=None)):
        truthy = {"1", "true", "yes"}
        bypass = (x_cache_bypass or "").lower() in truthy
        fallback = (x_no_fallback or "").lower() not in truthy
        try:
            if req.stream:
                iterator, meta = await service.stream(req, key.name, bypass, fallback)
                return StreamingResponse(iterator, media_type="text/event-stream", headers=meta.headers())
            body, meta = await service.complete(req, key.name, bypass, fallback)
            return JSONResponse(body, headers=meta.headers())
        except KeyError:
            raise _error(404, f"Unknown model '{req.model}'. Use 'auto' or one of {list(service.models)}",
                         "invalid_request_error") from None
        except NoHealthyModelError as exc:
            log.warning("all models failed: %s", exc.errors)
            raise _error(503, "All upstream models failed: " + "; ".join(exc.errors),
                         "upstream_unavailable") from exc

    @app.get("/v1/models")
    async def list_models(_: ApiKeyConfig = Depends(authenticate)):
        data = [{"id": "auto", "object": "model", "owned_by": "gateway"}]
        data += [{"id": m.name, "object": "model", "owned_by": m.provider} for m in service.models.values()]
        return {"object": "list", "data": data}

    @app.get("/health")
    async def health():
        states = service.router.circuit_states()
        healthy = sum(1 for s in states.values() if s.value != "open")
        return {"status": "ok" if healthy else "degraded", "healthy_models": healthy, "total_models": len(states)}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(service.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/admin/usage")
    async def usage(_: ApiKeyConfig = Depends(admin_only)):
        return {"usage": service.usage.report(), "cache": service.cache.stats() if service.cache else None}

    @app.get("/admin/providers")
    async def providers(_: ApiKeyConfig = Depends(admin_only)):
        return {"models": service.router.snapshot()}

    @app.post("/admin/providers/{name}/chaos")
    async def chaos(name: str, body: ChaosRequest, _: ApiKeyConfig = Depends(admin_only)):
        """Fault injection for mock providers - used by the outage simulation benchmark."""
        provider = service.providers.get(name)
        if provider is None:
            raise _error(404, f"Unknown provider '{name}'", "invalid_request_error")
        if not isinstance(provider, MockProvider):
            raise _error(400, "Chaos controls only apply to mock providers", "invalid_request_error")
        if body.failure_rate is not None:
            provider.failure_rate = body.failure_rate
        if body.latency_ms is not None:
            provider.latency_ms = body.latency_ms
        return {"provider": name, "failure_rate": provider.failure_rate, "latency_ms": provider.latency_ms}

    return app
