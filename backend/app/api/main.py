"""FastAPI主应用（基于 04229f7，增加 Stateful Agent 路由）。"""

from fastapi import FastAPI, Request

from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings, validate_config, print_config
from .routes import trip_lg, poi_lg, map_lg, trip_stateful


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于LangGraph框架的智能旅行规划助手API",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_utf8_charset_to_json(request: Request, call_next):
    """兼容 Windows PowerShell 5.1 对 JSON 响应的中文解码。"""
    response = await call_next(request)

    content_type = response.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type == "application/json" or media_type.endswith("+json"):
        response.headers["content-type"] = (
            f"{media_type}; charset=utf-8"
        )

    return response

# 原 04229f7 路由：保持兼容
app.include_router(trip_lg.router, prefix="/api")
app.include_router(poi_lg.router, prefix="/api")
app.include_router(map_lg.router, prefix="/api")

# 新增：Supervisor + Constraint/Replan + Memory + HITL
app.include_router(trip_stateful.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """应用启动事件。"""
    print("\n" + "=" * 60)
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print("=" * 60)
    print_config()
    try:
        validate_config()
        print("\n✅ 配置验证通过")
    except ValueError as exc:
        print(f"\n❌ 配置验证失败:\n{exc}")
        raise

    print("\n" + "=" * 60)
    print("📚 API文档: http://localhost:8000/docs")
    print("🧠 增强 Agent: /api/trip-agent/*")
    print("🔧 框架: LangGraph + LangChain MCP Tools")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n👋 应用正在关闭...\n")


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "framework": "LangGraph",
        "enhanced_agent": "/api/trip-agent/plan",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "framework": "LangGraph",
        "features": ["supervisor", "constraints", "validator_replan", "memory", "hitl"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
