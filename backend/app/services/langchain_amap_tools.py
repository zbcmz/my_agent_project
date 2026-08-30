"""LangChain 兼容的高德地图服务封装

本模块是 LangGraph 重构后的新 MCP 调用层，独立于旧的 amap_service.py。
使用 langchain-mcp-adapters 官方适配器替代 hello_agents.MCPTool，
通过 Streamable HTTP 传输连接高德官方 MCP 云服务
提供 LangChainAmapService 异步服务类，统一封装所有高德地图 MCP 工具调用。
"""

import asyncio
import json
import random
import threading
from time import perf_counter
from typing import Optional, Dict, Any, List


from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import get_settings


_mcp_client: Optional[MultiServerMCPClient] = None
_mcp_tools: Optional[List[BaseTool]] = None
_mcp_async_lock: Optional[asyncio.Lock] = None
MCP_TOOL_TIMEOUT_SECONDS = 75
MCP_TOOL_MAX_ATTEMPTS = 2


def _get_async_lock() -> asyncio.Lock:
    global _mcp_async_lock
    if _mcp_async_lock is None:
        _mcp_async_lock = asyncio.Lock()
    return _mcp_async_lock


def _build_mcp_config() -> Dict[str, Any]:
    settings = get_settings()
    if not settings.amap_api_key:
        raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")
    return {
    "amap": {
        "transport": "streamable_http",
        "url": f"https://mcp.amap.com/mcp?key={settings.amap_api_key}",
        "timeout": 60,
        "sse_read_timeout": 300,
    }
}



async def _init_mcp_client() -> None:
    global _mcp_client, _mcp_tools

    if _mcp_client is not None and _mcp_tools is not None:
        return

    async_lock = _get_async_lock()
    async with async_lock:
        if _mcp_client is not None and _mcp_tools is not None:
            return

        config = _build_mcp_config()
        _mcp_client = MultiServerMCPClient(
            config,
            handle_tool_errors=False,
        )

        _mcp_tools = await _mcp_client.get_tools()

        print(f"✅ LangChain MCP 适配器初始化成功 (langchain-mcp-adapters)")
        print(f"   工具数量: {len(_mcp_tools)}")
        if _mcp_tools:
            print("   可用工具:")
            for t in _mcp_tools[:5]:
                print(f"     - {t.name}")
            if len(_mcp_tools) > 5:
                print(f"     ... 还有 {len(_mcp_tools) - 5} 个工具")


async def get_mcp_tools() -> List[BaseTool]:
    await _init_mcp_client()
    return _mcp_tools


async def get_mcp_tool_by_name(name: str) -> Optional[BaseTool]:
    tools = await get_mcp_tools()
    for t in tools:
        if t.name == name:
            return t
    return None


def _parse_result(result: Any) -> Any:
    """
    规范化 LangChain MCP 工具返回值。

    兼容：
    1. 原生 dict / list
    2. JSON 字符串
    3. LangChain MCP text content block
    4. LangChain MCP content block list
    """

    # MCP content block list
    if isinstance(result, list):
        is_content_block_list = (
            len(result) > 0
            and all(
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
                for item in result
            )
        )

        if is_content_block_list:
            parsed_blocks = [
                _parse_result(item["text"])
                for item in result
            ]

            if len(parsed_blocks) == 1:
                return parsed_blocks[0]

            return parsed_blocks

        # 普通业务 list 保持原样，
        # 避免误解析 POI / 路线等列表。
        return result

    # 单个 MCP text content block
    if isinstance(result, dict):
        if (
            result.get("type") == "text"
            and isinstance(result.get("text"), str)
        ):
            return _parse_result(result["text"])

        return result

    # JSON 字符串
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return _parse_result(parsed)
        except (json.JSONDecodeError, TypeError):
            return {"raw_result": result}

    return {"raw_result": str(result)}

def _collect_exception_text(exc: BaseException) -> str:
    """
    收集普通异常、cause/context 和 ExceptionGroup
    中的错误信息，便于识别 MCP TaskGroup 内部错误。
    """
    parts = [
        type(exc).__name__,
        str(exc),
    ]

    nested = getattr(exc, "exceptions", None)

    if nested:
        for child in nested:
            parts.append(
                _collect_exception_text(child)
            )

    cause = getattr(exc, "__cause__", None)

    if cause is not None:
        parts.append(
            _collect_exception_text(cause)
        )

    context = getattr(exc, "__context__", None)

    if (
        context is not None
        and context is not cause
    ):
        parts.append(
            _collect_exception_text(context)
        )

    return " | ".join(
        part
        for part in parts
        if part
    )


def _is_retryable_mcp_error(
    exc: BaseException
) -> bool:
    """
    仅重试明显属于临时网络 / MCP transport
    的错误。

    参数校验、工具不存在等确定性错误直接失败，
    避免无意义等待。
    """
    if isinstance(
        exc,
        (asyncio.TimeoutError, TimeoutError)
    ):
        return True

    text = _collect_exception_text(
        exc
    ).lower()

    transient_markers = (
        "timeout",
        "timed out",
        "502",
        "503",
        "504",
        "bad gateway",
        "brokenresource",
        "closedresource",
        "connection reset",
        "connectionreset",
        "connection error",
        "connecterror",
        "server disconnected",
        "remoteprotocolerror",
        "endofstream",
        "temporarily unavailable",
        "taskgroup",
    )

    return any(
        marker in text
        for marker in transient_markers
    )


class LangChainAmapService:
    """LangChain 兼容的高德地图服务封装类

    使用 langchain-mcp-adapters 官方适配器，提供统一的异步服务接口。
    MCP 工具自动从服务器加载，无需手动定义 @tool 函数。
    """

    async def _ensure_initialized(self) -> None:
        await _init_mcp_client()

    async def invoke_tool(
            self,
            tool_name: str,
            arguments: Dict[str, Any]
    ) -> Any:
        """
        所有 MCP 工具的唯一执行入口。

        负责：
        - timeout
        - transient error retry
        - attempt 日志
        - MCP 调用耗时记录

        上层 Agent 不再自行 retry，
        避免重复重试。
        """
        last_error = None

        for attempt in range(
                1,
                MCP_TOOL_MAX_ATTEMPTS + 1
        ):
            started_at = perf_counter()

            try:
                await self._ensure_initialized()

                tool = await get_mcp_tool_by_name(
                    tool_name
                )

                if tool is None:
                    raise ValueError(
                        f"MCP 工具不存在: {tool_name}"
                    )

                result = await asyncio.wait_for(
                    tool.ainvoke(arguments),
                    timeout=MCP_TOOL_TIMEOUT_SECONDS,
                )

                elapsed = (
                        perf_counter() - started_at
                )

                print(
                    f"🛠️ MCP [{tool_name}] "
                    f"attempt={attempt}/"
                    f"{MCP_TOOL_MAX_ATTEMPTS} "
                    f"elapsed={elapsed:.2f}s "
                    f"status=success"
                )

                return result

            except Exception as exc:
                last_error = exc

                elapsed = (
                        perf_counter() - started_at
                )

                error_text = (
                    _collect_exception_text(exc)
                )

                retryable = (
                    _is_retryable_mcp_error(exc)
                )

                print(
                    f"⚠️ MCP [{tool_name}] "
                    f"attempt={attempt}/"
                    f"{MCP_TOOL_MAX_ATTEMPTS} "
                    f"elapsed={elapsed:.2f}s "
                    f"status=failed "
                    f"error={type(exc).__name__}: "
                    f"{error_text[:180]}"
                )

                if (
                        not retryable
                        or attempt >= MCP_TOOL_MAX_ATTEMPTS
                ):
                    break

                base_wait = min(
                    2 ** (attempt - 1),
                    5
                )

                jitter = random.uniform(
                    0,
                    0.5
                )

                wait_time = (
                        base_wait + jitter
                )

                print(
                    f"↻ MCP [{tool_name}] "
                    f"{wait_time:.1f}s 后重试"
                )

                await asyncio.sleep(
                    wait_time
                )

        raise last_error

    async def _call_tool(
            self,
            tool_name: str,
            arguments: Dict[str, Any]
    ) -> Any:
        """
        保留原有内部接口，
        实际统一委托给 invoke_tool。
        """
        return await self.invoke_tool(
            tool_name,
            arguments
        )

    async def search_poi(self, keywords: str, city: str) -> Any:
        result = await self._call_tool("maps_text_search", {
            "keywords": keywords,
            "city": city,
            "citylimit": "true"
        })
        return _parse_result(result)

    async def get_weather(self, city: str) -> Any:
        result = await self._call_tool("maps_weather", {"city": city})
        return _parse_result(result)

    async def _geocode_address(self, address: str, city: Optional[str] = None) -> Optional[str]:
        arguments = {"address": address}
        if city:
            arguments["city"] = city
        result = await self._call_tool("maps_geo", arguments)
        parsed = _parse_result(result)
        if isinstance(parsed, dict):
            geocodes = parsed.get("geocodes", [])
            if geocodes:
                location = geocodes[0].get("location", "")
                if location:
                    return location
        return None

    async def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking"
    ) -> Any:
        tool_name_map = {
            "walking": "maps_direction_walking",
            "driving": "maps_direction_driving",
            "transit": "maps_direction_transit_integrated"
        }
        tool_name = tool_name_map.get(route_type)
        if not tool_name:
            return {"error": f"不支持的路线类型: {route_type}"}

        origin_coord = await self._geocode_address(origin_address, origin_city)
        dest_coord = await self._geocode_address(destination_address, destination_city)

        if not origin_coord:
            return {"error": f"无法解析起点地址: {origin_address}"}
        if not dest_coord:
            return {"error": f"无法解析终点地址: {destination_address}"}

        arguments = {
            "origin": origin_coord,
            "destination": dest_coord
        }
        if route_type == "transit":
            arguments["city"] = origin_city or destination_city or ""
            arguments["cityd"] = destination_city or origin_city or ""

        result = await self._call_tool(tool_name, arguments)
        return _parse_result(result)

    async def geocode(self, address: str, city: Optional[str] = None) -> Any:
        arguments = {"address": address}
        if city:
            arguments["city"] = city
        result = await self._call_tool("maps_geo", arguments)
        return _parse_result(result)

    async def get_poi_detail(self, poi_id: str) -> Any:
        result = await self._call_tool("maps_search_detail", {"id": poi_id})
        return _parse_result(result)

    async def get_tools(self) -> List[BaseTool]:
        await self._ensure_initialized()
        return _mcp_tools

    async def get_tool(self, name: str) -> Optional[BaseTool]:
        return await get_mcp_tool_by_name(name)


_langchain_amap_service: Optional[LangChainAmapService] = None
_service_lock = threading.Lock()


def get_langchain_amap_service() -> LangChainAmapService:
    global _langchain_amap_service
    if _langchain_amap_service is None:
        with _service_lock:
            if _langchain_amap_service is None:
                _langchain_amap_service = LangChainAmapService()
    return _langchain_amap_service
