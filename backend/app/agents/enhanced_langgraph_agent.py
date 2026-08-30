"""Stateful / Constraint-Aware 旅行规划 Agent。

本文件以 04229f7 的 ``langgraph_agent.py`` 为 baseline：
- 复用原有 MCP Worker、景点聚类、路线规划、Structured Output、retry/fallback；
- 新增 Supervisor、硬约束解析、Validator -> Replan、自定义 Memory、HITL；
- 不直接修改原 1374 行 Agent，便于清晰区分 baseline 与个人改造。
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .langgraph_agent import (
    _ground_overall_weather_suggestions,
    _create_fallback_plan,
    cluster_attractions_node,
    gather_search_node,
    generate_plan_node,
    plan_route_node,
    search_food_node,
    search_hotel_node,
    search_poi_node,
    search_weather_node,
)
from ..models.agent_schemas import (
    AgentRunResponse,
    EditIntent,
    HITLDecision,
    ReplanOutput,
    SupervisorDecision,
    TravelConstraints,
    UserPreferenceProfile,
    Violation,
)
from ..models.schemas import Budget, DayPlan, TripPlan, TripRequest
from ..services.langchain_amap_tools import get_mcp_tools
from ..services.llm_service import get_llm
from ..services.memory_service import get_user_memory_service

import inspect
from functools import wraps
from time import perf_counter

import traceback

MAX_REPLAN = 2


class EnhancedTripPlannerState(TypedDict, total=False):
    """增强版 Graph State。

    total=False 很重要：同一 thread 的后续 edit 调用只需提交增量输入，
    其余 TripState 会从 LangGraph Checkpointer 中恢复。
    """

    # baseline 数据
    request: TripRequest
    attractions_info: str
    weather_info: str
    hotels_info: str
    food_info: str
    cluster_info: str
    route_info: str
    trip_plan: Optional[TripPlan]
    errors: List[str]

    # 新增：执行模式与身份
    mode: str
    user_id: str
    human_review_enabled: bool

    # 新增：Supervisor / Constraint / Reliability
    supervisor_decision: Dict[str, Any]
    constraints: Dict[str, Any]
    explicit_constraints: Optional[Dict[str, Any]]

    violations: List[Dict[str, Any]]
    revision_count: int

    # 用户可见的行程版本号。
    # 与 revision_count 分离：
    # revision_count 只控制单轮 Auto-Replan 次数；
    # plan_version 只在用户成功修改行程后递增。
    plan_version: int

    # 新增：Memory / 多轮修改 / HITL
    memory_profile: Dict[str, Any]
    edit_feedback: str
    edit_intent: Dict[str, Any]
    human_decision: str


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _timed_node(
    node_name: str,
    node_func
):
    """
    为 LangGraph Node 统一记录执行耗时。

    支持 async / sync node，
    不改变节点原有返回值和异常行为。
    """

    @wraps(node_func)
    async def wrapped(*args, **kwargs):
        started_at = perf_counter()
        status = "success"

        print(
            f"⏱️ NODE START [{node_name}]"
        )

        try:
            result = node_func(
                *args,
                **kwargs
            )

            if inspect.isawaitable(result):
                result = await result

            return result

        except Exception as exc:
            if type(exc).__name__ == "GraphInterrupt":
                status = "interrupt"
            else:
                status = (
                    f"failed:"
                    f"{type(exc).__name__}"
                )

            raise

        finally:
            elapsed = (
                perf_counter()
                - started_at
            )

            print(
                f"⏱️ NODE END   "
                f"[{node_name}] "
                f"elapsed={elapsed:.2f}s "
                f"status={status}"
            )

    return wrapped


def _model_dump(obj: Any) -> Dict[str, Any]:
    """兼容 Pydantic v2/v1 的最小序列化辅助。"""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"无法序列化对象: {type(obj)}")


def _merge_unique(items: List[str]) -> List[str]:
    """去空、去重，同时保持原有顺序。"""
    result: List[str] = []
    seen = set()
    for item in items:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _cn_num_to_int(text: str) -> Optional[int]:
    """解析本项目会用到的一到十中文数字，作为约束解析的本地 fallback。"""
    mapping = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    text = text.strip()
    if text.isdigit():
        return int(text)
    return mapping.get(text)


def _is_explicit_must_include(
    free_text: str,
    keyword: str,
) -> bool:
    """
    只有用户在自由文本中明确使用强制措辞，
    才允许普通偏好升级为 must_include。

    preferences / long-term memory 本身只是 soft preference。
    """
    text = free_text or ""
    keyword = (keyword or "").strip()

    if not text or not keyword:
        return False

    escaped = re.escape(keyword)

    patterns = [
        rf"(必须|一定要|一定|务必|必须要|不能少|不可缺少|至少要).*{escaped}",
        rf"{escaped}.*(必须|一定要|一定|务必|必须要|不能少|不可缺少)",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )



def _fallback_parse_constraints(request: TripRequest, memory: UserPreferenceProfile) -> TravelConstraints:
    """LLM 结构化解析失败时的规则兜底。

    目标不是覆盖所有中文表达，而是保证最常见的预算/景点数/老人/轻松节奏不会完全丢失。
    """
    text = " ".join(
        [request.free_text_input or "", " ".join(request.preferences or [])]
    )

    max_budget: Optional[int] = None
    budget_patterns = [
        r"预算(?:不超过|控制在|最好控制在|大约|约|为|是)?\s*(\d{2,7})\s*元?",
        r"(\d{2,7})\s*元(?:以内|之内)",
    ]
    for pattern in budget_patterns:
        match = re.search(pattern, text)
        if match:
            max_budget = int(match.group(1))
            break

    max_per_day: Optional[int] = None
    day_match = re.search(r"每天(?:最多|不要超过|不超过)\s*([一二两三四五六七八九十\d]+)\s*个?景点", text)
    if day_match:
        max_per_day = _cn_num_to_int(day_match.group(1))

    elderly = any(k in text for k in ["老人", "老年", "长辈", "父母同行"])
    relaxed = any(k in text for k in ["轻松", "慢节奏", "别太累", "不要太累", "悠闲"])

    # 这里仅对高频禁止项做保守识别，复杂表达交给 LLM parser。
    candidates = ["爬山", "登山", "徒步", "夜店", "酒吧", "网红店", "过山车"]
    excluded = [k for k in candidates if any(prefix + k in text for prefix in ["不想", "不要", "避免", "不喜欢", "拒绝"])]
    excluded = _merge_unique(memory.avoid_keywords + excluded)

    return TravelConstraints(
        max_budget=max_budget,
        max_attractions_per_day=max_per_day,
        excluded_keywords=excluded,
        elderly_friendly=elderly or memory.elderly_friendly,
        relaxed_pace=relaxed or memory.relaxed_pace,
    )


def _format_constraints_for_prompt(constraints: TravelConstraints) -> str:
    """把结构化约束转换为 Planner 可读的简洁中文。"""
    lines: List[str] = []
    if constraints.max_budget is not None:
        lines.append(f"- 总预算不得超过 {constraints.max_budget} 元")
    if constraints.max_attractions_per_day is not None:
        lines.append(f"- 每天景点数不得超过 {constraints.max_attractions_per_day} 个")
    if constraints.excluded_keywords:
        lines.append(f"- 禁止安排：{', '.join(constraints.excluded_keywords)}")
    if constraints.must_include_keywords:
        lines.append(f"- 必须尽量包含：{', '.join(constraints.must_include_keywords)}")
    if constraints.elderly_friendly:
        lines.append("- 需要老人友好，避免连续高强度步行、攀爬和过密行程")
    if constraints.relaxed_pace:
        lines.append("- 行程节奏偏轻松，预留休息时间")
    for note in constraints.hard_notes:
        lines.append(f"- {note}")
    return "\n".join(lines) or "- 无额外硬约束"


def _extract_interrupt_payload(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把 LangGraph Interrupt 对象转换为 FastAPI 可直接 JSON 序列化的字典。"""
    raw = result.get("__interrupt__") if isinstance(result, dict) else None
    if not raw:
        return None
    first = raw[0] if isinstance(raw, (list, tuple)) else raw
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"message": str(value)}


def _state_violations(state: EnhancedTripPlannerState) -> List[Violation]:
    result: List[Violation] = []
    for item in state.get("violations", []) or []:
        try:
            result.append(Violation.model_validate(item))
        except Exception:
            continue
    return result

def _format_budget_amount(value: Any) -> str:
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"
    except (TypeError, ValueError):
        return str(value)


def _ground_overall_budget_suggestions(
    trip_plan: TripPlan,
    constraints: TravelConstraints,
) -> TripPlan:
    """
    删除 LLM 生成的预算结论，
    再根据最终结构化 budget 和 max_budget
    确定性生成预算说明。
    """
    text = getattr(
        trip_plan,
        "overall_suggestions",
        "",
    ) or ""

    # 只删除明确涉及“预算事实/预算结论”的句子。
    budget_keywords = (
        "总预算",
        "预算控制",
        "控制预算",
        "预算以内",
        "预算内",
        "超预算",
        "超过预算",
        "超出预算",
        "预算上限",
    )

    pieces = re.split(
        r"(?<=[。！？!?；;])",
        text,
    )

    kept = []

    for piece in pieces:
        piece = piece.strip()

        if not piece:
            continue

        if any(
            keyword in piece
            for keyword in budget_keywords
        ):
            continue

        kept.append(piece)

    cleaned = "".join(kept).strip()

    budget = getattr(
        trip_plan,
        "budget",
        None,
    )

    total = (
        getattr(budget, "total", None)
        if budget is not None
        else None
    )

    max_budget = constraints.max_budget

    # 没有结构化预算数据，就不自行补预算事实。
    if total is None:
        trip_plan.overall_suggestions = cleaned
        return trip_plan

    total_text = _format_budget_amount(total)

    if max_budget is None:
        budget_note = (
            f"预算方面：当前估算总预算约{total_text}元。"
        )

    else:
        max_text = _format_budget_amount(max_budget)

        if total <= max_budget:
            budget_note = (
                f"预算方面：当前估算总预算约{total_text}元，"
                f"未超过预算上限{max_text}元。"
            )
        else:
            gap = total - max_budget
            gap_text = _format_budget_amount(gap)

            budget_note = (
                f"预算方面：当前估算总预算约{total_text}元，"
                f"超过预算上限{max_text}元约{gap_text}元，"
                f"当前方案仍未满足预算硬约束。"
            )

    if cleaned:
        if not cleaned.endswith(
            ("。", "！", "？", ".", "!", "?")
        ):
            cleaned += "。"

        cleaned += " "

    cleaned += budget_note

    trip_plan.overall_suggestions = cleaned

    return trip_plan


# ---------------------------------------------------------------------------
# Memory + Constraint + Supervisor 节点
# ---------------------------------------------------------------------------


async def load_memory_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    """读取 user-level 长期偏好，并形成当前请求的“有效偏好”。"""
    user_id = state.get("user_id", "anonymous")
    service = get_user_memory_service()
    profile = await asyncio.to_thread(service.load, user_id)

    request = state["request"]
    merged_preferences = _merge_unique((request.preferences or []) + profile.travel_preferences)

    # 不直接篡改用户显式住宿/交通选择，只在用户字段为空时使用 Memory。
    updates: Dict[str, Any] = {"preferences": merged_preferences}
    if not request.transportation and profile.preferred_transportation:
        updates["transportation"] = profile.preferred_transportation
    if not request.accommodation and profile.preferred_accommodation:
        updates["accommodation"] = profile.preferred_accommodation
    if (not request.food_preference or request.food_preference == "无特殊要求") and profile.preferred_food:
        updates["food_preference"] = profile.preferred_food

    effective_request = request.model_copy(update=updates)
    return {
        "request": effective_request,
        "memory_profile": profile.model_dump(),
    }


async def constraint_parser_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    """将自然语言要求转成结构化 TravelConstraints。

    优先级：显式 constraints > LLM Structured Output > 本地规则 fallback。
    """
    explicit = state.get("explicit_constraints")
    memory = UserPreferenceProfile.model_validate(state.get("memory_profile", {}))
    request = state["request"]

    if explicit:
        constraints = TravelConstraints.model_validate(explicit)
        # 长期 avoid Memory 只做补充，不覆盖本次用户显式字段。
        constraints.excluded_keywords = _merge_unique(memory.avoid_keywords + constraints.excluded_keywords)
        return {"constraints": constraints.model_dump()}

    fallback = _fallback_parse_constraints(request, memory)
    llm = get_llm()
    try:
        structured = llm.with_structured_output(TravelConstraints, method="function_calling")
        prompt = f"""你负责把旅行需求解析为可验证约束，不负责生成行程。

城市：{request.city}
旅行偏好：{request.preferences}
额外要求：{request.free_text_input or '无'}
历史长期偏好：{memory.model_dump_json(ensure_ascii=False)}

规则：
1. 只有用户明确表达的预算/每天景点数才填写数值，不要猜；
2. “不想/不要/避免/不喜欢”的活动放到 excluded_keywords；
3. 老人同行、轻松慢节奏分别填写对应布尔字段；
4. 不要把普通旅行偏好错误升级成 hard constraint。
"""
        parsed = await structured.ainvoke([HumanMessage(content=prompt)])

        if parsed:
            # LLM 结果与本地 fallback 合并，增强鲁棒性。
            parsed.excluded_keywords = _merge_unique(memory.avoid_keywords + fallback.excluded_keywords + parsed.excluded_keywords)
            if parsed.max_budget is None:
                parsed.max_budget = fallback.max_budget
            if parsed.max_attractions_per_day is None:
                parsed.max_attractions_per_day = fallback.max_attractions_per_day
            parsed.elderly_friendly = parsed.elderly_friendly or fallback.elderly_friendly
            parsed.relaxed_pace = parsed.relaxed_pace or fallback.relaxed_pace
            # ---------------------------------------------------------
            # Memory/preferences 是 soft preference，
            # 不允许 LLM 自动升级成 must_include hard constraint。
            #
            # 只有用户在本轮 free_text_input 中明确使用
            # “必须 / 一定要 / 务必”等措辞时才保留。
            # ---------------------------------------------------------
            parsed.must_include_keywords = _merge_unique([
                keyword
                for keyword in parsed.must_include_keywords
                if _is_explicit_must_include(
                    request.free_text_input or "",
                    keyword,
                )
            ])
            return {"constraints": parsed.model_dump()}

    except Exception as exc:
        print(f"⚠️ Constraint Parser Structured Output 失败，使用规则兜底: {exc}")

    return {"constraints": fallback.model_dump()}


async def supervisor_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    """Supervisor：决定本次请求需要调用哪些 Specialized Workers。

    Graph 仍然用并行 super-step 执行 POI/Weather/Hotel；
    每个 Worker 会读取此决策，不需要的 Worker 直接 no-op，从而避免无效 LLM/MCP 调用。
    """
    request = state["request"]
    constraints = TravelConstraints.model_validate(state.get("constraints", {}))

    default_workers = ["poi", "weather", "hotel", "food", "route"]
    # 明显的当天往返请求，不必调用酒店 Worker。
    free_text = request.free_text_input or ""
    if any(k in free_text for k in ["当天往返", "不住宿", "不用酒店", "不住酒店"]):
        default_workers.remove("hotel")

    fallback = SupervisorDecision(workers=default_workers, routing_reason="规则路由 fallback")
    llm = get_llm()
    try:
        structured = llm.with_structured_output(SupervisorDecision, method="function_calling")
        prompt = f"""你是旅行多智能体系统的 Supervisor，只做任务路由，不生成旅行计划。

请求：{request.model_dump_json(ensure_ascii=False)}
约束：{constraints.model_dump_json(ensure_ascii=False)}

可用 Worker：
- poi：景点检索，通常需要
- weather：天气查询
- hotel：住宿搜索；如果用户明确当天往返/不住宿可跳过
- food：餐饮搜索；若行程需要三餐推荐则执行
- route：交通路线规划；完整旅行计划通常执行

请只输出必要 Worker，并给一句简短 routing_reason。
"""
        decision = await structured.ainvoke([HumanMessage(content=prompt)])
        if decision and "poi" not in decision.workers:
            # 当前 baseline 的聚类/路线高度依赖景点，强制保留 poi 防止后续链路失效。
            decision.workers.insert(0, "poi")
        if decision:
            return {"supervisor_decision": decision.model_dump()}
    except Exception as exc:
        print(f"⚠️ Supervisor Structured Output 失败，使用规则路由: {exc}")

    return {"supervisor_decision": fallback.model_dump()}


def _worker_enabled(state: EnhancedTripPlannerState, name: str) -> bool:
    decision = SupervisorDecision.model_validate(state.get("supervisor_decision", {}))
    return name in decision.workers


# ---------------------------------------------------------------------------
# Supervisor 管理下的并行 Worker 包装
# ---------------------------------------------------------------------------


async def supervised_poi_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    if not _worker_enabled(state, "poi"):
        print("⏭️ Supervisor 跳过 POI Worker")
        return {"attractions_info": ""}
    return await search_poi_node(state)  # 复用 baseline 的 MCP + retry


async def supervised_weather_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    if not _worker_enabled(state, "weather"):
        print("⏭️ Supervisor 跳过 Weather Worker")
        return {"weather_info": ""}
    return await search_weather_node(state)


async def supervised_hotel_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    if not _worker_enabled(state, "hotel"):
        print("⏭️ Supervisor 跳过 Hotel Worker")
        return {"hotels_info": ""}
    return await search_hotel_node(state)


async def supervised_food_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    if not _worker_enabled(state, "food"):
        print("⏭️ Supervisor 跳过 Food Worker")
        return {"food_info": ""}
    return await search_food_node(state)


async def supervised_route_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    if not _worker_enabled(state, "route"):
        print("⏭️ Supervisor 跳过 Route Worker")
        return {"route_info": ""}
    return await plan_route_node(state)


async def enhanced_generate_plan_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    """在 baseline Planner 前注入结构化约束与长期偏好。"""
    request = state["request"]
    constraints = TravelConstraints.model_validate(state.get("constraints", {}))
    memory = UserPreferenceProfile.model_validate(state.get("memory_profile", {}))

    extra = f"""

【结构化硬约束——必须遵守】
{_format_constraints_for_prompt(constraints)}

【用户长期偏好——用于个性化，但不得覆盖本次显式要求】
{memory.model_dump_json(ensure_ascii=False)}
"""
    enriched_request = request.model_copy(
        update={"free_text_input": (request.free_text_input or "") + extra}
    )
    temp_state = dict(state)
    temp_state["request"] = enriched_request
    return await generate_plan_node(temp_state)


# ---------------------------------------------------------------------------
# Validator + Targeted Replan
# ---------------------------------------------------------------------------


def validate_trip_plan(plan: TripPlan, request: TripRequest, constraints: TravelConstraints) -> List[Violation]:
    """确定性 Validator。

    只把适合程序化判断的条件放在这里，避免用 LLM 去做预算加法、计数、查重等确定性任务。
    """
    violations: List[Violation] = []

    # 1) 旅行天数完整性
    if len(plan.days) != request.travel_days:
        violations.append(Violation(
            code="DAY_COUNT_MISMATCH",
            message=f"计划包含 {len(plan.days)} 天，但请求为 {request.travel_days} 天",
            expected=str(request.travel_days),
            actual=str(len(plan.days)),
        ))

    # 2) 总预算
    if constraints.max_budget is not None:
        if plan.budget is None:
            violations.append(Violation(
                code="BUDGET_MISSING",
                message="用户设置了预算上限，但计划没有返回 budget 信息",
                expected=f"<= {constraints.max_budget}",
                actual="missing",
            ))
        elif plan.budget.total > constraints.max_budget:
            violations.append(Violation(
                code="BUDGET_EXCEEDED",
                message=f"总预算 {plan.budget.total} 元，超过上限 {constraints.max_budget} 元",
                expected=f"<= {constraints.max_budget}",
                actual=str(plan.budget.total),
            ))

    # 3) 每日景点数量
    if constraints.max_attractions_per_day is not None:
        for day in plan.days:
            if len(day.attractions) > constraints.max_attractions_per_day:
                violations.append(Violation(
                    code="TOO_MANY_ATTRACTIONS",
                    message=(
                        f"第 {day.day_index + 1} 天安排 {len(day.attractions)} 个景点，"
                        f"超过上限 {constraints.max_attractions_per_day} 个"
                    ),
                    affected_days=[day.day_index],
                    expected=f"<= {constraints.max_attractions_per_day}",
                    actual=str(len(day.attractions)),
                ))

    # 4) 重复 POI
    seen: Dict[str, int] = {}
    for day in plan.days:
        for attr in day.attractions:
            key = re.sub(r"\s+", "", attr.name).lower()
            if key in seen:
                violations.append(Violation(
                    code="DUPLICATE_POI",
                    message=f"景点“{attr.name}”在第 {seen[key] + 1} 天和第 {day.day_index + 1} 天重复出现",
                    affected_days=[day.day_index],
                    expected="每个主要景点只安排一次",
                    actual=attr.name,
                ))
            else:
                seen[key] = day.day_index

    # 5) 明确禁止项：在名称/类别/描述中做可解释的字符串匹配
    excluded = [k.strip().lower() for k in constraints.excluded_keywords if k.strip()]
    if excluded:
        for day in plan.days:
            for attr in day.attractions:
                haystack = " ".join([
                    attr.name or "",
                    attr.category or "",
                    attr.description or "",
                ]).lower()
                hit = next((k for k in excluded if k in haystack), None)
                if hit:
                    violations.append(Violation(
                        code="EXCLUDED_ACTIVITY",
                        message=f"第 {day.day_index + 1} 天景点“{attr.name}”命中用户禁止项“{hit}”",
                        affected_days=[day.day_index],
                        expected=f"不得包含 {hit}",
                        actual=attr.name,
                    ))

    # 6) 必须包含项：只做简单可解释匹配
    if constraints.must_include_keywords:
        all_text = " ".join(
            f"{attr.name} {attr.category or ''} {attr.description or ''}"
            for day in plan.days for attr in day.attractions
        ).lower()
        for keyword in constraints.must_include_keywords:
            if keyword and keyword.lower() not in all_text:
                violations.append(Violation(
                    code="MUST_INCLUDE_MISSING",
                    message=f"计划未体现用户明确要求的主题/景点关键词“{keyword}”",
                    expected=f"包含 {keyword}",
                    actual="not found",
                ))

    return violations


async def validator_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    plan = state.get("trip_plan")
    if not plan:
        return {"violations": [Violation(
            code="PLAN_MISSING",
            message="Planner 未生成有效 TripPlan",
            expected="TripPlan",
            actual="None",
        ).model_dump()]}

    constraints = TravelConstraints.model_validate(state.get("constraints", {}))
    violations = validate_trip_plan(plan, state["request"], constraints)
    if violations:
        print(f"⚠️ Validator 检出 {len(violations)} 个问题")
    else:
        print("✅ Validator: 所有确定性硬约束通过")
    return {"violations": [v.model_dump() for v in violations]}


def _route_after_validator(state: EnhancedTripPlannerState) -> str:
    violations = state.get("violations", []) or []
    revision_count = state.get("revision_count", 0)
    if violations and revision_count < MAX_REPLAN:
        return "replan"
    if state.get("human_review_enabled", False):
        return "human_review"
    return "save_memory"


async def edit_analyzer_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    """把“第二天删一个景点，第一天别动”解析为 affected/preserve day_index。"""
    feedback = (state.get("edit_feedback") or "").strip()
    plan = state.get("trip_plan")
    if not feedback or not plan:
        return {"edit_intent": EditIntent(intent_summary=feedback).model_dump(), "revision_count": 0}

    # 先做简单规则解析，LLM 失败时仍可局部修改。
    affected: List[int] = []
    preserve: List[int] = []
    for match in re.finditer(r"第\s*([一二两三四五六七八九十\d]+)\s*天", feedback):
        value = _cn_num_to_int(match.group(1))
        if value is None:
            continue
        index = value - 1
        window = feedback[max(0, match.start() - 8): match.end() + 8]
        if any(k in window for k in ["不要改", "别改", "保持", "保留"]):
            preserve.append(index)
        else:
            affected.append(index)

    fallback = EditIntent(
        affected_days=sorted(set(affected)),
        preserve_days=sorted(set(preserve)),
        intent_summary=feedback,
    )

    llm = get_llm()
    try:
        structured = llm.with_structured_output(EditIntent, method="function_calling")
        parsed = await structured.ainvoke([HumanMessage(content=f"""解析用户对现有旅行计划的修改范围。
现有天数：{len(plan.days)}
用户意见：{feedback}

规则：day_index 从 0 开始；“第一天不要改”应进入 preserve_days；
没有明确日期但属于全局修改时 affected_days 可以为空。
""")])
        if parsed:
            parsed.affected_days = sorted(set(parsed.affected_days + fallback.affected_days))
            parsed.preserve_days = sorted(set(parsed.preserve_days + fallback.preserve_days))
            return {"edit_intent": parsed.model_dump(), "revision_count": 0}
    except Exception as exc:
        print(f"⚠️ Edit Analyzer Structured Output 失败，使用规则解析: {exc}")

    return {"edit_intent": fallback.model_dump(), "revision_count": 0}


async def replan_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    """根据 violation 或用户 feedback 做定向重规划，然后程序化 Merge。

    对局部问题只替换 affected_days，未受影响日期直接保留原对象，
    这比“整份行程重新生成”更稳定，也更便于面试解释 token/漂移成本。
    """
    plan = state.get("trip_plan")
    if not plan:
        return {"revision_count": state.get("revision_count", 0) + 1}

    constraints = TravelConstraints.model_validate(state.get("constraints", {}))
    violations = _state_violations(state)
    edit_intent = EditIntent.model_validate(state.get("edit_intent", {}))
    feedback = state.get("edit_feedback", "")
    is_user_edit = bool(feedback.strip())

    affected_days = set(edit_intent.affected_days)
    for violation in violations:
        affected_days.update(violation.affected_days)

    # 全局预算/天数问题无法可靠定位到单日，允许 Planner 返回所有日进行调整。
    global_codes = {"BUDGET_EXCEEDED", "BUDGET_MISSING", "DAY_COUNT_MISMATCH", "MUST_INCLUDE_MISSING", "PLAN_MISSING"}
    if any(v.code in global_codes for v in violations):
        affected_days = {day.day_index for day in plan.days}

    # 用户明确要求保留的天永远从修改范围中移除。
    affected_days -= set(edit_intent.preserve_days)

    # 没有明确范围时，视为全局 edit。
    if not affected_days and feedback:
        affected_days = {day.day_index for day in plan.days if day.day_index not in edit_intent.preserve_days}

    target_days = sorted(affected_days)
    original_dict = plan.model_dump()

    current_total = plan.budget.total if plan.budget else None
    max_budget = constraints.max_budget

    required_savings = (
        max(0, current_total - max_budget)
        if current_total is not None and max_budget is not None
        else 0
    )

    if current_total is not None and max_budget is not None:
        budget_repair_context = (
            f"当前程序计算总预算：{current_total} 元\n"
            f"用户预算上限：{max_budget} 元\n"
            f"本次至少需要节省：{required_savings} 元\n"
            f"修复后的程序重算总预算必须 <= {max_budget} 元"
        )
    else:
        budget_repair_context = "当前没有明确的预算修复目标"


    prompt = f"""你是旅行计划修复 Agent。请只修复指定日期，不要无关重写。

【原始完整 TripPlan】
{json.dumps(original_dict, ensure_ascii=False)}

【硬约束】
{_format_constraints_for_prompt(constraints)}

【Validator 反馈】
{json.dumps([v.model_dump() for v in violations], ensure_ascii=False)}

【预算修复目标】
{budget_repair_context}

【用户修改意见】
{feedback or '无'}

【允许修改的 day_index】
{target_days}

【必须保持不变的 day_index】
{edit_intent.preserve_days}

要求：
1. revised_days 只返回需要替换的日期；
2. 每个 DayPlan 字段保持完整；
3. 优先通过删减/替换局部内容修复，而不是推翻整个行程；
4. 如果存在 BUDGET_EXCEEDED，必须真实降低结构化费用字段，而不是只修改文字说明；
5. 可以通过减少付费景点、选择更便宜酒店、降低餐饮费用等方式满足预算；
6. ticket_price、hotel.estimated_cost、meal.estimated_cost 等费用必须与修改后的方案一致；
7. 如果提示“至少需要节省 N 元”，修改后的总费用必须至少减少 N 元，最终程序重算预算必须 <= max_budget；
8. 所有费用不得为负数，不得通过虚构负费用规避预算限制；
9. 不要修改 preserve_days。
10. overall_suggestions 必须根据修改后的最新完整行程重新生成；
11. overall_suggestions 不得再提及已经删除、替换或取消的景点/餐厅/安排；
12. 即使本次只修改一天，总体建议也必须与最终合并后的完整行程保持一致。

"""

    llm = get_llm()
    try:
        messages = [
            SystemMessage(content="你是约束感知的旅行行程修复 Agent。"),
            HumanMessage(content=prompt),
        ]

        # 第一层：优先使用 function_calling。
        # include_raw=True 便于诊断 parsed=None 的真实原因。
        structured = llm.with_structured_output(
            ReplanOutput,
            method="function_calling",
            include_raw=True,
        )

        structured_result = await structured.ainvoke(messages)
        output = structured_result.get("parsed")

        # 第二层：function_calling 没拿到结构化结果时，
        # 改用 DeepSeek JSON Output 再尝试一次。
        if output is None:
            raw = structured_result.get("raw")
            parsing_error = structured_result.get("parsing_error")

            raw_content = getattr(raw, "content", "") if raw else ""
            tool_calls = getattr(raw, "tool_calls", []) if raw else []
            response_metadata = getattr(raw, "response_metadata", {}) if raw else {}
            finish_reason = response_metadata.get("finish_reason")

            print(
                "⚠️ Replan function_calling 未返回 parsed，尝试 JSON fallback: "
                f"finish_reason={finish_reason}, "
                f"tool_calls={len(tool_calls)}, "
                f"parsing_error={parsing_error}, "
                f"content={str(raw_content)[:300]}"
            )

            schema_json = json.dumps(
                ReplanOutput.model_json_schema(),
                ensure_ascii=False,
            )

            example = {
                "revised_days": (
                    [original_dict["days"][0]]
                    if original_dict.get("days")
                    else []
                ),
                "overall_suggestions": "根据硬约束修复后的总体建议",
            }

            json_prompt = f"""
    {prompt}

    【JSON 输出要求】
    必须只输出合法 JSON，不要输出 Markdown，不要输出解释文字。

    JSON Schema:
    {schema_json}

    JSON 示例:
    {json.dumps(example, ensure_ascii=False)}

    注意：
    - revised_days 必须包含需要修改的完整 DayPlan。
    - day_index 必须与 target_days 对应。
    - 不要省略 DayPlan 的必要字段。
    """

            json_structured = llm.with_structured_output(
                ReplanOutput,
                method="json_mode",
            )

            output = await json_structured.ainvoke([
                SystemMessage(
                    content="你是旅行行程修复 Agent。必须按照用户要求输出 JSON。"
                ),
                HumanMessage(content=json_prompt),
            ])

        if output is None:
            raise ValueError("Replan function_calling 和 JSON fallback 均未返回有效结果")

        if target_days and not output.revised_days:
            raise ValueError(
                f"Replan 返回 revised_days 为空，但需要修改 target_days={target_days}"
            )

        returned_days = sorted({day.day_index for day in output.revised_days})
        matched_days = sorted(set(returned_days) & set(target_days))
        unexpected_days = sorted(set(returned_days) - set(target_days))

        attempt_no = state.get("revision_count", 0) + 1

        print(
            f"🔧 Replan #{attempt_no}: "
            f"target_days={target_days}, "
            f"returned_days={returned_days}, "
            f"matched_days={matched_days}"
        )

        if unexpected_days:
            raise ValueError(
                f"Replan 返回了不允许修改的 day_index: {unexpected_days}; "
                f"允许范围={target_days}"
            )

        if target_days and not matched_days:
            raise ValueError(
                f"Replan 没有返回任何目标日期: "
                f"target_days={target_days}, returned_days={returned_days}"
            )

        replacement = {day.day_index: day for day in output.revised_days}
        merged_days: List[DayPlan] = []
        for old_day in plan.days:
            if old_day.day_index in target_days and old_day.day_index in replacement:
                merged_days.append(replacement[old_day.day_index])
            else:
                merged_days.append(old_day)

        # Replan 后由程序重新汇总预算，而不是相信 LLM 自报的 total。
        # 这使“预算上限”成为真正可测试的 deterministic feedback loop。
        total_attractions = sum(
            max(0, attr.ticket_price or 0)
            for day in merged_days for attr in day.attractions
        )
        total_hotels = sum(
            max(0, day.hotel.estimated_cost or 0)
            for day in merged_days if day.hotel is not None
        )
        total_meals = sum(
            max(0, meal.estimated_cost or 0)
            for day in merged_days for meal in day.meals
        )
        # RouteSegment 当前没有费用字段，因此交通费用沿用 baseline 已有估算；
        # 后续如果接入交通票价工具，可以在这里替换为真实重算。
        total_transportation = plan.budget.total_transportation if plan.budget else 0
        recalculated_budget = Budget(
            total_attractions=total_attractions,
            total_hotels=total_hotels,
            total_meals=total_meals,
            total_transportation=total_transportation,
            total=total_attractions + total_hotels + total_meals + total_transportation,
        )

        before_total = plan.budget.total if plan.budget else 0
        after_total = recalculated_budget.total

        print(
            f"💰 Replan #{attempt_no} budget: "
            f"{before_total} → {after_total} 元 "
            f"(景点={recalculated_budget.total_attractions}, "
            f"酒店={recalculated_budget.total_hotels}, "
            f"餐饮={recalculated_budget.total_meals}, "
            f"交通={recalculated_budget.total_transportation})"
        )

        if (
            constraints.max_budget is not None
            and any(v.code == "BUDGET_EXCEEDED" for v in violations)
        ):
            if after_total <= constraints.max_budget:
                print(
                    f"✅ Replan #{attempt_no} 已满足预算: "
                    f"{after_total} <= {constraints.max_budget}"
                )
            elif after_total >= before_total:
                print(
                    f"⚠️ Replan #{attempt_no} 预算没有改善: "
                    f"{before_total} → {after_total}"
                )
            else:
                remaining_gap = after_total - constraints.max_budget
                print(
                    f"⚠️ Replan #{attempt_no} 虽然降低预算，"
                    f"但仍超出 {remaining_gap} 元"
                )

        updated_suggestions = (
            (output.overall_suggestions or "").strip()
        )

        if not updated_suggestions:
            day_summaries = []

            for day in merged_days:
                attraction_names = [
                    attr.name
                    for attr in day.attractions
                    if getattr(attr, "name", None)
                ]

                if attraction_names:
                    day_summaries.append(
                        f"第{day.day_index + 1}天安排"
                        f"{'、'.join(attraction_names)}"
                    )

            if day_summaries:
                updated_suggestions = (
                        "行程已根据最新修改更新："
                        + "；".join(day_summaries)
                        + "。请以当前每日行程安排为准。"
                )
            else:
                updated_suggestions = (
                    "行程已根据最新修改更新，"
                    "请以当前每日行程安排为准。"
                )

        repaired = plan.model_copy(update={
            "days": merged_days,
            "overall_suggestions": updated_suggestions,
            "budget": recalculated_budget,
        })

        next_plan_version = (
            state.get("plan_version", 0) + 1
            if is_user_edit
            else state.get("plan_version", 0)
        )

        return {
            "trip_plan": repaired,
            "revision_count": state.get("revision_count", 0) + 1,
            "plan_version": next_plan_version,
            "violations": [],
            "edit_feedback": "",
        }

    except Exception as exc:
        print(f"❌ Targeted Replan 失败: {exc}")
        return {
            "revision_count": state.get("revision_count", 0) + 1,
            "errors": (state.get("errors", []) or []) + [f"replan_failed: {exc}"],
        }


# ---------------------------------------------------------------------------
# HITL + 保存长期 Memory
# ---------------------------------------------------------------------------


async def human_review_node(state: EnhancedTripPlannerState) -> Dict[str, Any]:
    """使用 LangGraph interrupt 暂停执行，等待用户 approve/edit。"""
    plan = state.get("trip_plan")
    payload = {
        "type": "trip_review",
        "message": "行程已生成，请确认；如需修改可提交 edit + feedback。",
        "trip_plan": plan.model_dump() if plan else None,
        "violations": state.get("violations", []),
        "revision_count": state.get("revision_count", 0),
        "plan_version": state.get("plan_version", 0),
    }

    # interrupt 会由 checkpointer 保存当前位置；resume 后返回 HITLDecision 对应字典。
    decision = interrupt(payload)
    if not isinstance(decision, dict):
        decision = {"action": "approve", "feedback": ""}

    action = decision.get("action", "approve")
    feedback = (decision.get("feedback") or "").strip()
    if action == "edit" and feedback:
        return {
            "human_decision": "edit",
            "edit_feedback": feedback,
            "violations": [],
            "revision_count": 0,
        }
    return {"human_decision": "approve"}


def _route_after_human(state: EnhancedTripPlannerState) -> str:
    return "edit_analyzer" if state.get("human_decision") == "edit" else "save_memory"


async def save_memory_node(
    state: EnhancedTripPlannerState,
) -> Dict[str, Any]:
    """
    Graph 最终出口：

    1. 始终对最终 TripPlan 执行 Weather Grounding；
    2. 若仍存在 hard violation，则不写长期 Memory；
    3. 只有有效完成的规划才保存稳定用户偏好。
    """
    updates: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # 1. 最终输出边界：Weather Grounding
    # ---------------------------------------------------------
    plan = state.get("trip_plan")

    constraints = TravelConstraints.model_validate(
        state.get("constraints", {})
    )

    if plan is not None:
        # 天气事实确定性 grounding
        grounded_plan = _ground_overall_weather_suggestions(
            plan
        )

        # 预算事实确定性 grounding
        grounded_plan = _ground_overall_budget_suggestions(
            grounded_plan,
            constraints,
        )

        updates["trip_plan"] = grounded_plan

        print("✅ Final Weather Grounding 已执行")
        print("✅ Final Budget Grounding 已执行")

    # ---------------------------------------------------------
    # 2. 最终仍有 hard violation：不保存 Memory
    # ---------------------------------------------------------
    violations = _state_violations(state)

    hard_violations = [
        violation
        for violation in violations
        if violation.severity == "hard"
    ]

    if hard_violations:
        print(
            f"⚠️ Finalize: 仍存在 "
            f"{len(hard_violations)} 个硬约束问题，"
            f"跳过长期 Memory 保存"
        )
        return updates

    # ---------------------------------------------------------
    # 3. Memory
    # ---------------------------------------------------------
    user_id = state.get("user_id", "anonymous")

    if not user_id or user_id == "anonymous":
        return updates

    request = state["request"]

    service = get_user_memory_service()

    old = await asyncio.to_thread(
        service.load,
        user_id,
    )

    profile = UserPreferenceProfile(
        travel_preferences=_merge_unique(
            old.travel_preferences + (request.preferences or [])
        ),
        avoid_keywords=_merge_unique(
            old.avoid_keywords + constraints.excluded_keywords
        ),
        preferred_transportation=(
            request.transportation
            or old.preferred_transportation
        ),
        preferred_accommodation=(
            request.accommodation
            or old.preferred_accommodation
        ),
        preferred_food=(
            request.food_preference
            or old.preferred_food
        ),
        relaxed_pace=(
            old.relaxed_pace
            or constraints.relaxed_pace
        ),
        elderly_friendly=(
            old.elderly_friendly
            or constraints.elderly_friendly
        ),
    )

    await asyncio.to_thread(
        service.save,
        user_id,
        profile,
    )

    updates["memory_profile"] = profile.model_dump()

    return updates




# ---------------------------------------------------------------------------
# Graph 路由与构建
# ---------------------------------------------------------------------------


def _entry_router(state: EnhancedTripPlannerState) -> str:
    """同一 thread 的新一轮 edit 直接进入 Edit Analyzer，不重复做 MCP Research。"""
    if state.get("mode") == "edit" and state.get("trip_plan"):
        return "edit"
    return "new"


def create_enhanced_trip_planner_graph(checkpointer: Optional[InMemorySaver] = None):
    """构建增强版 Graph。

    当前使用 InMemorySaver，适合简历 Demo 和本地开发。
    生产环境可替换 AsyncSqliteSaver / AsyncPostgresSaver，无需改节点代码。
    """
    workflow = StateGraph(EnhancedTripPlannerState)

    workflow.add_node(
        "load_memory",
        _timed_node(
            "load_memory",
            load_memory_node
        )
    )

    workflow.add_node(
        "constraint_parser",
        _timed_node(
            "constraint_parser",
            constraint_parser_node
        )
    )

    workflow.add_node(
        "supervisor",
        _timed_node(
            "supervisor",
            supervisor_node
        )
    )

    workflow.add_node(
        "search_poi",
        _timed_node(
            "search_poi",
            supervised_poi_node
        )
    )

    workflow.add_node(
        "search_weather",
        _timed_node(
            "search_weather",
            supervised_weather_node
        )
    )

    workflow.add_node(
        "search_hotel",
        _timed_node(
            "search_hotel",
            supervised_hotel_node
        )
    )

    workflow.add_node(
        "gather_search",
        _timed_node(
            "gather_search",
            gather_search_node
        )
    )

    workflow.add_node(
        "cluster_attractions",
        _timed_node(
            "cluster_attractions",
            cluster_attractions_node
        )
    )

    workflow.add_node(
        "search_food",
        _timed_node(
            "search_food",
            supervised_food_node
        )
    )

    workflow.add_node(
        "plan_route",
        _timed_node(
            "plan_route",
            supervised_route_node
        )
    )

    workflow.add_node(
        "generate_plan",
        _timed_node(
            "generate_plan",
            enhanced_generate_plan_node
        )
    )

    workflow.add_node(
        "validator",
        _timed_node(
            "validator",
            validator_node
        )
    )

    workflow.add_node(
        "replan",
        _timed_node(
            "replan",
            replan_node
        )
    )

    workflow.add_node(
        "edit_analyzer",
        _timed_node(
            "edit_analyzer",
            edit_analyzer_node
        )
    )

    workflow.add_node(
        "human_review",
        _timed_node(
            "human_review",
            human_review_node
        )
    )

    workflow.add_node(
        "save_memory",
        _timed_node(
            "save_memory",
            save_memory_node
        )
    )

    workflow.add_conditional_edges(
        START,
        _entry_router,
        {"new": "load_memory", "edit": "edit_analyzer"},
    )

    workflow.add_edge("load_memory", "constraint_parser")
    workflow.add_edge("constraint_parser", "supervisor")

    # Supervisor 后进入同一 super-step：三个无数据依赖 Worker 并行执行。
    workflow.add_edge("supervisor", "search_poi")
    workflow.add_edge("supervisor", "search_weather")
    workflow.add_edge("supervisor", "search_hotel")
    workflow.add_edge(["search_poi", "search_weather", "search_hotel"], "gather_search")

    workflow.add_edge("gather_search", "cluster_attractions")
    workflow.add_edge("cluster_attractions", "search_food")
    workflow.add_edge("search_food", "plan_route")
    workflow.add_edge("plan_route", "generate_plan")
    workflow.add_edge("generate_plan", "validator")

    workflow.add_conditional_edges(
        "validator",
        _route_after_validator,
        {
            "replan": "replan",
            "human_review": "human_review",
            "save_memory": "save_memory",
        },
    )
    workflow.add_edge("replan", "validator")

    workflow.add_conditional_edges(
        "human_review",
        _route_after_human,
        {"edit_analyzer": "edit_analyzer", "save_memory": "save_memory"},
    )
    workflow.add_edge("edit_analyzer", "replan")
    workflow.add_edge("save_memory", END)

    return workflow.compile(checkpointer=checkpointer or InMemorySaver())


# ---------------------------------------------------------------------------
# 对外封装类
# ---------------------------------------------------------------------------


class EnhancedLangGraphTripPlanner:
    """增强版旅行 Agent 服务层封装。"""

    def __init__(self):
        print("🔄 初始化 Enhanced LangGraph 旅行规划系统...")
        self.checkpointer = InMemorySaver()
        self.app = create_enhanced_trip_planner_graph(self.checkpointer)

    @staticmethod
    def _config(thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def _warmup(self) -> None:
        """复用 baseline 的 LLM/MCP 预初始化逻辑。"""
        try:
            get_llm()
            await get_mcp_tools()
        except Exception as exc:
            # 保持 baseline 的 graceful degradation 风格。
            print(f"⚠️ Enhanced Agent 服务预初始化失败: {exc}")

    async def plan_trip(
        self,
        request: TripRequest,
        thread_id: str,
        user_id: str = "anonymous",
        constraints: Optional[TravelConstraints] = None,
        enable_human_review: bool = True,
    ) -> AgentRunResponse:
        """创建新旅行 thread；若启用 HITL，通常会停在 waiting_human。"""
        await self._warmup()
        initial_state: EnhancedTripPlannerState = {
            "mode": "new",
            "user_id": user_id,
            "request": request,
            "human_review_enabled": enable_human_review,
            "explicit_constraints": constraints.model_dump() if constraints else None,
            "attractions_info": "",
            "weather_info": "",
            "hotels_info": "",
            "food_info": "",
            "cluster_info": "",
            "route_info": "",
            "trip_plan": None,
            "errors": [],
            "violations": [],
            "revision_count": 0,
            "plan_version": 0,
            "edit_feedback": "",
            "edit_intent": {},
            "human_decision": "",
            "memory_profile": {},
            "supervisor_decision": {},
            "constraints": {},
        }

        try:
            result = await self.app.ainvoke(initial_state, config=self._config(thread_id))
            return self._build_response(result, thread_id, request)
        except Exception as exc:
            print(
                f"❌ Enhanced Agent 生成失败: "
                f"{type(exc).__name__}: {exc}"
            )

            traceback.print_exc()
            fallback = _create_fallback_plan(request)
            return AgentRunResponse(
                success=False,
                status="failed",
                message=f"增强版 Graph 执行失败，已返回 baseline fallback: {exc}",
                thread_id=thread_id,
                data=fallback,
            )

    async def resume(self, thread_id: str, decision: HITLDecision) -> AgentRunResponse:
        """使用相同 thread_id 恢复 HITL interrupt。"""
        try:
            result = await self.app.ainvoke(
                Command(resume=decision.model_dump()),
                config=self._config(thread_id),
            )
            snapshot = await self.app.aget_state(self._config(thread_id))
            request = snapshot.values.get("request")
            return self._build_response(result, thread_id, request)
        except Exception as exc:
            return AgentRunResponse(
                success=False,
                status="failed",
                message=f"恢复 HITL 失败: {exc}",
                thread_id=thread_id,
            )

    async def edit_trip(self, thread_id: str, feedback: str, enable_human_review: bool = True) -> AgentRunResponse:
        """对已经完成/已确认的 thread 发起新一轮局部修改。

        同一 thread 的 checkpoint 会保留原 TripPlan；本轮输入仅更新 mode/edit_feedback，
        START router 随即进入 edit_analyzer -> replan，不重复搜索 POI/天气/酒店。
        """
        config = self._config(thread_id)
        snapshot = await self.app.aget_state(config)
        if not snapshot.values or not snapshot.values.get("trip_plan"):
            return AgentRunResponse(
                success=False,
                status="failed",
                message="未找到该 thread 的历史 TripPlan；请先创建行程",
                thread_id=thread_id,
            )

        result = await self.app.ainvoke(
            {
                "mode": "edit",
                "edit_feedback": feedback,
                "human_review_enabled": enable_human_review,
                "human_decision": "",
                "violations": [],
                "revision_count": 0,
            },
            config=config,
        )
        request = snapshot.values.get("request")
        return self._build_response(result, thread_id, request)

    async def get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """面试演示/调试接口：查看 Checkpointer 中的最新状态。"""
        snapshot = await self.app.aget_state(self._config(thread_id))
        values = dict(snapshot.values or {})
        plan = values.get("trip_plan")
        if plan is not None and hasattr(plan, "model_dump"):
            values["trip_plan"] = plan.model_dump()
        request = values.get("request")
        if request is not None and hasattr(request, "model_dump"):
            values["request"] = request.model_dump()
        return {
            "values": values,
            "next": list(snapshot.next or []),
            "config": snapshot.config,
        }

    def _build_response(
            self,
            result: Dict[str, Any],
            thread_id: str,
            request: Optional[TripRequest],
    ) -> AgentRunResponse:
        """统一把 Graph state / interrupt 转成 API 响应。"""

        payload = _extract_interrupt_payload(result)

        plan = (
            result.get("trip_plan")
            if isinstance(result, dict)
            else None
        )

        revision_count = (
            result.get("revision_count", 0)
            if isinstance(result, dict)
            else 0
        )

        plan_version = (
            result.get("plan_version", 0)
            if isinstance(result, dict)
            else 0
        )

        constraints = None
        supervisor = None
        violations: List[Violation] = []

        if isinstance(result, dict):
            try:
                constraints = TravelConstraints.model_validate(
                    result.get("constraints", {})
                )
            except Exception:
                pass

            try:
                supervisor = SupervisorDecision.model_validate(
                    result.get("supervisor_decision", {})
                )
            except Exception:
                pass

            violations = _state_violations(result)

        # ---------------------------------------------------------
        # HITL interrupt
        # ---------------------------------------------------------
        if payload is not None:
            return AgentRunResponse(
                success=True,
                status="waiting_human",
                message="Graph 已暂停，等待用户确认或修改",
                thread_id=thread_id,
                data=plan,
                interrupt=payload,
                constraints=constraints,
                violations=violations,
                revision_count=revision_count,
                plan_version=plan_version,
                supervisor=supervisor,
            )

        # ---------------------------------------------------------
        # Graph 没有生成有效 TripPlan
        # ---------------------------------------------------------
        if plan is None:
            fallback = (
                _create_fallback_plan(request)
                if request is not None
                else None
            )

            return AgentRunResponse(
                success=False,
                status="failed",
                message=(
                    "未生成有效旅行计划，已返回 baseline fallback"
                    if fallback is not None
                    else "未生成有效旅行计划"
                ),
                thread_id=thread_id,
                data=fallback,
                constraints=constraints,
                violations=violations,
                revision_count=revision_count,
                plan_version=plan_version,
                supervisor=supervisor,
            )

        # ---------------------------------------------------------
        # Auto-Replan 已耗尽，但仍有 hard violation
        # ---------------------------------------------------------
        hard_violations = [
            violation
            for violation in violations
            if violation.severity == "hard"
        ]

        if hard_violations:
            return AgentRunResponse(
                success=False,
                status="failed",
                message=(
                    f"旅行计划在 {revision_count} 次重规划后"
                    f"仍存在 {len(hard_violations)} 个硬约束问题"
                ),
                thread_id=thread_id,
                data=plan,
                constraints=constraints,
                violations=violations,
                revision_count=revision_count,
                plan_version=plan_version,
                supervisor=supervisor,
            )

        # ---------------------------------------------------------
        # Completed
        # ---------------------------------------------------------
        return AgentRunResponse(
            success=True,
            status="completed",
            message="旅行计划已完成",
            thread_id=thread_id,
            data=plan,
            constraints=constraints,
            violations=violations,
            revision_count=revision_count,
            plan_version=plan_version,
            supervisor=supervisor,
        )


_enhanced_planner: Optional[EnhancedLangGraphTripPlanner] = None


def get_enhanced_trip_planner_agent() -> EnhancedLangGraphTripPlanner:
    """进程内单例：保证 InMemorySaver 不会因为每个请求重新实例化而丢失 thread。"""
    global _enhanced_planner
    if _enhanced_planner is None:
        _enhanced_planner = EnhancedLangGraphTripPlanner()
    return _enhanced_planner
