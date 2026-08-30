import asyncio
from unittest.mock import patch

from app.agents.enhanced_langgraph_agent import (
    ReplanOutput,
    replan_node,
)
from app.models.schemas import Attraction, Budget, DayPlan, TripPlan


def _attr(name: str, price: float = 50) -> Attraction:
    return Attraction(
        name=name,
        address="北京市",
        visit_duration=120,
        description=f"{name}介绍",
        category="历史文化",
        ticket_price=price,
    )


def _day(day_index: int, name: str, price: float = 50) -> DayPlan:
    return DayPlan(
        date=f"2026-09-0{day_index + 1}",
        day_index=day_index,
        description=f"第{day_index + 1}天",
        transportation="公共交通",
        accommodation="经济型酒店",
        attractions=[_attr(name, price)],
    )


def _plan() -> TripPlan:
    return TripPlan(
        city="北京",
        start_date="2026-09-01",
        end_date="2026-09-02",
        days=[
            _day(0, "故宫", 50),
            _day(1, "天坛", 50),
        ],
        weather_info=[],
        overall_suggestions="原始建议",
        budget=Budget(
            total_attractions=100,
            total_hotels=0,
            total_meals=0,
            total_transportation=20,
            total=120,
        ),
    )


class FakeStructuredOutput:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def ainvoke(self, messages):
        if self.error is not None:
            raise self.error
        return self.result


class FakeLLM:
    def __init__(self, parsed=None, error=None):
        self.parsed = parsed
        self.error = error

    def with_structured_output(self, *args, **kwargs):
        if self.error is not None:
            return FakeStructuredOutput(error=self.error)

        return FakeStructuredOutput(
            result={
                "parsed": self.parsed,
                "raw": None,
                "parsing_error": None,
            }
        )


def test_internal_replan_does_not_increment_plan_version():
    """内部 Replan 增加 revision_count，但不能增加用户可见版本号。"""
    original = _plan()

    revised_day = _day(1, "颐和园", 30)

    fake_llm = FakeLLM(
        parsed=ReplanOutput(
            revised_days=[revised_day],
            overall_suggestions="已完成内部修复",
        )
    )

    state = {
        "trip_plan": original,
        "constraints": {},
        "violations": [],
        "edit_intent": {
            "intent_summary": "",
            "affected_days": [1],
            "preserve_days": [],
        },
        "edit_feedback": "",
        "revision_count": 2,
        "plan_version": 3,
        "errors": [],
    }

    with (
        patch(
            "app.agents.enhanced_langgraph_agent.get_llm",
            return_value=fake_llm,
        ),
        patch(
            "app.agents.enhanced_langgraph_agent._state_violations",
            return_value=[],
        ),
    ):
        result = asyncio.run(replan_node(state))

    assert result["revision_count"] == 3
    assert result["plan_version"] == 3

    # 目标日期被替换。
    assert result["trip_plan"].days[1].attractions[0].name == "颐和园"

    # 非目标日期保持原样。
    assert (
        result["trip_plan"].days[0].model_dump()
        == original.days[0].model_dump()
    )


def test_user_edit_increments_plan_version_once():
    """用户成功 Edit 后，plan_version 应只增加一次。"""
    original = _plan()

    revised_day = _day(1, "颐和园", 30)

    fake_llm = FakeLLM(
        parsed=ReplanOutput(
            revised_days=[revised_day],
            overall_suggestions="第二天已改为颐和园",
        )
    )

    state = {
        "trip_plan": original,
        "constraints": {},
        "violations": [],
        "edit_intent": {
            "intent_summary": "修改第二天",
            "affected_days": [1],
            "preserve_days": [0],
        },
        "edit_feedback": "把第二天改成颐和园，第一天不要动",
        "revision_count": 0,
        "plan_version": 3,
        "errors": [],
    }

    with (
        patch(
            "app.agents.enhanced_langgraph_agent.get_llm",
            return_value=fake_llm,
        ),
        patch(
            "app.agents.enhanced_langgraph_agent._state_violations",
            return_value=[],
        ),
    ):
        result = asyncio.run(replan_node(state))

    assert result["revision_count"] == 1
    assert result["plan_version"] == 4
    assert result["edit_feedback"] == ""

    assert result["trip_plan"].days[1].attractions[0].name == "颐和园"

    # 用户明确要求第一天不动。
    assert (
        result["trip_plan"].days[0].model_dump()
        == original.days[0].model_dump()
    )


def test_failed_replan_does_not_increment_plan_version():
    """Replan 失败只能消耗一次尝试，不能产生新的用户版本。"""
    fake_llm = FakeLLM(
        error=RuntimeError("mock replan failure")
    )

    state = {
        "trip_plan": _plan(),
        "constraints": {},
        "violations": [],
        "edit_intent": {
            "intent_summary": "修改第二天",
            "affected_days": [1],
            "preserve_days": [],
        },
        "edit_feedback": "修改第二天",
        "revision_count": 0,
        "plan_version": 5,
        "errors": [],
    }

    with (
        patch(
            "app.agents.enhanced_langgraph_agent.get_llm",
            return_value=fake_llm,
        ),
        patch(
            "app.agents.enhanced_langgraph_agent._state_violations",
            return_value=[],
        ),
    ):
        result = asyncio.run(replan_node(state))

    assert result["revision_count"] == 1

    # Partial state update 中没有 plan_version，
    # 因此 LangGraph 会保留原来的版本号 5，而不是错误递增。
    assert "plan_version" not in result

    assert result["errors"]
    assert "replan_failed" in result["errors"][0]
