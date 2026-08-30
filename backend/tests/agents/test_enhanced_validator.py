"""Validator 的纯本地单元测试。

这组测试不调用 LLM/MCP，可以快速验证你最重要的个人改造：确定性约束校验。
"""

from app.agents.enhanced_langgraph_agent import validate_trip_plan
from app.models.agent_schemas import TravelConstraints
from app.models.schemas import Attraction, Budget, DayPlan, TripPlan, TripRequest


def _request() -> TripRequest:
    return TripRequest(
        city="北京",
        start_date="2026-09-01",
        end_date="2026-09-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
        free_text_input="预算1000元，每天最多2个景点，不要爬山",
    )


def _attr(name: str, category: str = "博物馆") -> Attraction:
    return Attraction(
        name=name,
        address="北京市",
        visit_duration=120,
        description=f"{name}介绍",
        category=category,
        ticket_price=50,
    )


def test_validator_detects_budget_daily_count_and_duplicate():
    request = _request()
    plan = TripPlan(
        city="北京",
        start_date=request.start_date,
        end_date=request.end_date,
        days=[
            DayPlan(
                date="2026-09-01",
                day_index=0,
                description="day1",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[_attr("故宫"), _attr("国博"), _attr("天坛")],
            ),
            DayPlan(
                date="2026-09-02",
                day_index=1,
                description="day2",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[_attr("故宫")],
            ),
        ],
        weather_info=[],
        overall_suggestions="test",
        budget=Budget(total=1500),
    )
    constraints = TravelConstraints(max_budget=1000, max_attractions_per_day=2)

    codes = {v.code for v in validate_trip_plan(plan, request, constraints)}
    assert "BUDGET_EXCEEDED" in codes
    assert "TOO_MANY_ATTRACTIONS" in codes
    assert "DUPLICATE_POI" in codes


def test_validator_passes_simple_valid_plan():
    request = _request()
    plan = TripPlan(
        city="北京",
        start_date=request.start_date,
        end_date=request.end_date,
        days=[
            DayPlan(
                date="2026-09-01", day_index=0, description="day1",
                transportation="公共交通", accommodation="经济型酒店",
                attractions=[_attr("故宫"), _attr("国博")],
            ),
            DayPlan(
                date="2026-09-02", day_index=1, description="day2",
                transportation="公共交通", accommodation="经济型酒店",
                attractions=[_attr("天坛")],
            ),
        ],
        weather_info=[],
        overall_suggestions="test",
        budget=Budget(total=900),
    )
    constraints = TravelConstraints(max_budget=1000, max_attractions_per_day=2)
    assert validate_trip_plan(plan, request, constraints) == []
