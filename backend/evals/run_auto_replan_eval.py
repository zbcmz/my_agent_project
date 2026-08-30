import asyncio

from app.agents.enhanced_langgraph_agent import (
    MAX_REPLAN,
    replan_node,
    validate_trip_plan,
)
from app.models.agent_schemas import TravelConstraints
from app.models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    TripPlan,
    TripRequest,
)


def attraction(name: str, price: float) -> Attraction:
    return Attraction(
        name=name,
        address="北京市",
        visit_duration=120,
        description=f"{name}测试景点",
        category="历史文化",
        ticket_price=price,
    )


def build_request() -> TripRequest:
    return TripRequest(
        city="北京",
        start_date="2026-09-02",
        end_date="2026-09-03",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
        food_preference="本地特色",
        free_text_input="总预算不超过600元。",
    )


def build_intentionally_invalid_plan() -> TripPlan:
    """
    故意制造一个确定超预算的方案。

    景点：
    4 * 250 = 1000

    交通：
    100

    总计：
    1100 > 600
    """
    return TripPlan(
        city="北京",
        start_date="2026-09-02",
        end_date="2026-09-03",
        days=[
            DayPlan(
                date="2026-09-02",
                day_index=0,
                description="第一天",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    attraction("故宫博物院", 250),
                    attraction("测试收费景点A", 250),
                ],
            ),
            DayPlan(
                date="2026-09-03",
                day_index=1,
                description="第二天",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    attraction("测试收费景点B", 250),
                    attraction("测试收费景点C", 250),
                ],
            ),
        ],
        weather_info=[],
        overall_suggestions="原始高预算测试方案",
        budget=Budget(
            total_attractions=1000,
            total_hotels=0,
            total_meals=0,
            total_transportation=100,
            total=1100,
        ),
    )


def violation_codes(violations):
    return [v.code for v in violations]


async def main():
    request = build_request()

    constraints = TravelConstraints(
        max_budget=600,
    )

    plan = build_intentionally_invalid_plan()

    initial_violations = validate_trip_plan(
        plan,
        request,
        constraints,
    )

    print("=" * 72)
    print("Live Auto-Replan Eval")
    print("=" * 72)

    print()
    print("Initial plan")
    print(f"- budget: {plan.budget.total}")
    print(f"- max_budget: {constraints.max_budget}")
    print(
        f"- violations: "
        f"{violation_codes(initial_violations)}"
    )

    if "BUDGET_EXCEEDED" not in violation_codes(
        initial_violations
    ):
        print()
        print(
            "❌ FAIL: 测试夹具没有成功制造 "
            "BUDGET_EXCEEDED"
        )
        raise SystemExit(1)

    state = {
        "request": request,
        "trip_plan": plan,
        "constraints": constraints.model_dump(),
        "violations": [
            v.model_dump()
            for v in initial_violations
        ],
        "edit_intent": {},
        "edit_feedback": "",
        "revision_count": 0,
        "plan_version": 0,
        "errors": [],
    }

    final_plan = plan
    final_violations = initial_violations
    revision_count = 0
    plan_version = 0

    for attempt in range(1, MAX_REPLAN + 1):
        print()
        print(f"--- Auto-Replan attempt {attempt} ---")

        result = await replan_node(state)

        revision_count = result.get(
            "revision_count",
            revision_count,
        )

        plan_version = result.get(
            "plan_version",
            plan_version,
        )

        if result.get("errors"):
            print(
                f"Replan errors: "
                f"{result['errors']}"
            )

        repaired = result.get("trip_plan")

        if repaired is None:
            print("没有返回新的 TripPlan。")
            break

        final_plan = repaired

        final_violations = validate_trip_plan(
            final_plan,
            request,
            constraints,
        )

        print(
            f"- repaired budget: "
            f"{final_plan.budget.total}"
        )
        print(
            f"- violations: "
            f"{violation_codes(final_violations)}"
        )
        print(
            f"- revision_count: "
            f"{revision_count}"
        )
        print(
            f"- plan_version: "
            f"{plan_version}"
        )

        if not final_violations:
            break

        state = {
            **state,
            "trip_plan": final_plan,
            "violations": [
                v.model_dump()
                for v in final_violations
            ],
            "revision_count": revision_count,
            "plan_version": plan_version,
            "errors": result.get(
                "errors",
                [],
            ),
        }

    final_codes = violation_codes(
        final_violations
    )

    budget_pass = (
        final_plan.budget is not None
        and final_plan.budget.total <= 600
    )

    replan_triggered = revision_count > 0

    version_pass = plan_version == 0

    validator_pass = (
        "BUDGET_EXCEEDED"
        not in final_codes
    )

    passed = all([
        replan_triggered,
        budget_pass,
        validator_pass,
        version_pass,
    ])

    print()
    print("=" * 72)
    print("Auto-Replan Eval Summary")
    print("=" * 72)

    print(
        f"Auto-Replan triggered: "
        f"{replan_triggered}"
    )
    print(
        f"Revision count:         "
        f"{revision_count}"
    )
    print(
        f"Final budget:           "
        f"{final_plan.budget.total}"
    )
    print(
        f"Budget <= 600:          "
        f"{budget_pass}"
    )
    print(
        f"Budget violation fixed: "
        f"{validator_pass}"
    )
    print(
        f"Plan version unchanged: "
        f"{version_pass}"
    )

    if passed:
        print()
        print("✅ PASS: Auto-Replan Recovery")
        return

    print()
    print("❌ FAIL: Auto-Replan Recovery")
    raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
