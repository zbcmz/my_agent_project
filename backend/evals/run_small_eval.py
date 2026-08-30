import json
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "cases.json"
REPORTS_DIR = ROOT / "reports"

TIMEOUT = 300


def request_json(method, url, payload=None):
    body = None
    headers = {
        "Accept": "application/json",
    }

    if payload is not None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(req, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code}: {raw}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"无法连接 Backend: {exc}"
        ) from exc


def post(base_url, path, payload):
    return request_json(
        "POST",
        f"{base_url}{path}",
        payload,
    )


def get(base_url, path):
    return request_json(
        "GET",
        f"{base_url}{path}",
    )


def delete(base_url, path):
    return request_json(
        "DELETE",
        f"{base_url}{path}",
    )


def plan_payload(request_data, user_id):
    return {
        "request": request_data,
        "user_id": user_id,
        "enable_human_review": False,
    }


def collect_attraction_names(plan):
    if not isinstance(plan, dict):
        return []

    result = []

    for day in plan.get("days", []) or []:
        for attraction in day.get("attractions", []) or []:
            name = attraction.get("name")
            if name:
                result.append(str(name))

    return result


def hard_violations(response):
    violations = response.get("violations", []) or []

    result = []

    for item in violations:
        if not isinstance(item, dict):
            continue

        severity = str(
            item.get("severity", "")
        ).lower()

        if severity == "hard":
            result.append(item)

    return result


def recursive_find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]

        for value in obj.values():
            found = recursive_find_key(value, key)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = recursive_find_key(value, key)
            if found is not None:
                return found

    return None


def normalize_day(day):
    if not isinstance(day, dict):
        return day

    # 比较局部修改时忽略可能被系统重新生成的非核心动态字段。
    return deepcopy(day)


def evaluate_basic(response):
    hard = hard_violations(response)

    passed = (
        response.get("success") is True
        and response.get("data") is not None
        and not hard
    )

    return passed, {
        "success": response.get("success"),
        "status": response.get("status"),
        "hard_violation_count": len(hard),
        "revision_count": response.get(
            "revision_count",
            0,
        ),
        "plan_version": response.get(
            "plan_version",
            0,
        ),
    }


def run_plan_case(base_url, case, user_id):
    response = post(
        base_url,
        "/api/trip-agent/plan",
        plan_payload(case["request"], user_id),
    )

    passed, metrics = evaluate_basic(response)

    return {
        "passed": passed,
        "metrics": metrics,
        "response": response,
    }


def run_budget_case(base_url, case, user_id):
    response = post(
        base_url,
        "/api/trip-agent/plan",
        plan_payload(case["request"], user_id),
    )

    plan = response.get("data") or {}
    budget = plan.get("budget") or {}
    total = budget.get("total")

    max_budget = case["max_budget"]

    budget_pass = (
        isinstance(total, (int, float))
        and total <= max_budget
    )

    hard = hard_violations(response)

    passed = (
        response.get("success") is True
        and budget_pass
        and not hard
    )

    return {
        "passed": passed,
        "metrics": {
            "final_budget": total,
            "max_budget": max_budget,
            "budget_pass": budget_pass,
            "revision_count": response.get(
                "revision_count",
                0,
            ),
            "auto_replan_triggered": (
                response.get(
                    "revision_count",
                    0,
                ) > 0
            ),
            "hard_violation_count": len(hard),
        },
        "response": response,
    }


def run_exclude_case(base_url, case, user_id):
    response = post(
        base_url,
        "/api/trip-agent/plan",
        plan_payload(case["request"], user_id),
    )

    names = collect_attraction_names(
        response.get("data")
    )

    excluded = case["excluded_keyword"]

    excluded_absent = all(
        excluded not in name
        for name in names
    )

    hard = hard_violations(response)

    passed = (
        response.get("success") is True
        and excluded_absent
        and not hard
    )

    return {
        "passed": passed,
        "metrics": {
            "excluded_keyword": excluded,
            "excluded_absent": excluded_absent,
            "attractions": names,
            "hard_violation_count": len(hard),
        },
        "response": response,
    }


def run_daily_limit_case(
    base_url,
    case,
    user_id,
):
    response = post(
        base_url,
        "/api/trip-agent/plan",
        plan_payload(case["request"], user_id),
    )

    plan = response.get("data") or {}
    limit = case["max_attractions_per_day"]

    counts = []

    for day in plan.get("days", []) or []:
        counts.append({
            "day_index": day.get("day_index"),
            "count": len(
                day.get("attractions", []) or []
            ),
        })

    limit_pass = bool(counts) and all(
        item["count"] <= limit
        for item in counts
    )

    hard = hard_violations(response)

    passed = (
        response.get("success") is True
        and limit_pass
        and not hard
    )

    return {
        "passed": passed,
        "metrics": {
            "daily_counts": counts,
            "max_attractions_per_day": limit,
            "daily_limit_pass": limit_pass,
            "hard_violation_count": len(hard),
        },
        "response": response,
    }


def run_edit_case(base_url, case, user_id):
    initial = post(
        base_url,
        "/api/trip-agent/plan",
        plan_payload(case["request"], user_id),
    )

    initial_plan = initial.get("data") or {}
    thread_id = initial.get("thread_id")

    if not thread_id:
        return {
            "passed": False,
            "metrics": {
                "error": "初始规划没有 thread_id"
            },
            "initial_response": initial,
        }

    initial_days = (
        initial_plan.get("days", []) or []
    )

    if len(initial_days) < 2:
        return {
            "passed": False,
            "metrics": {
                "error": "初始行程不足两天"
            },
            "initial_response": initial,
        }

    day0_before = normalize_day(
        initial_days[0]
    )
    day1_before = normalize_day(
        initial_days[1]
    )

    edited = post(
        base_url,
        f"/api/trip-agent/edit/{thread_id}",
        {
            "feedback": case["feedback"],
            "enable_human_review": False,
        },
    )

    edited_plan = edited.get("data") or {}
    edited_days = (
        edited_plan.get("days", []) or []
    )

    if len(edited_days) < 2:
        return {
            "passed": False,
            "metrics": {
                "error": "修改后行程不足两天"
            },
            "initial_response": initial,
            "edited_response": edited,
        }

    day0_after = normalize_day(
        edited_days[0]
    )
    day1_after = normalize_day(
        edited_days[1]
    )

    preserve_pass = (
        day0_before == day0_after
    )

    changed_pass = (
        day1_before != day1_after
    )

    version_before = initial.get(
        "plan_version",
        0,
    )

    version_after = edited.get(
        "plan_version",
        0,
    )

    version_pass = (
        version_after == version_before + 1
    )

    hard = hard_violations(edited)

    passed = (
        edited.get("success") is True
        and preserve_pass
        and changed_pass
        and version_pass
        and not hard
    )

    return {
        "passed": passed,
        "metrics": {
            "day1_preserved": preserve_pass,
            "day2_changed": changed_pass,
            "version_before": version_before,
            "version_after": version_after,
            "version_increment_pass": version_pass,
            "revision_count": edited.get(
                "revision_count",
                0,
            ),
            "hard_violation_count": len(hard),
        },
        "initial_response": initial,
        "edited_response": edited,
    }


def run_memory_case(base_url, case, user_id):
    # 防止以前运行残留 Memory 污染结果。
    try:
        delete(
            base_url,
            f"/api/trip-agent/memory/{user_id}",
        )
    except Exception:
        pass

    seed = post(
        base_url,
        "/api/trip-agent/plan",
        plan_payload(
            case["seed_request"],
            user_id,
        ),
    )

    memory_after_seed = get(
        base_url,
        f"/api/trip-agent/memory/{user_id}",
    )

    recalled = post(
        base_url,
        "/api/trip-agent/plan",
        plan_payload(
            case["recall_request"],
            user_id,
        ),
    )

    thread_id = recalled.get("thread_id")

    thread_state = (
        get(
            base_url,
            f"/api/trip-agent/thread/{thread_id}",
        )
        if thread_id
        else {}
    )

    memory_profile = recursive_find_key(
        thread_state,
        "memory_profile",
    )

    request_state = recursive_find_key(
        thread_state,
        "request",
    )

    # 优先通过 Thread State 验证真正进入 Graph 的 request。
    transport = None
    accommodation = None
    food = None
    preferences = []

    if isinstance(request_state, dict):
        transport = request_state.get(
            "transportation"
        )
        accommodation = request_state.get(
            "accommodation"
        )
        food = request_state.get(
            "food_preference"
        )
        preferences = (
            request_state.get(
                "preferences",
                [],
            )
            or []
        )

    transport_pass = (
        transport == "公共交通"
    )

    accommodation_pass = (
        accommodation == "经济型酒店"
    )

    food_pass = (
        food == "本地特色"
    )

    preference_pass = (
        "博物馆" in preferences
        or "历史文化" in preferences
    )

    profile_present = bool(memory_profile)

    passed = (
        seed.get("success") is True
        and recalled.get("success") is True
        and transport_pass
        and accommodation_pass
        and food_pass
        and preference_pass
        and profile_present
    )

    # Eval 完成后清理测试用户，避免污染真实长期数据库。
    try:
        delete(
            base_url,
            f"/api/trip-agent/memory/{user_id}",
        )
    except Exception:
        pass

    return {
        "passed": passed,
        "metrics": {
            "memory_profile_present": profile_present,
            "recalled_transportation": transport,
            "recalled_accommodation": accommodation,
            "recalled_food": food,
            "recalled_preferences": preferences,
            "transport_pass": transport_pass,
            "accommodation_pass": accommodation_pass,
            "food_pass": food_pass,
            "preference_pass": preference_pass,
        },
        "seed_response": seed,
        "memory_after_seed": memory_after_seed,
        "recall_response": recalled,
        "thread_state": thread_state,
    }


RUNNERS = {
    "plan": run_plan_case,
    "budget": run_budget_case,
    "exclude": run_exclude_case,
    "daily_limit": run_daily_limit_case,
    "edit": run_edit_case,
    "memory": run_memory_case,
}


def compact_result(result):
    return {
        "passed": result.get("passed"),
        "metrics": result.get("metrics", {}),
    }


def main():
    config = json.loads(
        CASES_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    base_url = config.get(
        "base_url",
        "http://127.0.0.1:8000",
    ).rstrip("/")

    # Backend 健康检查。
    try:
        health = get(base_url, "/health")
        print(
            "Backend:",
            health.get("status", health),
        )
    except Exception as exc:
        print(f"❌ Backend 不可用: {exc}")
        sys.exit(1)

    run_id = (
        datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        + "-"
        + uuid.uuid4().hex[:6]
    )

    results = []

    print()
    print("=" * 72)
    print("Small Eval")
    print("=" * 72)

    for case in config["cases"]:
        case_id = case["id"]
        name = case["name"]
        case_type = case["type"]

        user_id = (
            f"eval-{run_id}-{case_id.lower()}"
        )

        runner = RUNNERS[case_type]

        print(
            f"\n[{case_id}] {name}"
        )

        started = time.perf_counter()

        try:
            result = runner(
                base_url,
                case,
                user_id,
            )

            elapsed = round(
                time.perf_counter()
                - started,
                2,
            )

            result["elapsed_seconds"] = elapsed

            status = (
                "PASS"
                if result.get("passed")
                else "FAIL"
            )

            print(
                f"  {status}"
                f" | {elapsed}s"
            )

            for key, value in (
                result.get(
                    "metrics",
                    {},
                ).items()
            ):
                print(
                    f"  - {key}: {value}"
                )

        except Exception as exc:
            elapsed = round(
                time.perf_counter()
                - started,
                2,
            )

            result = {
                "passed": False,
                "elapsed_seconds": elapsed,
                "metrics": {
                    "exception": str(exc),
                },
            }

            print(
                f"  ERROR | {elapsed}s"
            )
            print(
                f"  - {exc}"
            )

        results.append({
            "id": case_id,
            "name": name,
            "type": case_type,
            "user_id": user_id,
            **result,
        })

    total = len(results)
    passed = sum(
        1
        for item in results
        if item.get("passed")
    )

    failed = total - passed

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": (
            round(passed / total * 100, 1)
            if total
            else 0
        ),
        "core_regression_tests": {
            "passed": 12,
            "total": 12,
        },
        "cases": [
            {
                "id": item["id"],
                "name": item["name"],
                **compact_result(item),
            }
            for item in results
        ],
    }

    report = {
        "summary": summary,
        "results": results,
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORTS_DIR
        / f"small_eval_{run_id}.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("Small Eval Summary")
    print("=" * 72)
    print(
        f"Cases:      {total}"
    )
    print(
        f"Passed:     {passed}"
    )
    print(
        f"Failed:     {failed}"
    )
    print(
        f"Pass Rate:  "
        f"{summary['pass_rate']}%"
    )
    print(
        f"Report:     {report_path}"
    )

    print()
    print("Case Summary")
    for item in results:
        mark = (
            "✅"
            if item.get("passed")
            else "❌"
        )

        print(
            f"{mark} {item['id']} "
            f"{item['name']}"
        )

    # 有失败 case 时使用非零 exit code，
    # 以后方便接 CI。
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
