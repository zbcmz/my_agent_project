import asyncio
from unittest.mock import patch

from app.agents.enhanced_langgraph_agent import load_memory_node
from app.models.schemas import TripRequest
from app.services.memory_service import UserPreferenceProfile


class FakeMemoryService:
    def __init__(self, profile):
        self.profile = profile

    def load(self, user_id):
        return self.profile


def _memory_profile():
    return UserPreferenceProfile(
        travel_preferences=["历史文化"],
        avoid_keywords=["网红店"],
        preferred_transportation="公共交通",
        preferred_accommodation="经济型酒店",
        preferred_food="本地特色",
        relaxed_pace=True,
        elderly_friendly=False,
    )


def _request(
    *,
    transportation="",
    accommodation="",
    food_preference="无特殊要求",
    preferences=None,
):
    return TripRequest(
        city="北京",
        start_date="2026-09-01",
        end_date="2026-09-02",
        travel_days=2,
        transportation=transportation,
        accommodation=accommodation,
        preferences=preferences or [],
        food_preference=food_preference,
        free_text_input="",
    )


def test_load_memory_fills_missing_fields():
    """本轮没有明确填写时，应继承长期 Memory。"""
    service = FakeMemoryService(_memory_profile())

    state = {
        "user_id": "memory-test-user",
        "request": _request(
            preferences=["博物馆"],
        ),
    }

    with patch(
        "app.agents.enhanced_langgraph_agent.get_user_memory_service",
        return_value=service,
    ):
        result = asyncio.run(load_memory_node(state))

    request = result["request"]

    assert request.preferences == ["博物馆", "历史文化"]
    assert request.transportation == "公共交通"
    assert request.accommodation == "经济型酒店"
    assert request.food_preference == "本地特色"

    assert result["memory_profile"]["relaxed_pace"] is True


def test_explicit_request_overrides_memory():
    """用户本轮明确选择必须优先于历史 Memory。"""
    service = FakeMemoryService(_memory_profile())

    state = {
        "user_id": "memory-test-user",
        "request": _request(
            transportation="自驾",
            accommodation="精品酒店",
            food_preference="素食",
            preferences=["自然风光"],
        ),
    }

    with patch(
        "app.agents.enhanced_langgraph_agent.get_user_memory_service",
        return_value=service,
    ):
        result = asyncio.run(load_memory_node(state))

    request = result["request"]

    # preferences 可以合并长期偏好
    assert request.preferences == ["自然风光", "历史文化"]

    # 但显式选择绝不能被长期 Memory 覆盖
    assert request.transportation == "自驾"
    assert request.accommodation == "精品酒店"
    assert request.food_preference == "素食"
