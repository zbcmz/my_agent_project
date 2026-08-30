from app.services import memory_service


def _profile():
    return memory_service.UserPreferenceProfile(
        travel_preferences=["历史文化", "博物馆"],
        avoid_keywords=["网红店"],
        preferred_transportation="公共交通",
        preferred_accommodation="经济型酒店",
        preferred_food="本地特色",
        relaxed_pace=True,
        elderly_friendly=False,
    )


def test_user_memory_save_and_load(tmp_path):
    """保存后应能从 SQLite 正确读取长期用户偏好。"""
    db_path = tmp_path / "user_memory_test.db"

    service = memory_service.UserMemoryService(
        db_path=str(db_path)
    )

    user_id = "pytest-memory-user"
    expected = _profile()

    service.save(user_id, expected)

    actual = service.load(user_id)

    assert actual.travel_preferences == [
        "历史文化",
        "博物馆",
    ]
    assert actual.avoid_keywords == ["网红店"]
    assert actual.preferred_transportation == "公共交通"
    assert actual.preferred_accommodation == "经济型酒店"
    assert actual.preferred_food == "本地特色"
    assert actual.relaxed_pace is True
    assert actual.elderly_friendly is False


def test_user_memory_survives_new_service_instance(tmp_path):
    """重新创建 Service 后仍应读取到同一 SQLite 中的数据。"""
    db_path = tmp_path / "user_memory_test.db"
    user_id = "pytest-persistent-user"

    service_a = memory_service.UserMemoryService(
        db_path=str(db_path)
    )
    service_a.save(user_id, _profile())

    # 模拟新的 Service / Backend 实例重新连接数据库。
    service_b = memory_service.UserMemoryService(
        db_path=str(db_path)
    )

    actual = service_b.load(user_id)

    assert actual.travel_preferences == [
        "历史文化",
        "博物馆",
    ]
    assert actual.preferred_transportation == "公共交通"
    assert actual.relaxed_pace is True


def test_user_memory_isolated_by_user_id(tmp_path):
    """不同 user_id 的长期 Memory 不应串数据。"""
    db_path = tmp_path / "user_memory_test.db"

    service = memory_service.UserMemoryService(
        db_path=str(db_path)
    )

    service.save(
        "user-a",
        memory_service.UserPreferenceProfile(
            travel_preferences=["历史文化"],
            preferred_transportation="公共交通",
        ),
    )

    service.save(
        "user-b",
        memory_service.UserPreferenceProfile(
            travel_preferences=["自然风光"],
            preferred_transportation="自驾",
        ),
    )

    user_a = service.load("user-a")
    user_b = service.load("user-b")

    assert user_a.travel_preferences == ["历史文化"]
    assert user_a.preferred_transportation == "公共交通"

    assert user_b.travel_preferences == ["自然风光"]
    assert user_b.preferred_transportation == "自驾"


def test_user_memory_delete(tmp_path):
    """删除用户 Memory 后应恢复为空 Profile。"""
    db_path = tmp_path / "user_memory_test.db"
    user_id = "pytest-delete-user"

    service = memory_service.UserMemoryService(
        db_path=str(db_path)
    )

    service.save(user_id, _profile())
    assert service.load(user_id).travel_preferences

    service.delete(user_id)

    actual = service.load(user_id)

    assert actual.travel_preferences == []
    assert actual.avoid_keywords == []
    assert actual.preferred_transportation is None
    assert actual.preferred_accommodation is None
    assert actual.preferred_food is None


def test_anonymous_user_is_not_persisted(tmp_path):
    """anonymous 用户不应写入长期 Memory。"""
    db_path = tmp_path / "user_memory_test.db"

    service = memory_service.UserMemoryService(
        db_path=str(db_path)
    )

    service.save("anonymous", _profile())

    actual = service.load("anonymous")

    assert actual.travel_preferences == []
    assert actual.preferred_transportation is None
