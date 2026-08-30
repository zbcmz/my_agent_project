"""旅行 Agent 长期 Memory 服务。

设计意图：
1. LangGraph Checkpointer 负责 thread-level 短期状态（当前行程、执行位置、HITL 恢复）；
2. 本服务负责 user-level 长期偏好（跨 thread 共享）。

为降低本地 Demo 成本，这里直接使用 Python 标准库 sqlite3。
生产环境可替换为 PostgreSQL/Redis/向量数据库，而不影响 Graph 节点接口。
"""

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..models.agent_schemas import UserPreferenceProfile


class UserMemoryService:
    """轻量 SQLite 用户偏好存储。"""

    def __init__(self, db_path: Optional[str] = None):
        default_path = Path(__file__).resolve().parents[3] / "data" / "user_memory.db"
        self.db_path = Path(db_path or os.getenv("TRIP_MEMORY_DB", str(default_path)))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def load(self, user_id: str) -> UserPreferenceProfile:
        """读取用户长期偏好；匿名用户不共享 Memory。"""
        if not user_id or user_id == "anonymous":
            return UserPreferenceProfile()

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if not row:
            return UserPreferenceProfile()

        try:
            return UserPreferenceProfile.model_validate(json.loads(row[0]))
        except Exception:
            # Memory 数据损坏时宁可降级为空偏好，也不要阻断旅行主链路。
            return UserPreferenceProfile()

    def save(self, user_id: str, profile: UserPreferenceProfile) -> None:
        """覆盖保存用户偏好。"""
        if not user_id or user_id == "anonymous":
            return

        payload = json.dumps(profile.model_dump(), ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences(user_id, profile_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, payload),
            )
            conn.commit()

    def delete(self, user_id: str) -> None:
        """删除用户长期 Memory，方便调试/演示。"""
        if not user_id or user_id == "anonymous":
            return
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
            conn.commit()


_memory_service: Optional[UserMemoryService] = None
_memory_lock = threading.Lock()


def get_user_memory_service() -> UserMemoryService:
    """线程安全单例。"""
    global _memory_service
    if _memory_service is not None:
        return _memory_service
    with _memory_lock:
        if _memory_service is None:
            _memory_service = UserMemoryService()
    return _memory_service
