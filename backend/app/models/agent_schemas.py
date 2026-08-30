"""增强版旅行 Agent 使用的数据模型。

这些模型与原项目 ``models/schemas.py`` 解耦，避免破坏原有前端/API。
主要服务于：Supervisor、约束校验、HITL、多轮修改与 Memory。
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .schemas import DayPlan, TripPlan, TripRequest


class TravelConstraints(BaseModel):
    """从自然语言中解析出的旅行约束。

    说明：
    - 能程序化判断的约束放在这里，交给 Validator 做确定性校验；
    - 老人友好、轻松节奏等语义偏好也会保存，但主要用于 Planner/Replan 提示。
    """

    max_budget: Optional[int] = Field(default=None, description="总预算上限（元）")
    max_attractions_per_day: Optional[int] = Field(default=None, ge=1, le=10, description="每天最多景点数")
    excluded_keywords: List[str] = Field(default_factory=list, description="明确禁止/排除的活动或景点关键词")
    must_include_keywords: List[str] = Field(default_factory=list, description="明确要求包含的主题/景点关键词")
    elderly_friendly: bool = Field(default=False, description="是否需要老人友好")
    relaxed_pace: bool = Field(default=False, description="是否偏好轻松/慢节奏")
    hard_notes: List[str] = Field(default_factory=list, description="其他难以结构化、但需要严格遵守的硬约束")


class Violation(BaseModel):
    """Validator 输出的结构化违规项。"""

    code: str = Field(..., description="稳定的违规编码，便于 Eval 统计")
    message: str = Field(..., description="面向 Planner/HITL 的中文说明")
    severity: Literal["hard", "soft"] = "hard"
    affected_days: List[int] = Field(default_factory=list, description="受影响 day_index；为空表示全局问题")
    expected: Optional[str] = None
    actual: Optional[str] = None


class SupervisorDecision(BaseModel):
    """Supervisor 的任务路由决策。

    route/planner/validator 属于核心链路；这里主要决定哪些 Research Worker 值得执行。
    """

    workers: List[Literal["poi", "weather", "hotel", "food", "route"]] = Field(
        default_factory=lambda: ["poi", "weather", "hotel", "food", "route"]
    )
    routing_reason: str = Field(default="默认执行完整旅行研究链路", description="简短路由原因，不要求模型输出思维链")


class EditIntent(BaseModel):
    """将用户的自然语言修改意见解析为局部重规划范围。"""

    affected_days: List[int] = Field(default_factory=list, description="需要修改的 day_index（从 0 开始）")
    preserve_days: List[int] = Field(default_factory=list, description="明确要求保持不变的 day_index")
    intent_summary: str = Field(default="", description="修改意图摘要")


class UserPreferenceProfile(BaseModel):
    """跨 thread 的长期用户偏好 Memory。"""

    travel_preferences: List[str] = Field(default_factory=list)
    avoid_keywords: List[str] = Field(default_factory=list)
    preferred_transportation: Optional[str] = None
    preferred_accommodation: Optional[str] = None
    preferred_food: Optional[str] = None
    relaxed_pace: bool = False
    elderly_friendly: bool = False


class StatefulPlanRequest(BaseModel):
    """增强版规划接口请求。

    原项目 TripRequest 保持不变，因此前端老接口仍然能继续使用。
    """

    request: TripRequest
    user_id: str = Field(default="anonymous", description="长期 Memory 的用户标识")
    thread_id: Optional[str] = Field(default=None, description="LangGraph thread_id；为空时后端自动生成")
    constraints: Optional[TravelConstraints] = Field(default=None, description="可选：显式结构化约束；为空则从自然语言解析")
    enable_human_review: bool = Field(default=True, description="是否在最终返回前进入 HITL 审核")


class HITLDecision(BaseModel):
    """恢复 interrupt 时的人类决策。"""

    action: Literal["approve", "edit"]
    feedback: str = Field(default="", description="action=edit 时的修改意见")


class TripEditRequest(BaseModel):
    """已完成行程的多轮增量修改请求。"""

    feedback: str = Field(..., min_length=1, description="例如：第二天下午太累，删掉一个景点，第一天不要改")
    enable_human_review: bool = True


class AgentRunResponse(BaseModel):
    """增强版 Agent 统一响应。"""

    success: bool
    status: Literal["completed", "waiting_human", "failed"]
    message: str
    thread_id: str
    data: Optional[TripPlan] = None
    interrupt: Optional[Dict] = None
    constraints: Optional[TravelConstraints] = None
    violations: List[Violation] = Field(default_factory=list)
    # Auto-Replan 本轮尝试次数
    revision_count: int = 0

    # 用户可见的行程版本号
    plan_version: int = 0

    supervisor: Optional[SupervisorDecision] = None


class ReplanOutput(BaseModel):
    """Replan 节点的结构化输出。

    让模型只返回需要替换的日期，降低全量重生成导致的无关内容漂移。
    """
    """Replan 节点的结构化输出。"""

    revised_days: List[DayPlan] = Field(
        default_factory=list
    )

    overall_suggestions: Optional[str] = Field(
        default=None,
        description=(
            "基于修改后的最新完整行程生成的总体建议。"
            "不得提及已经删除或被替换的安排。"
        ),
    )
