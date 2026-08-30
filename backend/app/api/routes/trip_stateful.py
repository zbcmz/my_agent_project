"""增强版 Stateful Travel Agent API。

保留原 /api/trip/plan 不动；本路由提供简历 Demo 所需的：
- 新建带 thread_id 的 Stateful 规划；
- HITL approve/edit 恢复；
- 已完成行程的多轮局部修改；
- Thread Memory / Long-term Preference Memory 查看与清理。
"""

from uuid import uuid4

from fastapi import APIRouter, HTTPException

from ...agents.enhanced_langgraph_agent import get_enhanced_trip_planner_agent
from ...models.agent_schemas import (
    AgentRunResponse,
    HITLDecision,
    StatefulPlanRequest,
    TripEditRequest,
)
from ...services.memory_service import get_user_memory_service


router = APIRouter(prefix="/trip-agent", tags=["增强版旅行 Agent"])


@router.post("/plan", response_model=AgentRunResponse, summary="创建 Stateful 旅行规划")
async def plan_trip_stateful(payload: StatefulPlanRequest):
    """新建 thread；enable_human_review=true 时最终会暂停等待用户审核。"""
    thread_id = payload.thread_id or f"trip-{uuid4().hex[:12]}"
    agent = get_enhanced_trip_planner_agent()
    return await agent.plan_trip(
        request=payload.request,
        thread_id=thread_id,
        user_id=payload.user_id,
        constraints=payload.constraints,
        enable_human_review=payload.enable_human_review,
    )


@router.post("/resume/{thread_id}", response_model=AgentRunResponse, summary="恢复 HITL")
async def resume_trip(thread_id: str, decision: HITLDecision):
    """使用同一 thread_id + Command(resume=...) 恢复 interrupt。"""
    agent = get_enhanced_trip_planner_agent()
    return await agent.resume(thread_id, decision)


@router.post("/edit/{thread_id}", response_model=AgentRunResponse, summary="多轮增量修改已有行程")
async def edit_trip(thread_id: str, payload: TripEditRequest):
    """例如：第二天下午太累，少一个景点，第一天不要改。"""
    agent = get_enhanced_trip_planner_agent()
    return await agent.edit_trip(
        thread_id=thread_id,
        feedback=payload.feedback,
        enable_human_review=payload.enable_human_review,
    )


@router.get("/thread/{thread_id}", summary="查看 LangGraph Thread Memory")
async def get_thread_state(thread_id: str):
    agent = get_enhanced_trip_planner_agent()
    try:
        return await agent.get_thread_state(thread_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"读取 thread 失败: {exc}")


@router.get("/memory/{user_id}", summary="查看用户长期偏好 Memory")
async def get_user_memory(user_id: str):
    service = get_user_memory_service()
    profile = service.load(user_id)
    return {"user_id": user_id, "profile": profile.model_dump()}


@router.delete("/memory/{user_id}", summary="清理用户长期偏好 Memory")
async def delete_user_memory(user_id: str):
    service = get_user_memory_service()
    service.delete(user_id)
    return {"success": True, "message": f"已清理 {user_id} 的长期 Memory"}
