"""
LangGraph Workflow — HBM Competitive R&D Intelligence System
설계 산출물 3-3절 기반 구현
통합: 팀 구조(build_workflow/compile_graph) + TRLEvaluator 노드(교수 피드백)

흐름:
  START → Supervisor
  Supervisor → RAGAgent | WebAgent | TRLEvaluator | ReportGen | END
  *Agent → Supervisor  (복귀)
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.report_generator import report_generator_node
from agents.supervisor import get_next, supervisor_node
from agents.rag_agent import rag_agent_node
from agents.web_search_agent import web_search_node
from agents.trl_evaluator import trl_evaluator_node
from graph.state import AgentState


def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # ── 노드 등록 ─────────────────────────────────────────────
    workflow.add_node("Supervisor",    supervisor_node)
    workflow.add_node("RAGAgent",      rag_agent_node)
    workflow.add_node("WebAgent",      web_search_node)
    workflow.add_node("TRLEvaluator",  trl_evaluator_node)   # 교수 피드백
    workflow.add_node("ReportGen",     report_generator_node)

    # ── 진입점 ───────────────────────────────────────────────
    workflow.add_edge(START, "Supervisor")

    # ── Sub-Agent → Supervisor 복귀 ──────────────────────────
    workflow.add_edge("RAGAgent",     "Supervisor")
    workflow.add_edge("WebAgent",     "Supervisor")
    workflow.add_edge("TRLEvaluator", "Supervisor")
    workflow.add_edge("ReportGen",    "Supervisor")

    # ── Supervisor Conditional Branch ────────────────────────
    workflow.add_conditional_edges(
        "Supervisor",
        get_next,
        {
            "RAGAgent":     "RAGAgent",
            "WebAgent":     "WebAgent",
            "TRLEvaluator": "TRLEvaluator",
            "ReportGen":    "ReportGen",
            "END":          END,
        },
    )
    return workflow


def compile_graph(checkpointer=None):
    """체크포인터 포함 그래프 컴파일"""
    workflow = build_workflow()
    cp = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=cp)


# ── 하위 호환 alias (graph/__init__.py, main.ipynb, app.py 호환) ──────
def build_graph(use_checkpointer: bool = True):
    """compile_graph()의 alias — 기존 코드 호환용."""
    from langgraph.checkpoint.memory import MemorySaver
    cp = MemorySaver() if use_checkpointer else None
    return build_workflow().compile(checkpointer=cp) if cp else build_workflow().compile()


def get_initial_state(query: str) -> dict:
    """워크플로우 초기 State 생성."""
    from langchain_core.messages import HumanMessage
    return {
        "messages":               [HumanMessage(content=query)],
        "query":                  query,
        "rag_results":            None,
        "web_results":            None,
        "trl_assessment":         None,
        "report":                 None,
        "next":                   "",
        "retry_count":            0,
    }