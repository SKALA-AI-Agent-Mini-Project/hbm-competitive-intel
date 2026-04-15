"""
Supervisor Agent — HBM Competitive R&D Intelligence System
통합: 팀 개선(벡터스토어 가드, retry 로직, format helpers)
     + TRLEvaluator 라우팅(교수 피드백)
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import AgentState
from prompts.supervisor_prompt import SUPERVISOR_ROUTE_PROMPT, SUPERVISOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0)
VECTORSTORE_FILE = Path("rag/vectorstore/index.faiss")


def _format_rag_status(state: AgentState) -> str:
    rag = state.get("rag_results")
    if not rag:
        return "미수집"
    return f"{len(rag)}건 수집됨 (출처: {', '.join(r['source'] for r in rag[:3])})"


def _format_web_status(state: AgentState) -> str:
    web = state.get("web_results")
    if not web:
        return "미수집"
    return f"{len(web)}건 수집됨 (최신: {web[0].get('date', 'N/A')})"


def _format_trl_status(state: AgentState) -> str:
    trl = state.get("trl_assessment")
    if not trl:
        return "미완료"
    techs = list(trl.get("technologies", {}).keys())
    return f"완료 (평가 기술: {', '.join(techs)})"


def _format_recent_messages(state: AgentState, n: int = 3) -> str:
    messages = state.get("messages", [])
    recent = messages[-n:] if len(messages) >= n else messages
    lines = []
    for m in recent:
        role = m.__class__.__name__.replace("Message", "")
        content = m.content[:200] if isinstance(m.content, str) else str(m.content)[:200]
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines) if lines else "없음"


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    query       = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    rag_results = state.get("rag_results")
    web_results = state.get("web_results")
    trl_assessment = state.get("trl_assessment")
    report      = state.get("report")

    # ── 0. 보고서 완성 → 즉시 종료 ──────────────────────────────
    if report:
        return _make_return("END", "보고서가 이미 생성되어 종료합니다.", "", retry_count)

    # ── 1. 벡터스토어 없으면 RAGAgent 우회 ──────────────────────
    if not VECTORSTORE_FILE.exists():
        if web_results:
            return _make_return(
                "ReportGen" if not trl_assessment else "ReportGen",
                "벡터스토어 없음 — Web 결과로 보고서 작성.",
                "RAG 자료 없으므로 한계점 명시 후 보고서 작성.",
                retry_count,
            )
        return _make_return(
            "WebAgent",
            "벡터스토어 없음 — WebAgent로 전환.",
            "공개 웹 정보로 최신 동향 수집.",
            retry_count,
        )

    # ── 2. RAG+Web 완료 & TRL 미완료 → TRLEvaluator (교수 피드백) ─
    if rag_results and web_results and not trl_assessment:
        return _make_return(
            "TRLEvaluator",
            "RAG+Web 수집 완료 — TRL 평가 단계로 이동.",
            "",
            retry_count,
        )

    # ── 3. RAG+Web+TRL 완료 → ReportGen ────────────────────────
    if rag_results and web_results and trl_assessment:
        return _make_return(
            "ReportGen",
            "수집·TRL 평가 완료 — 보고서 생성 단계로 이동.",
            "TRL 평가 결과를 반드시 3.3절 매트릭스에 반영하세요.",
            retry_count,
        )

    # ── 4. LLM 라우팅 (RAG/Web 미완료 시) ──────────────────────
    route_prompt = SUPERVISOR_ROUTE_PROMPT.format(
        query=query,
        rag_status=_format_rag_status(state),
        web_status=_format_web_status(state),
        retry_count=retry_count,
        recent_messages=_format_recent_messages(state),
    )

    try:
        response = _supervisor_llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=route_prompt),
        ])
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        decision   = json.loads(raw)
        next_node  = decision.get("next", "END")
        instruction = decision.get("instruction", "")
        reason     = decision.get("reason", "")
    except Exception as e:
        logger.warning(f"Supervisor 파싱 실패: {e} → END")
        return _make_return("END", "파싱 오류로 종료.", "", retry_count)

    # ── 5. retry 카운터 증가 및 한도 초과 처리 ──────────────────
    next_retry = retry_count + (1 if next_node == "RAGAgent" else 0)
    if next_node == "RAGAgent" and next_retry >= 2:
        next_node   = "WebAgent" if not web_results else "ReportGen"
        reason      = "RAG 재시도 한도 초과 — 전환."
        instruction = "현재 수집 정보 기반으로 한계점 명시 후 보고서 작성."

    logger.info(f"Supervisor → {next_node} | {reason}")
    return _make_return(next_node, reason, instruction, next_retry)


def _make_return(next_node: str, reason: str, instruction: str, retry_count: int) -> dict:
    msg = AIMessage(content=f"[Supervisor → {next_node}] {reason}\n지시: {instruction}")
    return {"messages": [msg], "next": next_node, "retry_count": retry_count}


def get_next(state: AgentState) -> str:
    return state.get("next", "END")
