"""
AgentState — HBM Competitive R&D Intelligence System
설계 산출물 3-1절 기반 구현
통합: RAGResult/WebResult TypedDict (팀 개선) + trl_assessment (교수 피드백)
"""
import operator
from typing import Annotated, Dict, List, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class RAGResult(TypedDict):
    chunk: str       # 검색된 문서 청크
    summary: str     # 청크 요약
    source: str      # 출처 (파일명, 페이지 등)
    score: float     # 유사도 점수


class WebResult(TypedDict):
    title: str       # 문서/기사 제목
    url: str         # 원문 URL
    date: str        # 발행일
    summary: str     # 내용 요약
    trl_clue: str    # TRL 추정 간접지표


class AgentState(TypedDict):
    # ── 공통 ───────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str

    # ── 수집 결과 ───────────────────────────────────────────────
    rag_results: Optional[List[RAGResult]]
    web_results: Optional[List[WebResult]]

    # ── TRL 평가 결과 (교수 피드백 — 전용 노드 산출물) ──────────
    trl_assessment: Optional[Dict]

    # ── 보고서 ─────────────────────────────────────────────────
    report: Optional[str]

    # ── Supervisor 제어 ────────────────────────────────────────
    next: str
    retry_count: int
