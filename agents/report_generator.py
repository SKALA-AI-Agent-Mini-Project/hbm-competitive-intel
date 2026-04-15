"""
Report Generator — HBM Competitive R&D Intelligence System
통합: 팀 개선(RAGResult/WebResult TypedDict, 섹션 검증, outputs/ 저장)
     + trl_assessment 활용(교수 피드백)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import AgentState, RAGResult, WebResult
from prompts.report_prompt import REPORT_GENERATION_PROMPT, REPORT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")


def _format_rag_results(rag_results: List[RAGResult]) -> str:
    if not rag_results:
        return "[RAG 검색 결과 없음]"
    lines = []
    for i, r in enumerate(rag_results, 1):
        lines.append(
            f"[{i}] 출처: {r['source']} (유사도: {r['score']:.3f})\n"
            f"    요약: {r['summary']}\n"
            f"    원문(일부): {r['chunk'][:300]}"
        )
    return "\n\n".join(lines)


def _format_web_results(web_results: List[WebResult]) -> str:
    if not web_results:
        return "[Web 검색 결과 없음]"
    lines = []
    for i, r in enumerate(web_results, 1):
        lines.append(
            f"[{i}] 제목: {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    날짜: {r['date']}\n"
            f"    요약: {r['summary'][:300]}\n"
            f"    TRL 단서: {r['trl_clue']}"
        )
    return "\n\n".join(lines)


def _format_trl_assessment(trl: Dict) -> str:
    """TRL 평가 결과를 보고서 입력용 텍스트로 변환 (교수 피드백)."""
    if not trl:
        return "[TRL 평가 결과 없음]"
    try:
        techs = trl.get("technologies", {})
        lines = ["## TRL 평가 결과 (TRLEvaluator 산출)"]
        for tech, companies in techs.items():
            lines.append(f"\n### {tech}")
            for company, info in companies.items():
                lines.append(
                    f"- {company}: TRL {info.get('trl','?')} "
                    f"| 신뢰도: {info.get('confidence','?')} "
                    f"| {info.get('note','')}\n"
                    f"  근거: {'; '.join(info.get('evidence', []))}"
                )
        threats = trl.get("threat_summary", {})
        if threats:
            lines.append("\n### 위협 요약")
            for tech, t in threats.items():
                lines.append(f"- {tech}: {t}")
        return "\n".join(lines)
    except Exception:
        return json.dumps(trl, ensure_ascii=False)[:500]


def _save_report(content: str, query: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "-_" else "_" for c in query[:30])
    path       = OUTPUT_DIR / f"report_{safe_query}_{ts}.md"
    path.write_text(content, encoding="utf-8")
    logger.info(f"보고서 저장: {path}")
    return path


def report_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Report Generator 노드.
    RAG + Web + TRL 평가 결과를 통합하여 Markdown 보고서 초안 작성.
    """
    llm            = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4096)
    query          = state.get("query", "")
    rag_results    = state.get("rag_results") or []
    web_results    = state.get("web_results") or []
    trl_assessment = state.get("trl_assessment")

    rag_formatted = _format_rag_results(rag_results)
    web_formatted = _format_web_results(web_results)
    trl_formatted = _format_trl_assessment(trl_assessment)   # 교수 피드백

    generation_prompt = REPORT_GENERATION_PROMPT.format(
        query=query,
        rag_results=rag_formatted,
        web_results=web_formatted,
    ) + f"\n\n## TRL Evaluator 산출 결과 (3.3절에 반드시 반영)\n{trl_formatted}"

    today = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    system_prompt = REPORT_SYSTEM_PROMPT.format(query=query, date=today)

    logger.info("Report Generator: 보고서 초안 생성 시작")
    missing     = []
    saved_path  = None

    try:
        response       = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=generation_prompt),
        ])
        report_content = response.content.strip()

        # 섹션 완결성 검증
        required = ["SUMMARY", "1. 분석 배경", "2. 분석 대상", "3. 경쟁사 동향", "4. 전략적 시사점", "REFERENCE"]
        missing  = [s for s in required if s not in report_content]
        for s in missing:
            report_content += f"\n\n## {s}\n[공개 정보 부재 또는 수집 정보 부족으로 작성 불가]\n"
        if missing:
            logger.warning(f"보고서 섹션 누락 보완: {missing}")

        saved_path = _save_report(report_content, query)
        logger.info(f"보고서 완성 ({len(report_content)}자)")

    except Exception as e:
        logger.error(f"보고서 생성 오류: {e}")
        report_content = (
            f"# 보고서 생성 오류\n\n오류: {e}\n\n"
            f"수집된 정보:\n{rag_formatted[:500]}\n\n{web_formatted[:500]}"
        )

    missing_info = f"누락 섹션 {missing} → placeholder 삽입" if missing else "완료"
    ai_message   = AIMessage(
        content=(
            f"[ReportGen 완료] 보고서 초안 작성 완료 ({len(report_content)}자)\n"
            f"저장 경로: {saved_path}\n"
            f"섹션 체크: {missing_info}"
        )
    )
    return {"messages": [ai_message], "report": report_content}