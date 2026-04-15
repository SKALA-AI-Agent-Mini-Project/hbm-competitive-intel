"""
RAG Agent — HBM Competitive R&D Intelligence System
통합: 팀 개선(RAGResult TypedDict, _summarize_chunk, prompts 연동)
     + HBMRetriever 사용(reranker·query expansion — rag/retriever.py)

이전 init_rag_retriever() 인터페이스 유지 (main.ipynb 호환)
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from graph.state import AgentState, RAGResult
from prompts.rag_prompt import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

TOP_K = 5
_hbm_retriever = None   # main.ipynb에서 init_rag_retriever()로 주입 가능


def init_rag_retriever(retriever, all_chunks=None) -> None:
    """외부(main.ipynb/app.py)에서 리트리버를 주입할 때 사용."""
    global _hbm_retriever
    _hbm_retriever = retriever
    logger.info("RAG Agent: 외부 리트리버 등록 완료")


def _get_retriever():
    """리트리버 반환. 주입된 게 없으면 HBMRetriever 자동 초기화."""
    global _hbm_retriever
    if _hbm_retriever is None:
        from rag.retriever import HBMRetriever
        _hbm_retriever = HBMRetriever(use_reranker=True)
        logger.info("RAG Agent: HBMRetriever 자동 초기화")
    return _hbm_retriever


def _extract_instruction(state: AgentState) -> str:
    """마지막 Supervisor 메시지에서 지시 내용 추출."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and "지시:" in msg.content:
            return msg.content.split("지시:")[-1].strip()
    return state.get("query", "")


def _summarize_chunk(chunk: str, llm: ChatOpenAI) -> str:
    """청크를 2~3문장으로 요약."""
    try:
        resp = llm.invoke(
            f"다음 반도체 기술 문서 청크를 2~3문장으로 요약하세요. "
            f"전문 용어를 그대로 사용하세요:\n\n{chunk[:800]}"
        )
        return resp.content.strip()
    except Exception:
        return chunk[:200] + "..."


def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    RAG Agent 노드.
    HBMRetriever(BM25+Dense RRF + reranker)로 검색 후 rag_results 반환.
    """
    llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    instruction = _extract_instruction(state)
    query       = state.get("query", "")
    search_q    = f"{query} {instruction}".strip() if instruction != query else query

    logger.info(f"RAG Agent 검색: {search_q[:80]}")

    try:
        retriever = _get_retriever()
        raw_docs  = retriever.retrieve(search_q, k=TOP_K)  # List[Tuple[Document, float]]
    except FileNotFoundError:
        logger.warning("벡터스토어 없음 → 빈 결과 반환")
        raw_docs = []
    except Exception as e:
        logger.error(f"RAG 검색 오류: {e}")
        raw_docs = []

    if not raw_docs:
        summary_msg = "[해당 문서 없음] 벡터스토어에서 관련 문서를 찾지 못했습니다."
        rag_results: List[RAGResult] = []
    else:
        rag_results = []
        for rank, (doc, raw_score) in enumerate(raw_docs):
            chunk_text = doc.page_content
            source     = doc.metadata.get("source", "unknown")
            page       = doc.metadata.get("page", "")
            summary    = _summarize_chunk(chunk_text, llm)

            # 순위 기반 점수 (reranker/FAISS/RRF 점수 타입과 무관하게 0~1 보장)
            rank_score = max(0.1, round(1.0 - rank * 0.15, 2))  # 1.0, 0.85, 0.70, 0.55, 0.40

            rag_results.append(RAGResult(
                chunk=chunk_text[:500],
                summary=summary,
                source=f"{source} (p.{page})" if page else source,
                score=rank_score,
            ))

        summary_msg = f"{len(rag_results)}건의 관련 문서 청크를 검색했습니다."

    logger.info(f"RAG Agent: {summary_msg}")

    ai_message = AIMessage(
        content=f"[RAGAgent 완료] {summary_msg}\n"
        + "\n".join(f"- {r['source']}: {r['summary'][:80]}..." for r in rag_results[:3])
    )
    return {"messages": [ai_message], "rag_results": rag_results}