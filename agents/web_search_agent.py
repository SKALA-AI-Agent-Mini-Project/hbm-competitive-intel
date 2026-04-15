"""
Web Search Agent 노드
설계서 2-3: 경쟁사 최신 공개 정보 수집 — TRL 간접지표 획득
"""
"""
Web Search Agent — HBM Competitive R&D Intelligence System
"""

import json
import logging
from typing import Any, Dict, List

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from graph.state import AgentState, WebResult
from prompts.web_search_prompt import WEB_SEARCH_SYSTEM_PROMPT
from tool_semantic_scholar import fetch_samsung_papers
from tool_careers_crawler import fetch_samsung_jobs
from tool_kipris import fetch_samsung_patents

logger = logging.getLogger(__name__)

_FALLBACK_QUERIES = {
    "samsung_hbm4": "Samsung HBM4 Hybrid Bonding technology 2024 2025",
    "samsung_pim": "Samsung PIM Processing-In-Memory patent HBM 2024",
    "samsung_cxl": "Samsung CXL memory module standard 2024 2025",
    "micron_hbm4": "Micron HBM4 leapfrog strategy 3D integration 2025",
    "micron_cxl": "Micron CXL consortium standard 2024 2025",
    "hbm4_hybrid_bonding": "HBM4 hybrid bonding AMAT BESI semiconductor 2025",
    "isscc_iedm_hbm": "ISSCC IEDM HBM PIM CXL paper 2024 2025",
}


def _generate_search_queries(instruction: str, query: str, llm: ChatOpenAI) -> List[str]:
    prompt = f"""아래 분석 목표에 맞는 웹 검색 쿼리를 최대 4개 생성하세요.
경쟁사(Samsung·Micron)의 특허·논문·IR·채용공고를 수집하기 위한 영어 검색 쿼리입니다.

## 분석 목표
{query}

## Supervisor 지시
{instruction}

## 출력 형식 (JSON 배열만, 설명 없이)
["쿼리1", "쿼리2", "쿼리3", "쿼리4"]
"""
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            logger.info(f"LLM 생성 검색 쿼리: {queries}")
            return queries[:4]
    except Exception as e:
        logger.warning(f"LLM 쿼리 생성 실패, 폴백 사용: {e}")

    combined = (instruction + " " + query).lower()
    fallback = []
    if "samsung" in combined:
        fallback.append(_FALLBACK_QUERIES["samsung_hbm4"])
        if "pim" in combined:
            fallback.append(_FALLBACK_QUERIES["samsung_pim"])
        if "cxl" in combined:
            fallback.append(_FALLBACK_QUERIES["samsung_cxl"])
    if "micron" in combined:
        fallback.append(_FALLBACK_QUERIES["micron_hbm4"])
        if "cxl" in combined:
            fallback.append(_FALLBACK_QUERIES["micron_cxl"])
    if not fallback:
        fallback = [
            _FALLBACK_QUERIES["isscc_iedm_hbm"],
            _FALLBACK_QUERIES["samsung_hbm4"],
            _FALLBACK_QUERIES["micron_hbm4"],
        ]
    return fallback[:4]


def _collect_raw_results(search_queries: List[str]) -> List[Dict]:
    tavily = TavilySearch(max_results=3)
    arxiv = ArxivQueryRun()
    raw_results = []

    for sq in search_queries[:3]:
        try:
            results = tavily.invoke(sq)
            for r in results.get("results", []):
                raw_results.append({
                    "source": "tavily",
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "date": r.get("published_date", r.get("date", "N/A")),
                    "content": r.get("content", "")[:600],
                })
        except Exception as e:
            logger.warning(f"Tavily 검색 실패 ({sq}): {e}")

    arxiv_queries = [
        q for q in search_queries
        if any(kw in q.lower() for kw in ["pim", "cxl", "hbm", "hybrid bonding", "isscc"])
    ]
    for aq in arxiv_queries[:2]:
        try:
            arxiv_result = arxiv.invoke(aq)
            if arxiv_result and len(arxiv_result) > 50:
                raw_results.append({
                    "source": "arxiv",
                    "title": f"ArXiv: {aq}",
                    "url": "https://arxiv.org",
                    "date": "2024-2025",
                    "content": arxiv_result[:600],
                })
        except Exception as e:
            logger.warning(f"ArXiv 검색 실패 ({aq}): {e}")

    seen_urls = set()
    deduped = []
    for r in raw_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)

    return deduped


def _analyze_with_llm(raw_results: List[Dict], query: str, llm: ChatOpenAI) -> List[WebResult]:
    if not raw_results:
        return []

    formatted = ""
    for i, r in enumerate(raw_results, 1):
        formatted += (
            f"[{i}] 제목: {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    날짜: {r['date']}\n"
            f"    내용: {r['content']}\n\n"
        )

    prompt = f"""아래는 "{query}" 관련 웹 검색 결과입니다.
각 항목을 분석하여 TRL 추정 근거를 포함한 JSON 배열로 반환하세요.

## 수집된 정보
{formatted}

## 출력 형식 (JSON 배열만, 설명 없이)
[
  {{
    "title": "제목",
    "url": "URL",
    "date": "날짜",
    "summary": "내용 요약 2~3문장 (반도체 전문 용어 유지)",
    "trl_clue": "TRL 추정 근거"
  }}
]
"""

    try:
        response = llm.invoke([
            SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        analyzed = json.loads(raw)

        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "date": item.get("date", "N/A"),
                "summary": item.get("summary", ""),
                "trl_clue": item.get("trl_clue", "[TRL 추정 근거 없음]"),
            }
            for item in analyzed
            if isinstance(item, dict)
        ]
    except Exception as e:
        logger.warning(f"LLM 분석 실패, 원시 결과 직접 변환: {e}")
        return [
            {
                "title": r["title"],
                "url": r["url"],
                "date": r["date"],
                "summary": r["content"],
                "trl_clue": "[LLM 분석 실패 — TRL 추정 불가]",
            }
            for r in raw_results
        ]


def _papers_to_raw(papers: List[Dict]) -> List[Dict]:
    return [
        {
            "source": "semantic_scholar",
            "title": p["title"],
            "url": p["url"],
            "date": str(p.get("year", "N/A")),
            "content": f"[{p.get('venue', '')}] {p.get('abstract', '')}",
        }
        for p in papers
    ]


def _jobs_to_web_results(jobs: List[Dict]) -> List[WebResult]:
    return [
        {
            "title": f"[채용공고] {j['title']}",
            "url": j["url"],
            "date": j["date"],
            "summary": j["description"],
            "trl_clue": j["trl_clue"],
        }
        for j in jobs
    ]


def _patents_to_web_results(patents: List[Dict]) -> List[WebResult]:
    return [
        {
            "title": f"[특허] {p['title']}",
            "url": p["url"],
            "date": p.get("application_date", "N/A"),
            "summary": f"출원번호: {p.get('application_number', '')} | IPC: {p.get('ipc_code', '')}",
            "trl_clue": p["trl_clue"],
        }
        for p in patents
    ]


def web_search_node(state: AgentState) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query = state.get("query", "")

    messages = state.get("messages", [])
    instruction = query
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and "지시:" in msg.content:
            instruction = msg.content.split("지시:")[-1].strip()
            break

    search_queries = _generate_search_queries(instruction, query, llm)

    raw_results = _collect_raw_results(search_queries)
    logger.info(f"Tavily/ArXiv: {len(raw_results)}건")

    papers = []
    patents = []
    jobs = []

    try:
        papers = fetch_samsung_papers()
        raw_results += _papers_to_raw(papers)
        logger.info(f"Semantic Scholar: {len(papers)}건")
    except Exception as e:
        logger.warning(f"Semantic Scholar 수집 실패: {e}")

    web_results: List[WebResult] = _analyze_with_llm(raw_results, query, llm)

    try:
        patents = fetch_samsung_patents()
        web_results += _patents_to_web_results(patents)
        logger.info(f"KIPRIS: {len(patents)}건")
    except Exception as e:
        logger.warning(f"KIPRIS 수집 실패: {e}")

    try:
        jobs = fetch_samsung_jobs()
        web_results += _jobs_to_web_results(jobs)
        logger.info(f"Samsung Careers: {len(jobs)}건")
    except Exception as e:
        logger.warning(f"Samsung Careers 수집 실패: {e}")

    if not web_results:
        summary_msg = "[공개 정보 부재] 관련 공개 정보를 찾지 못했습니다. TRL 추정 불가."
        logger.warning("Web Search Agent: 최종 결과 없음")
    else:
        summary_msg = (
            f"총 {len(web_results)}건 수집 "
            f"(뉴스/논문: {len(raw_results)}건, 특허: {len(patents)}건, 채용공고: {len(jobs)}건)"
        )
        logger.info(f"Web Search Agent: {summary_msg}")

    ai_message = AIMessage(
        content=f"[WebAgent 완료] {summary_msg}\n"
        + "\n".join(
            f"- [{r['date']}] {r['title'][:80]} | {r['trl_clue'][:60]}"
            for r in web_results[:5]
        )
    )

    return {
        "messages": [ai_message],
        "web_results": web_results,
    }
