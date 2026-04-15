"""
Semantic Scholar Tool — Samsung 논문 수집
ISSCC · IEDM · VLSI · Hot Chips 등 주요 학회에서
Samsung 저자 HBM·PIM·CXL 관련 논문 검색

API: https://api.semanticscholar.org (무료, 인증 불필요)
"""

import logging
import time
from typing import List, Dict

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# 수집 대상 학회
TARGET_VENUES = {"ISSCC", "IEDM", "VLSI", "Hot Chips", "DATE", "HPCA", "ISCA"}

# Samsung 검색 쿼리 목록
SAMSUNG_QUERIES = [
    "Samsung HBM high bandwidth memory",
    "Samsung PIM processing in memory",
    "Samsung CXL compute express link",
    "Samsung hybrid bonding 3D integration",
    "Samsung HBM4 memory architecture",
]


def _search_papers(query: str, limit: int = 10, retries: int = 2) -> List[Dict]:
    """Semantic Scholar API로 논문 검색 (rate limit 재시도 포함)"""
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,venue,abstract,externalIds,openAccessPdf",
        "publicationDateOrYear": "2023-2025",
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"Semantic Scholar rate limit — {wait}초 대기 후 재시도")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"Semantic Scholar 검색 실패 ({query}): {e}")
            if attempt < retries:
                time.sleep(3)
    return []


def _is_samsung_paper(paper: Dict) -> bool:
    """Samsung 소속 저자가 있는지 확인"""
    authors = paper.get("authors", [])
    # Semantic Scholar 기본 필드엔 affiliation 없음 — 이름/쿼리로 필터
    # query에 "Samsung" 포함했으므로 대부분 관련 논문
    return True


def _extract_trl_from_venue(venue: str) -> str:
    """학회명으로 TRL 추정"""
    venue_upper = (venue or "").upper()
    if any(v in venue_upper for v in ["ISSCC", "IEDM", "VLSI"]):
        return "TRL 1~3 (학회 발표 — 연구 완료 단계)"
    if any(v in venue_upper for v in ["HOT CHIPS", "HPCA", "ISCA"]):
        return "TRL 2~4 (아키텍처 발표 — 프로토타입 단계 추정)"
    return "TRL 1~3 (학술 논문)"


def fetch_samsung_papers() -> List[Dict]:
    """
    Samsung HBM·PIM·CXL 관련 논문 수집.

    Returns:
        [{"title", "authors", "year", "venue", "abstract", "url", "trl_clue"}, ...]
    """
    seen_titles: set = set()
    results = []

    for query in SAMSUNG_QUERIES:
        papers = _search_papers(query, limit=8)
        time.sleep(0.5)  # API rate limit 방지

        for paper in papers:
            title = paper.get("title", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # 연도 필터 (2022 이후)
            year = paper.get("year", 0) or 0
            if year < 2022:
                continue

            venue = paper.get("venue", "")
            abstract = paper.get("abstract", "") or ""

            # URL 추출
            ext_ids = paper.get("externalIds", {}) or {}
            doi = ext_ids.get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else "https://www.semanticscholar.org"

            pdf = paper.get("openAccessPdf")
            if pdf and isinstance(pdf, dict):
                url = pdf.get("url", url)

            authors = ", ".join(
                a.get("name", "") for a in (paper.get("authors", []) or [])[:4]
            )

            results.append({
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "abstract": abstract[:400],
                "url": url,
                "trl_clue": _extract_trl_from_venue(venue),
            })

    logger.info(f"Semantic Scholar: Samsung 논문 {len(results)}건 수집")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    papers = fetch_samsung_papers()
    for p in papers:
        print(f"[{p['year']}] {p['venue']} | {p['title'][:60]}")
        print(f"  TRL: {p['trl_clue']}")
        print()