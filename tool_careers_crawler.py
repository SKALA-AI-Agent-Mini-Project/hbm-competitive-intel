"""
Samsung Careers Crawler — 채용공고 수집
Samsung 반도체 R&D 채용공고에서 HBM·PIM·CXL 관련 직무 추출
TRL 간접지표: 특정 키워드 직무 급증 = 해당 기술 개발 진입 신호

의존성: pip install playwright && playwright install chromium
"""

import logging
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

# 수집 대상 URL
SAMSUNG_CAREERS_URLS = [
    # Samsung Semiconductor 글로벌 채용
    "https://www.samsungsemiconductor.com/us/careers/search/?query={keyword}",
    # Samsung Electronics 글로벌 채용
    "https://careers.samsung.com/search/#q={keyword}&t=Jobs",
]

# TRL 신호 키워드 — 직무명에서 감지
TRL_SIGNAL_KEYWORDS = {
    "hybrid bonding": "TRL 4~6 추정 (간접지표) — Hybrid Bonding 엔지니어 채용 = HBM4 공정 내재화 진입 신호",
    "hbm4": "TRL 4~6 추정 (간접지표) — HBM4 전담 직무 채용 = 본격 개발 단계 진입",
    "3d integration": "TRL 4~6 추정 (간접지표) — 3D Integration 채용 급증 = 차세대 패키징 내재화",
    "cxl": "TRL 5~7 추정 (간접지표) — CXL 관련 채용 = 상용화 준비 단계",
    "pim": "TRL 4~6 추정 (간접지표) — PIM 전담 채용 = 제품화 단계 진입 추정",
    "processing-in-memory": "TRL 4~6 추정 (간접지표) — PIM 엔지니어링 채용",
    "hbm": "TRL 7~9 (양산 단계) — HBM 양산 직무 채용 지속",
}

SEARCH_KEYWORDS = [
    "HBM hybrid bonding",
    "HBM4",
    "PIM memory",
    "CXL memory",
    "3D integration memory",
]


def _infer_trl_from_job(title: str, description: str) -> str:
    """직무명·설명에서 TRL 신호 추출"""
    combined = (title + " " + description).lower()
    for keyword, trl_signal in TRL_SIGNAL_KEYWORDS.items():
        if keyword in combined:
            return trl_signal
    return "TRL 추정 근거 없음 — 일반 반도체 직무"


def _crawl_with_playwright(keyword: str) -> List[Dict]:
    """Playwright로 Samsung Careers 크롤링"""
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        logger.error("Playwright 미설치. 'pip install playwright && playwright install chromium' 실행 필요")
        return []

    jobs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        # Samsung Semiconductor 채용 검색
        url = f"https://www.samsungsemiconductor.com/us/careers/search/?query={keyword.replace(' ', '+')}"
        try:
            page.goto(url, timeout=15000)
            page.wait_for_load_state("networkidle", timeout=10000)
            time.sleep(2)

            # 직무 카드 추출 (구조 변경에 대비해 여러 셀렉터 시도)
            job_cards = page.query_selector_all("[class*='job'], [class*='career'], [class*='position']")

            for card in job_cards[:20]:
                try:
                    title_el = card.query_selector("h2, h3, h4, [class*='title'], [class*='name']")
                    title = title_el.inner_text().strip() if title_el else ""

                    desc_el = card.query_selector("p, [class*='desc'], [class*='summary']")
                    description = desc_el.inner_text().strip() if desc_el else ""

                    date_el = card.query_selector("[class*='date'], time")
                    date = date_el.inner_text().strip() if date_el else "N/A"

                    link_el = card.query_selector("a")
                    href = link_el.get_attribute("href") if link_el else ""
                    job_url = f"https://www.samsungsemiconductor.com{href}" if href and href.startswith("/") else href or url

                    if title:
                        jobs.append({
                            "title": title,
                            "description": description[:300],
                            "date": date,
                            "url": job_url,
                            "keyword": keyword,
                        })
                except Exception:
                    continue

        except PlaywrightTimeout:
            logger.warning(f"Samsung Semiconductor 페이지 타임아웃: {keyword}")
        except Exception as e:
            logger.warning(f"Samsung Semiconductor 크롤링 실패: {e}")

        browser.close()

    return jobs


def fetch_samsung_jobs() -> List[Dict]:
    """
    Samsung 채용공고 수집 및 TRL 신호 분석.

    Returns:
        [{"title", "description", "date", "url", "trl_clue"}, ...]
    """
    seen_titles: set = set()
    all_jobs = []

    for keyword in SEARCH_KEYWORDS:
        logger.info(f"Samsung Careers 크롤링: '{keyword}'")
        jobs = _crawl_with_playwright(keyword)
        time.sleep(1)

        for job in jobs:
            title = job.get("title", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            trl_clue = _infer_trl_from_job(title, job.get("description", ""))

            all_jobs.append({
                "title": title,
                "description": job["description"],
                "date": job["date"],
                "url": job["url"],
                "trl_clue": trl_clue,
            })

    logger.info(f"Samsung Careers: 총 {len(all_jobs)}건 직무 수집")
    return all_jobs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    jobs = fetch_samsung_jobs()
    for j in jobs:
        print(f"[{j['date']}] {j['title']}")
        print(f"  TRL: {j['trl_clue']}")
        print(f"  URL: {j['url']}")
        print()
