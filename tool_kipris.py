"""
KIPRIS Tool — Samsung 특허 수집 (한국특허청)
Samsung Electronics의 HBM·PIM·CXL 관련 특허 출원 패턴 분석

API 키 발급: https://www.kipris.or.kr → API 서비스 신청
.env에 추가: KIPRIS_API_KEY=발급받은키
"""

import logging
import os
import time
from typing import List, Dict

import requests

logger = logging.getLogger(__name__)

# KIPRIS REST API 엔드포인트
BASE_URL = "http://plus.kipris.or.kr/openapi/rest"

# Samsung 관련 IPC 코드
# H01L: 반도체 소자 (HBM TSV·패키징)
# G11C: 정적 저장장치 (메모리 아키텍처)
TARGET_IPC_CODES = ["H01L", "G11C"]

SAMSUNG_APPLICANT = "삼성전자"

SEARCH_KEYWORDS = [
    "HBM 하이브리드 본딩",
    "PIM 프로세싱 인 메모리",
    "CXL 컴퓨트 익스프레스 링크",
    "HBM4 적층 메모리",
    "3D 집적 메모리",
]


def _search_patents(keyword: str, api_key: str, start_date: str = "20230101") -> List[Dict]:
    """
    KIPRIS 특허 검색 API 호출.
    API 문서: https://www.kipris.or.kr/khome/rest.do
    """
    endpoint = f"{BASE_URL}/patUtiModInfoSearchSevice/patentUtilitySearch"
    params = {
        "word": f"{SAMSUNG_APPLICANT} {keyword}",
        "inventionTitle": keyword,
        "applicantName": SAMSUNG_APPLICANT,
        "ipcNumber": "H01L",
        "applicationDate": start_date,
        "numOfRows": 10,
        "pageNo": 1,
        "ServiceKey": api_key,
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        # KIPRIS는 XML 반환 — 파싱 필요
        return _parse_kipris_xml(resp.text, keyword)
    except Exception as e:
        logger.warning(f"KIPRIS 검색 실패 ({keyword}): {e}")
        return []


def _parse_kipris_xml(xml_text: str, keyword: str) -> List[Dict]:
    """KIPRIS XML 응답 파싱"""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        patents = []

        for item in root.findall(".//item"):
            title = item.findtext("inventionTitle", "")
            app_no = item.findtext("applicationNumber", "")
            app_date = item.findtext("applicationDate", "")
            ipc = item.findtext("ipcNumber", "")
            applicant = item.findtext("applicantName", "")

            if title:
                patents.append({
                    "title": title,
                    "application_number": app_no,
                    "application_date": app_date,
                    "ipc_code": ipc,
                    "applicant": applicant,
                    "keyword": keyword,
                    "url": f"https://www.kipris.or.kr/khome/main.do#app={app_no}",
                })

        return patents

    except Exception as e:
        logger.warning(f"KIPRIS XML 파싱 실패: {e}")
        return []


def _infer_trl_from_patent(title: str, ipc: str) -> str:
    """특허 내용으로 TRL 추정"""
    title_lower = title.lower()
    if any(kw in title_lower for kw in ["양산", "제조 방법", "공정"]):
        return "TRL 5~7 추정 (간접지표) — 제조 공정 특허 = 양산 준비 단계"
    if any(kw in title_lower for kw in ["구조", "소자", "회로"]):
        return "TRL 2~4 (특허 출원) — 기초 구조 특허 = 연구개발 단계"
    return "TRL 2~4 (특허 출원) — 출원 단계"


def fetch_samsung_patents() -> List[Dict]:
    """
    Samsung HBM·PIM·CXL 관련 특허 수집.

    Returns:
        [{"title", "application_number", "application_date", "ipc_code", "url", "trl_clue"}, ...]
    """
    api_key = os.getenv("KIPRIS_API_KEY")
    if not api_key:
        logger.warning("KIPRIS_API_KEY 없음 — .env에 추가 필요. 특허 수집 건너뜀.")
        return []

    seen: set = set()
    results = []

    for keyword in SEARCH_KEYWORDS:
        logger.info(f"KIPRIS 검색: {keyword}")
        patents = _search_patents(keyword, api_key)
        time.sleep(0.5)

        for p in patents:
            app_no = p.get("application_number", "")
            if not app_no or app_no in seen:
                continue
            seen.add(app_no)

            trl_clue = _infer_trl_from_patent(p["title"], p.get("ipc_code", ""))
            p["trl_clue"] = trl_clue
            results.append(p)

    logger.info(f"KIPRIS: Samsung 특허 {len(results)}건 수집")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    patents = fetch_samsung_patents()
    for p in patents:
        print(f"[{p['application_date']}] {p['title'][:60]}")
        print(f"  IPC: {p['ipc_code']} | TRL: {p['trl_clue']}")
        print()
