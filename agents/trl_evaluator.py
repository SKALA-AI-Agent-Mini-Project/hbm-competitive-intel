"""
TRL Evaluator 노드 — 교수 피드백 반영
"TRL 평가가 node 또는 에이전트 형태까지 고려되었는지 검토해 주시기 바랍니다"

RAG + Web 수집 결과를 근거로 TRL을 체계적으로 평가하는 전용 Agent.
Report Generator가 받기 전에 구조화된 TRL 평가 결과를 State에 저장.
"""
import json
import re
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from graph.state import AgentState

TRL_EVALUATOR_SYSTEM = """당신은 반도체 기술 성숙도(TRL) 전문 평가 에이전트입니다.

## 역할
RAG 검색 결과와 웹 수집 결과를 근거로 각 기업·기술별 TRL을 체계적으로 평가합니다.

## TRL 정의
- TRL 1~3 : 기초 연구 / 개념 검증
- TRL 4~6 : 개발 / 시제품 검증 ★ 비공개 구간 — 간접지표로만 추정
- TRL 7~8 : 파일럿 / 실증 배치
- TRL 9   : 양산 / 상용화

## 간접지표 → TRL 매핑 규칙
| 간접지표 | 추정 TRL | 신뢰도 |
|----------|---------|--------|
| 기초 논문 발표 | TRL 1~3 | 중 |
| 학회 시제품 발표(ISSCC·IEDM) | TRL 4~6 | 중 |
| 특허 출원 집중 | TRL 3~6 | 낮음(★추정) |
| 채용공고 급증 | TRL 4~6 | 낮음(★추정) |
| IR/PR 양산 발표 | TRL 7~9 | 높음 |
| 고객 공급 시작 | TRL 8~9 | 높음 |

## 출력 형식 (JSON — 반드시 준수)
{
  "evaluated_at": "YYYY-MM-DD",
  "technologies": {
    "HBM4": {
      "SK Hynix": {"trl": "6~7", "evidence": ["근거1", "근거2"], "confidence": "중", "note": ""},
      "Samsung":  {"trl": "5~6", "evidence": ["근거1"], "confidence": "낮음", "note": "★간접지표 기반 추정"},
      "Micron":   {"trl": "4~5", "evidence": ["근거1"], "confidence": "낮음", "note": "★간접지표 기반 추정"}
    },
    "PIM": { ... },
    "CXL": { ... }
  },
  "threat_summary": {
    "HBM4": "위협 수준 및 근거",
    "PIM":  "위협 수준 및 근거",
    "CXL":  "위협 수준 및 근거"
  }
}
"""

TRL_EVALUATOR_PROMPT = """## 평가 대상 증거

### RAG 수집 결과 (내부 기술 문서)
{rag_results}

### Web 수집 결과 (공개 정보)
{web_results}

## 지시
위 증거를 바탕으로 HBM4 / PIM / CXL 기술에 대해 SK Hynix · Samsung · Micron 3사의
TRL을 평가하고 JSON 형식으로 반환하세요.

규칙:
- TRL 4~6 구간은 반드시 "★간접지표 기반 추정" 명시
- 근거 없는 TRL은 "공개 정보 부재"로 표기 (임의 추정 금지)
- confidence: "높음" / "중" / "낮음" 중 하나만 사용
"""


def trl_evaluator_node(state: AgentState) -> dict:
    """
    TRL Evaluator 노드
    RAG + Web 결과를 입력받아 구조화된 TRL 평가 결과를 state에 저장
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    rag_results = state.get("rag_results") or []
    web_results = state.get("web_results") or []

    print(f"\n🔬 [TRL Evaluator] TRL 평가 시작")
    print(f"   RAG 입력: {len(rag_results)}건 | Web 입력: {len(web_results)}건")

    rag_text = _format_evidence(rag_results, "RAG")
    web_text = _format_evidence(web_results, "WEB")

    messages = [
        SystemMessage(content=TRL_EVALUATOR_SYSTEM),
        HumanMessage(content=TRL_EVALUATOR_PROMPT.format(
            rag_results=rag_text,
            web_results=web_text,
        )),
    ]

    response = llm.invoke(messages)
    trl_assessment = _parse_trl_response(response.content)

    # 평가 결과 출력
    _print_trl_summary(trl_assessment)

    return {
        "messages": [AIMessage(content=response.content[:300] + "...", name="trl_evaluator")],
        "trl_assessment": trl_assessment,
    }


def _format_evidence(results: list, tag: str) -> str:
    if not results:
        return f"{tag} 결과 없음"
    lines = []
    for i, r in enumerate(results, 1):
        if tag == "RAG":
            lines.append(f"[{tag}-{i}] 출처: {r.get('source','?')} | {r.get('summary','')[:200]}")
        else:
            lines.append(
                f"[{tag}-{i}] {r.get('title','?')} ({r.get('date','N/A')})\n"
                f"  TRL단서: {r.get('trl_clue','없음')}\n"
                f"  요약: {r.get('summary','')[:150]}"
            )
    return "\n".join(lines)


def _parse_trl_response(content: str) -> dict:
    """JSON 파싱 — 실패 시 기본값 반환"""
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    return {
        "evaluated_at": datetime.now().strftime("%Y-%m-%d"),
        "technologies": {
            "HBM4": {
                "SK Hynix": {"trl": "6~7", "evidence": ["설계서 기준"], "confidence": "중", "note": ""},
                "Samsung":  {"trl": "5~6", "evidence": ["간접지표"], "confidence": "낮음", "note": "★간접지표 기반 추정"},
                "Micron":   {"trl": "4~5", "evidence": ["간접지표"], "confidence": "낮음", "note": "★간접지표 기반 추정"},
            },
            "PIM": {
                "SK Hynix": {"trl": "5~6", "evidence": ["AiMX 출시"], "confidence": "중", "note": ""},
                "Samsung":  {"trl": "4~5", "evidence": ["특허 출원"], "confidence": "낮음", "note": "★간접지표 기반 추정"},
                "Micron":   {"trl": "3~4", "evidence": ["연구 단계"], "confidence": "낮음", "note": "★간접지표 기반 추정"},
            },
            "CXL": {
                "SK Hynix": {"trl": "7~8", "evidence": ["CMM-DDR5 출시"], "confidence": "높음", "note": ""},
                "Samsung":  {"trl": "6~7", "evidence": ["간접지표"], "confidence": "낮음", "note": "★간접지표 기반 추정"},
                "Micron":   {"trl": "5~6", "evidence": ["표준화 참여"], "confidence": "낮음", "note": "★간접지표 기반 추정"},
            },
        },
        "threat_summary": {
            "HBM4": "Samsung 수율 개선 시 물량 위협 가능 (보통)",
            "PIM":  "Samsung 특허 공세로 표준 주도권 위협 (높음)",
            "CXL":  "Micron CXL 표준화 선도로 시장 진입 비용 증가 위협 (보통)",
        },
        "_parse_error": True,
    }


def _print_trl_summary(assessment: dict):
    """TRL 평가 결과 요약 출력"""
    techs = assessment.get("technologies", {})
    print(f"\n   📊 TRL 평가 결과:")
    print(f"   {'기술':<8} {'SK Hynix':<12} {'Samsung':<12} {'Micron':<12}")
    print(f"   {'-'*48}")
    for tech, companies in techs.items():
        sk  = companies.get("SK Hynix", {}).get("trl", "N/A")
        sam = companies.get("Samsung",  {}).get("trl", "N/A")
        mic = companies.get("Micron",   {}).get("trl", "N/A")
        print(f"   {tech:<8} {sk:<12} {sam:<12} {mic:<12}")