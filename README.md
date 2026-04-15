# SK Hynix HBM Competitive Intelligence System
(수정용)
Samsung·Micron의 HBM 관련 R&D 공개 정보를 자동 수집·분석하여, SK Hynix R&D 팀이 즉시 활용 가능한 TRL 기반 경쟁사 기술 전략 보고서를 생성하는 Agentic Workflow 시스템.

## Overview

- **Objective** : Samsung·Micron의 HBM4 / PIM / CXL R&D 공개 정보를 자동 수집·분석하여 TRL 기반 경쟁사 기술 전략 보고서를 생성
- **Method** : Supervisor Pattern 기반 멀티에이전트 워크플로우 — RAG(내부 문서) + Web Search(최신 공개정보) 수집 후 보고서 자동 작성
- **Tools** : LangGraph, LangChain, FAISS, BM25, Tavily Search, GPT-4o

## Features

- 내부 기술 문서 기반 정보 추출 (PDF·TXT → FAISS 벡터스토어)
- 경쟁사 최신 공개정보 수집 (특허·논문·IR·채용공고 — Tavily Search)
- TRL 기반 경쟁사 비교 매트릭스 자동 생성 (Samsung·Micron × HBM4/PIM/CXL)
- 확증 편향 방지 전략 : RAG(내부 선행 분석) + Web(최신 공개정보)를 독립 수집 후 Supervisor가 통합 판단. TRL 4~6 비공개 구간은 '간접지표 기반 추정' 명시로 오정보 기술 방지
- 최종 산출물 자동 PDF 변환 (`outputs/` 디렉토리에 저장)
- Tavily API 키 없이도 LLM 폴백 모드로 실행 가능

## Tech Stack

| Category | Details |
|----------|---------|
| Framework | LangGraph ≥ 0.2.50, LangChain ≥ 0.3.0, Python 3.11+ |
| LLM | GPT-4o (Supervisor·Report), GPT-4o-mini (RAG·Web) via OpenAI API |
| Retrieval | FAISS + BM25 Hybrid Search (RRF) — Hit Rate@5, MRR |
| Embedding | BAAI/bge-m3 (한·영 혼용, 8192 tokens) |
| Web Search | Tavily Search API |
| PDF Export | weasyprint + markdown2 |

## Agents

- **Supervisor** : 쿼리 분류 → 에이전트 위임 → 수집 충분성 판단 → 보고서 품질 검증 → END 결정. GPT-4o 사용
- **RAG Agent** : 내부 벡터스토어(FAISS)에서 HBM 기술 개념 문서 검색. Hybrid Search(BM25+Dense RRF) 적용
- **Web Search Agent** : Tavily로 Samsung·Micron 최신 특허·논문·IR·채용공고 수집. TRL 간접지표 추출
- **Report Generator** : RAG + Web 수집 결과 통합 → TRL 비교 매트릭스 포함 Markdown 보고서 초안 작성. GPT-4o 사용

## Architecture

```
START
  │
  ▼
Supervisor  ──► RAG Agent      ──┐
  │         ──► Web Agent      ──┤ → Supervisor
  │         ──► Report Generator─┘
  │
  ▼
 END (PDF 보고서 반환)
```

> 핵심 원칙: 모든 Sub-Agent는 Supervisor로 복귀. Report Generator 초안 반환 후 Supervisor가 검토하여 END 결정.

## Directory Structure

```
hbm-competitive-intel/
├── data/                   # 내부 분석 문서 (PDF·TXT)
│   └── vectorstore/        # FAISS 인덱스 (자동 생성)
├── agents/                 # Agent 노드 모듈
│   ├── supervisor.py       # 라우팅·품질 판단·종료 결정
│   ├── rag_agent.py        # 내부 문서 벡터 검색
│   ├── web_agent.py        # 경쟁사 공개정보 수집
│   └── report_generator.py # Markdown 보고서 초안 생성
├── graph/
│   ├── state.py            # AgentState TypedDict
│   └── workflow.py         # LangGraph StateGraph 구성
├── prompts/                # 프롬프트 템플릿
├── rag/                    # RAG 파이프라인 (임베딩·벡터스토어·리트리버)
├── utils/
│   ├── pdf_exporter.py     # Markdown → PDF 변환
│   └── pretty_print.py     # 출력 유틸리티
├── outputs/                # 생성된 보고서 저장 (.pdf / .md)
├── main.ipynb              # 실습 진입점
├── app.py                  # CLI 실행 스크립트
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
# 1. 환경 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 2. 패키지 설치
pip install -r requirements.txt

# 3-A. Jupyter 노트북 실행 (권장)
jupyter notebook main.ipynb

# 3-B. CLI 실행
python app.py --query "Samsung HBM4 Hybrid Bonding 기술 동향 분석"

# 옵션
python app.py --query "..." --no-rag        # RAG 없이 LLM 폴백 모드
python app.py --query "..." --rebuild-rag   # 벡터스토어 강제 재빌드
```

## RAG 평가 기준

| 지표 | 목표값 | 측정 방법 |
|------|--------|-----------|
| Hit Rate@5 | ≥ 0.80 | HBM 도메인 특화 평가셋 5건 |
| MRR | ≥ 0.70 | 첫 번째 정답 순위 기반 역수 평균 |

`main.ipynb` Section 7에서 평가 셀 실행 가능.

## Contributors

- 최태성 : 전반적인 코드 흐름 설계 및 Agent 간 Workflow 구현
- 이다예 : web search agent 설계, api 연동, 최신 정보 수집 로직 고도화
- 신경은 : RAG Agent 성능 개선 (Hybrid Search, Reranker 적용 및 검색 정확도 향상)