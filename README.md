# SK Hynix HBM Competitive Intelligence System

Samsung·Micron의 HBM 관련 R&D 공개 정보를 자동 수집·분석하여 TRL 기반 경쟁사 기술 전략 보고서를 생성하는 멀티에이전트 시스템.

## Overview

- **Objective** : Samsung·Micron의 HBM4 / PIM / CXL R&D 공개 정보를 자동 수집·분석하여 SK Hynix R&D팀이 즉시 활용 가능한 TRL 기반 경쟁사 기술 전략 보고서 생성
- **Method** : Supervisor Pattern 기반 멀티에이전트 워크플로우 — RAG(내부 문서) + Web Search(최신 공개정보) 독립 수집 → TRL 전용 노드 평가 → 보고서 자동 생성
- **Tools** : LangGraph, LangChain, FAISS, BM25, CrossEncoder, Tavily Search, Semantic Scholar, KIPRIS, GPT-4o

## Features

- 내부 기술 문서 기반 정보 추출 (PDF·TXT·JSON → FAISS 벡터스토어, 문서 유형별 청킹 전략 적용)
- 경쟁사 최신 공개정보 수집 (Tavily 웹 검색 + Semantic Scholar 학회 논문 + KIPRIS 특허 + Samsung Careers 채용공고)
- TRL 전용 평가 노드 : RAG·Web 수집 결과를 근거로 HBM4·PIM·CXL 기술별 TRL을 구조화하여 State에 저장, 보고서에 자동 반영
- TRL 기반 경쟁사 비교 매트릭스 자동 생성 (Samsung·Micron × HBM4/PIM/CXL, TRL 4~6 비공개 구간 간접지표 명시)
- 확증 편향 방지 전략 : 웹 검색 시 긍정 쿼리와 반론 쿼리(수율 이슈·지연·한계) 균형 수집, 보고서 작성 시 양면 증거 제시 및 단일 출처 경고 원칙 적용
- RAG 검색 전략 4종 비교 평가 (Naive Dense / Hybrid BM25+RRF / HyDE / MultiQuery — Hit Rate@K, MRR 자동 산출)
- 최종 산출물 자동 PDF 변환 (`outputs/` 디렉토리 저장)

## Tech Stack

| Category | Details |
|----------|---------|
| Framework | LangGraph ≥ 0.2.50, LangChain ≥ 0.3.0, Python 3.11+ |
| LLM | GPT-4o (Supervisor · TRL Evaluator · Report Generator), GPT-4o-mini (RAG · Web) |
| Retrieval | FAISS + BM25 Hybrid Search (RRF) + CrossEncoder Reranker — Hit Rate@5, MRR |
| Embedding | intfloat/multilingual-e5-large (기본값) / BAAI/bge-m3 / jinaai/jina-embeddings-v3 / voyage-3-large |
| Web Search | Tavily Search API, Semantic Scholar API, KIPRIS API, Playwright (Samsung Careers) |
| PDF Export | weasyprint + markdown2 |

## Agents

- **Supervisor** : 쿼리 유형 분류 → 에이전트 위임 → 수집 충분성 판단 → TRL 평가 완료 여부 확인 → END 결정. 벡터스토어 미존재 시 자동 우회 처리
- **RAG Agent** : 내부 벡터스토어(FAISS)에서 HBM 기술 개념 문서 검색. BM25+Dense RRF + CrossEncoder Reranker + HBM 도메인 Query Expansion 적용
- **Web Search Agent** : Tavily·Semantic Scholar·KIPRIS·Samsung Careers 통합 수집. LLM 동적 쿼리 생성 및 확증 편향 방지를 위한 반론 쿼리 자동 추가
- **TRL Evaluator** : RAG·Web 수집 근거를 입력받아 기술별(HBM4·PIM·CXL) × 기업별(SK Hynix·Samsung·Micron) TRL을 구조화 평가. TRL 4~6 비공개 구간은 간접지표 명시 필수
- **Report Generator** : RAG + Web + TRL 평가 결과 통합 → TRL 비교 매트릭스 포함 Markdown 보고서 초안 작성. 섹션 완결성 자동 검증 및 누락 섹션 placeholder 삽입

## Architecture

```
START
  │
  ▼
Supervisor ──► RAG Agent      ──┐
  │        ──► Web Agent      ──┤ → Supervisor
  │        ──► TRL Evaluator  ──┤ → Supervisor  ← RAG+Web 완료 후 자동 실행
  │        ──► Report Generator─┘ → Supervisor
  │
  ▼
 END (PDF 보고서)
```

> 핵심 원칙: 모든 Sub-Agent는 Supervisor로 복귀. RAG+Web 수집 완료 후 TRL Evaluator가 반드시 실행된 뒤 Report Generator로 진행.

## Directory Structure

```
hbm-competitive-intel/
├── data/                        # 내부 분석 문서 (PDF·TXT·JSON)
│   └── vectorstore/             # FAISS 인덱스 (자동 생성)
├── agents/
│   ├── supervisor.py            # 라우팅·충분성 판단·종료 결정
│   ├── rag_agent.py             # Hybrid Search + Reranker 검색
│   ├── web_search_agent.py      # 웹·논문·특허·채용공고 수집
│   ├── trl_evaluator.py         # TRL 구조화 평가 전용 노드
│   └── report_generator.py      # Markdown 보고서 초안 생성
├── graph/
│   ├── state.py                 # AgentState TypedDict
│   └── workflow.py              # LangGraph StateGraph
├── prompts/                     # 에이전트별 프롬프트 템플릿
├── rag/
│   ├── indexer.py               # PDF·TXT·JSON 유형별 청킹 및 색인
│   ├── retriever.py             # HBMRetriever + 4전략 평가
│   └── embeddings.py            # 임베딩 모델 후보군 관리
├── tool_semantic_scholar.py     # Semantic Scholar API 연동
├── tool_kipris.py               # KIPRIS 특허 API 연동
├── tool_careers_crawler.py      # Samsung Careers Playwright 크롤러
├── utils/
│   ├── pdf_exporter.py          # Markdown → PDF 변환
│   └── pretty_print.py          # 스트림 출력 유틸
├── outputs/                     # 생성된 보고서 (.pdf / .md)
├── main.ipynb                   # 실습 진입점
├── app.py                       # CLI 실행 스크립트
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
# 1. 환경 설정
cp .env.example .env          # OPENAI_API_KEY, TAVILY_API_KEY 입력

# 2. 패키지 설치
pip install -r requirements.txt
playwright install chromium   # Samsung Careers 크롤러용

# 3. 벡터스토어 빌드 (최초 1회, data/ 파일 추가 후)
python -m rag.indexer

# 4-A. 노트북 실행 (권장)
jupyter notebook main.ipynb

# 4-B. CLI 실행
python app.py --query "Samsung HBM4 Hybrid Bonding 기술 동향 분석"
```

## RAG 평가 기준

| 지표 | 목표값 | 측정 방법 |
|------|--------|---------|
| Hit Rate@5 | ≥ 0.80 | HBM 도메인 특화 평가셋 (실제 data/ 파일 기준) |
| MRR | ≥ 0.70 | 첫 번째 정답 순위 기반 역수 평균 |

`main.ipynb` Section B에서 Naive / Hybrid / HyDE / MultiQuery 4전략 자동 비교 실행 가능.

## Contributors

- 최태성 : LangGraph Supervisor Pattern 설계, Multi-Agent Workflow 구현, 시스템 통합
- 이다예 : Web Search Agent 설계, Semantic Scholar·KIPRIS·Samsung Careers API 연동, 확증 편향 방지 전략 구현
- 신경은 : RAG Agent 성능 개선 (BM25+Dense Hybrid Search, CrossEncoder Reranker, Query Expansion, 문서 유형별 청킹 전략)