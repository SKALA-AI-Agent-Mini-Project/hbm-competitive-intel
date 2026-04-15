"""
임베딩 모델 설정 — 교수 피드백 반영
"Jina Embedding, Voyage AI voyage-3-large도 검토 가능합니다.
 도메인까지 고려해서 후보군 선정해 주시기 바랍니다."

후보군:
  1. BAAI/bge-m3                    (한·영 혼용, 8192t, 로컬)       ← 기존 1순위
  2. intfloat/multilingual-e5-large (한·영 혼용, 512t, 로컬)
  3. jinaai/jina-embeddings-v3      (한·영, 8192t, 로컬/API)        ← 신규
  4. voyage-3-large                 (영문 중심, API 필요)            ← 신규
  5. jhgan/ko-sroberta-multitask    (한국어 특화, 로컬)

HBM 도메인 선정 기준:
  - 한·영 혼용 전문 용어 처리력 (TSV·Hybrid Bonding·PIM·CXL)
  - 최대 컨텍스트 길이 (기술 문서는 청크가 길 수 있음)
  - Hit Rate@5 도메인 벤치마크 결과
  - 운영 비용 (로컬 vs API)
"""
import os
from typing import Literal

EmbeddingModel = Literal[
    "bge-m3",
    "multilingual-e5-large",
    "jina-v3",
    "voyage-3-large",
    "ko-sroberta",
]

# ── 모델 메타데이터 (비교 참고용) ──────────────────────────
EMBEDDING_CATALOG = {
    "bge-m3": {
        "model_id":    "BAAI/bge-m3",
        "type":        "local",
        "max_tokens":  8192,
        "lang":        "한·영 혼용",
        "note":        "멀티벡터(dense+sparse+colbert). HBM 전문용어 처리 우수. 1순위 기본값.",
        "cost":        "무료",
    },
    "multilingual-e5-large": {
        "model_id":    "intfloat/multilingual-e5-large",
        "type":        "local",
        "max_tokens":  512,
        "lang":        "한·영 혼용",
        "note":        "컨텍스트 길이 제한(512t). 짧은 청크 환경에서 안정적.",
        "cost":        "무료",
    },
    "jina-v3": {
        "model_id":    "jinaai/jina-embeddings-v3",
        "type":        "local",          # Trust Remote Code 필요
        "max_tokens":  8192,
        "lang":        "한·영 혼용",
        "note":        "태스크별 LoRA 어댑터. retrieval 태스크 특화 설정 가능. 신규 후보.",
        "cost":        "무료 (로컬)",
        "install":     "trust_remote_code=True 필요",
    },
    "voyage-3-large": {
        "model_id":    "voyage-3-large",
        "type":        "api",
        "max_tokens":  16000,
        "lang":        "영문 중심 (한국어 지원)",
        "note":        "영문 기술 문서(특허·논문) 검색 성능 최상위권. API 비용 발생.",
        "cost":        "유료 (VOYAGE_API_KEY 필요)",
        "install":     "pip install voyageai",
    },
    "ko-sroberta": {
        "model_id":    "jhgan/ko-sroberta-multitask",
        "type":        "local",
        "max_tokens":  512,
        "lang":        "한국어 특화",
        "note":        "한국어 문서 전용. 영문 특허·논문 혼용 환경에서는 성능 저하 우려.",
        "cost":        "무료",
    },
}


def get_embeddings(model: EmbeddingModel = "bge-m3"):
    """
    임베딩 모델 로드

    Args:
        model: 모델 키 ("bge-m3" / "multilingual-e5-large" / "jina-v3" /
                        "voyage-3-large" / "ko-sroberta")

    HBM 도메인 선정 권고:
        - 로컬 환경  → "bge-m3" (한·영 혼용, 8192t, 무료)
        - 영문 특허 비중 높을 때 → "voyage-3-large" (API)
        - 태스크 튜닝 필요 시  → "jina-v3"
    """
    meta = EMBEDDING_CATALOG.get(model)
    if not meta:
        raise ValueError(f"지원하지 않는 모델: {model}. 선택지: {list(EMBEDDING_CATALOG)}")

    print(f"  🔢 임베딩 모델: {meta['model_id']}")
    print(f"     타입: {meta['type']} | 최대토큰: {meta['max_tokens']} | 비용: {meta['cost']}")

    if meta["type"] == "api":
        return _load_api_embedding(model, meta)
    else:
        return _load_local_embedding(model, meta)


# ── 로컬 모델 ────────────────────────────────────────────────
def _load_local_embedding(model: str, meta: dict):
    from langchain_huggingface import HuggingFaceEmbeddings

    kwargs = {
        "model_name":    meta["model_id"],
        "model_kwargs":  {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32},
    }

    # Jina v3는 trust_remote_code 필요
    if model == "jina-v3":
        kwargs["model_kwargs"]["trust_remote_code"] = True

    return HuggingFaceEmbeddings(**kwargs)


# ── API 모델 ────────────────────────────────────────────────
def _load_api_embedding(model: str, meta: dict):
    if model == "voyage-3-large":
        return _load_voyage(meta)
    raise ValueError(f"API 모델 로더 미구현: {model}")


def _load_voyage(meta: dict):
    """Voyage AI 임베딩 (영문 기술 특허·논문 검색 특화)"""
    try:
        from langchain_voyageai import VoyageAIEmbeddings
    except ImportError:
        raise ImportError("pip install langchain-voyageai 를 먼저 실행하세요.")

    api_key = os.getenv("VOYAGE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "VOYAGE_API_KEY 환경변수가 없습니다.\n"
            ".env 파일에 VOYAGE_API_KEY=pa-... 를 추가하세요.\n"
            "발급: https://www.voyageai.com"
        )

    return VoyageAIEmbeddings(
        model="voyage-3-large",
        voyage_api_key=api_key,
    )


# ── 후보군 요약 출력 ─────────────────────────────────────────
def print_embedding_catalog():
    """임베딩 후보군 비교표 출력"""
    print("\n📊 임베딩 모델 후보군 비교 (HBM 도메인 기준)")
    print(f"{'키':<25} {'타입':<8} {'토큰':<7} {'언어':<15} {'비용':<12} 비고")
    print("-" * 90)
    for key, m in EMBEDDING_CATALOG.items():
        print(
            f"{key:<25} {m['type']:<8} {str(m['max_tokens']):<7} "
            f"{m['lang']:<15} {m['cost']:<12} {m['note'][:40]}"
        )
    print()
