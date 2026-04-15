"""
RAG Retriever — HBM Competitive R&D Intelligence System
통합: 팀 개선(HBMRetriever — reranker·query expansion·latency 측정·EvalSample)
     + 다중 전략 비교(교수 피드백 — Naive/HyDE/MultiQuery/Hybrid)

사용:
  from rag.retriever import HBMRetriever, evaluate_retriever, print_eval_report
  from rag.retriever import evaluate_retrievers, print_eval_table   # 전략 비교
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).resolve().parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
RRF_K           = 60
USE_RERANKER    = True
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N    = 12


# ════════════════════════════════════════════════════════════
# 평가 구조 (팀 버전 — 실제 파일 기준 eval set)
# ════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    query: str
    relevant_sources: List[str]


@dataclass
class EvalResult:
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    n_queries: int = 0
    avg_latency: float = 0.0
    details: List[Dict] = field(default_factory=list)


HBM_EVAL_SAMPLES: List[EvalSample] = [
    # ── 신규 txt 파일 기반 (경쟁사 분석 특화) ──────────────────
    EvalSample("Samsung HBM4 양산 현황과 Hybrid Bonding 전략",
               ["samsung_hbm4_analysis.txt"]),
    EvalSample("Micron HBM4 Leapfrog 전략과 CHIPS Act 지원",
               ["micron_hbm4_analysis.txt"]),
    EvalSample("Samsung과 Micron의 PIM 기술 TRL 비교",
               ["pim_competitive.txt"]),
    EvalSample("CXL 표준화 경쟁과 Micron의 전략",
               ["cxl_landscape.txt"]),
    EvalSample("HBM4 Hybrid Bonding Cu-Cu 접합 기술",
               ["hbm4_hybrid_bonding.txt"]),
    EvalSample("TRL 간접지표 추정 방법론과 채용공고 분석",
               ["trl_framework.txt"]),
    EvalSample("Samsung PIM 특허 출원 동향",
               ["samsung_hbm4_analysis.txt", "pim_competitive.txt"]),
    EvalSample("SK Hynix AiMX와 경쟁사 PIM 비교",
               ["pim_competitive.txt"]),
    # ── 원본 데이터 파일 기반 ──────────────────────────────────
    EvalSample("HBM 시장 성장 전망과 주요 원인",
               ["samilpwc_semicon-trends-outlook-2026.pdf"]),
    EvalSample("HBM PIM CXL 용어 정의",
               ["semiconductor_terms.json"]),
]


# ════════════════════════════════════════════════════════════
# HBMRetriever — 팀 메인 리트리버 (reranker + query expansion)
# ════════════════════════════════════════════════════════════

class HBMRetriever:
    """BM25 + Dense RRF + CrossEncoder reranker. HBM 도메인 query expansion 포함."""

    name = "Hybrid (BM25+Dense RRF + Reranker)"

    def __init__(
        self,
        vectorstore_path: Path = VECTORSTORE_PATH,
        use_reranker: bool = USE_RERANKER,
    ):
        self._vectorstore_path = vectorstore_path
        self._use_reranker     = use_reranker
        self._faiss: Optional[FAISS]            = None
        self._bm25: Optional[BM25Retriever]     = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._reranker = None

    def _load(self) -> None:
        if self._faiss is not None:
            return

        if not self._vectorstore_path.exists():
            raise FileNotFoundError(
                f"벡터스토어 없음: {self._vectorstore_path}\n"
                "먼저 'python -m rag.indexer' 를 실행하세요."
            )

        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._faiss = FAISS.load_local(
            str(self._vectorstore_path),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        all_docs = list(self._faiss.docstore._dict.values())
        if all_docs:
            self._bm25 = BM25Retriever.from_documents(all_docs, k=RERANK_TOP_N)

        if self._use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(RERANKER_MODEL)
                logger.info(f"Reranker 로드: {RERANKER_MODEL}")
            except Exception as e:
                logger.warning(f"Reranker 로드 실패, 비활성화: {e}")

        logger.info(f"HBMRetriever 초기화 완료 (문서 {len(all_docs)}개)")

    # ── query expansion ────────────────────────────────────
    def _expand_query(self, query: str) -> str:
        expansions, q = [], query.lower()
        if "hbm"            in q: expansions += ["high bandwidth memory", "tsv", "stacked dram"]
        if "pim"            in q: expansions += ["processing in memory", "in-memory computing"]
        if "cxl"            in q: expansions += ["compute express link", "memory pooling"]
        if "hybrid bonding" in q: expansions += ["3d integration", "advanced packaging"]
        if any(w in q for w in ["시장", "성장"]): expansions += ["market", "growth", "trend"]
        if "ai"             in q: expansions += ["ai accelerator", "gpu", "data center"]
        return (query + " " + " ".join(expansions)).strip() if expansions else query

    def _doc_id(self, doc: Document) -> str:
        m = doc.metadata
        return f"{m.get('source','')}__p{m.get('page','')}__c{m.get('chunk_index','')}__t{m.get('term','')}"

    # ── core search ────────────────────────────────────────
    def _dense(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self._faiss.similarity_search_with_score(query, k=k)

    def _bm25_search(self, query: str) -> List[Document]:
        if not self._bm25:
            return []
        try:
            return self._bm25.invoke(self._expand_query(query))
        except Exception as e:
            logger.warning(f"BM25 실패: {e}")
            return []

    def _rrf(
        self,
        dense: List[Tuple[Document, float]],
        bm25:  List[Document],
    ) -> List[Tuple[Document, float]]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for rank, (doc, _) in enumerate(dense, 1):
            did = self._doc_id(doc)
            scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
            doc_map[did] = doc
        for rank, doc in enumerate(bm25, 1):
            did = self._doc_id(doc)
            scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
            doc_map[did] = doc
        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [(doc_map[d], scores[d]) for d in ranked]

    def _rerank(
        self,
        query:      str,
        candidates: List[Tuple[Document, float]],
        top_n:      int,
    ) -> List[Tuple[Document, float]]:
        if not self._reranker:
            return candidates[:top_n]
        pairs = [(query, doc.page_content) for doc, _ in candidates[:RERANK_TOP_N]]
        try:
            scores  = self._reranker.predict(pairs)
            rescored = sorted(
                zip([doc for doc, _ in candidates[:RERANK_TOP_N]], scores),
                key=lambda x: x[1], reverse=True,
            )
            return [(doc, float(s)) for doc, s in rescored[:top_n]]
        except Exception as e:
            logger.warning(f"Reranker 실패: {e}")
            return candidates[:top_n]

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        self._load()
        dense  = self._dense(query, k=max(k * 4, RERANK_TOP_N))
        bm25   = self._bm25_search(query)
        merged = self._rrf(dense, bm25)
        return self._rerank(query, merged, top_n=k)

    # invoke() 인터페이스 (교수 피드백 다중 전략 비교와 호환)
    def invoke(self, query: str) -> List[Document]:
        return [doc for doc, _ in self.retrieve(query)]


# ════════════════════════════════════════════════════════════
# 교수 피드백 — 다중 전략 비교용 경량 리트리버들
# ════════════════════════════════════════════════════════════

class _NaiveRetriever:
    name = "Naive (Dense)"
    def __init__(self, faiss: FAISS, k: int = 5):
        self._f, self.k = faiss, k
    def invoke(self, q: str) -> List[Document]:
        return self._f.similarity_search(q, k=self.k)


class _HyDERetriever:
    name = "HyDE"
    def __init__(self, faiss: FAISS, k: int = 5, model: str = "gpt-4o-mini"):
        self._f, self.k, self._m = faiss, k, model
    def invoke(self, q: str) -> List[Document]:
        hypo = self._gen_hypo(q)
        return self._f.similarity_search(hypo, k=self.k)
    def _gen_hypo(self, q: str) -> str:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            r = ChatOpenAI(model=self._m, temperature=0.3).invoke(
                [HumanMessage(content=f"반도체 기술 문서 단락(200자):\n{q}")]
            )
            return r.content
        except Exception:
            return q


class _MultiQueryRetriever:
    name = "MultiQuery"
    def __init__(self, faiss: FAISS, k: int = 5, n: int = 3, model: str = "gpt-4o-mini"):
        self._f, self.k, self._n, self._m = faiss, k, n, model
    def invoke(self, q: str) -> List[Document]:
        variants = self._variants(q)
        seen, docs = set(), []
        for v in variants:
            for d in self._f.similarity_search(v, k=self.k):
                key = d.page_content[:100]
                if key not in seen:
                    seen.add(key); docs.append(d)
        return docs[:self.k]
    def _variants(self, q: str) -> List[str]:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            r = ChatOpenAI(model=self._m, temperature=0.5).invoke([
                HumanMessage(content=f"HBM 도메인 쿼리 {self._n}가지 변형:\n{q}")
            ])
            vs = [x.strip() for x in r.content.strip().split("\n") if x.strip()]
            return [q] + vs[:self._n]
        except Exception:
            return [q]


def evaluate_retrievers(
    vectorstore: FAISS,
    all_chunks: List[Document],
    eval_set: List[Dict],
    k: int = 5,
) -> Dict[str, Dict]:
    """
    4가지 전략 비교 (교수 피드백).
    eval_set: [{"query": str, "expected_source": str}, ...]
    """
    from langchain_community.retrievers import BM25Retriever as _BM25

    class _HybridSimple:
        name = "Hybrid (BM25+Dense RRF)"
        def __init__(self, vs, chunks, k):
            self._vs = vs; self.k = k
            bm25 = _BM25.from_documents(chunks, k=k * 2)
            bm25.k = k * 2
            self._bm25 = bm25
        def invoke(self, q):
            scores: Dict[str, float] = {}
            dmap: Dict[str, Document] = {}
            def key(d): return d.page_content[:100]
            for i, d in enumerate(self._bm25.invoke(q), 1):
                k_ = key(d); scores[k_] = scores.get(k_, 0) + 1/(RRF_K+i); dmap[k_] = d
            for i, d in enumerate(self._vs.similarity_search(q, k=self.k*2), 1):
                k_ = key(d); scores[k_] = scores.get(k_, 0) + 1/(RRF_K+i); dmap[k_] = d
            return [dmap[k] for k in sorted(scores, key=lambda x: scores[x], reverse=True)[:self.k]]

    strategies = {
        "Naive (Dense)":     _NaiveRetriever(vectorstore, k=k),
        "Hybrid (BM25+RRF)": _HybridSimple(vectorstore, all_chunks, k=k),
        "HyDE":              _HyDERetriever(vectorstore, k=k),
        "MultiQuery":        _MultiQueryRetriever(vectorstore, k=k),
    }
    results = {}
    for name, ret in strategies.items():
        print(f"  📏 평가: {name} ...", end=" ", flush=True)
        m = _compute_metrics_simple(ret, eval_set, k)
        results[name] = m
        print(f"Hit@{k}={m[f'Hit Rate@{k}']:.3f}  MRR={m['MRR']:.3f}")
    return results


def _compute_metrics_simple(retriever, eval_set, k):
    hits, rrs = 0, []
    for item in eval_set:
        try:
            docs = retriever.invoke(item["query"])[:k]
        except Exception:
            docs = []
        sources = [d.metadata.get("source", "") for d in docs]
        expected = item["expected_source"]
        hit = any(expected in s for s in sources)
        if hit:
            hits += 1
            rrs.append(next((1/r for r, s in enumerate(sources, 1) if expected in s), 0.0))
        else:
            rrs.append(0.0)
    n = len(eval_set)
    return {f"Hit Rate@{k}": round(hits/n, 3) if n else 0.0,
            "MRR": round(sum(rrs)/n, 3) if n else 0.0}


def print_eval_table(results: Dict[str, Dict], k: int = 5):
    print(f"\n{'='*60}\n📊 RAG 검색 전략 비교 (k={k})\n{'='*60}")
    print(f"  {'전략':<25} {'Hit Rate@'+str(k):<14} {'MRR'}")
    print(f"  {'-'*50}")
    bhr = max(v[f"Hit Rate@{k}"] for v in results.values())
    bmr = max(v["MRR"] for v in results.values())
    for name, m in results.items():
        hr, mr = m[f"Hit Rate@{k}"], m["MRR"]
        rec = "★" if hr == bhr and mr == bmr else ""
        print(f"  {name:<25} {'✅' if hr>=0.80 else '⚠️ '}{hr:.3f}{'':8} {'✅' if mr>=0.70 else '⚠️ '}{mr:.3f}  {rec}")
    print(f"\n  기준: Hit Rate@{k}≥0.80 / MRR≥0.70\n{'='*60}\n")


# ════════════════════════════════════════════════════════════
# 팀 버전 단일 리트리버 평가 (EvalSample 기반)
# ════════════════════════════════════════════════════════════

def evaluate_retriever(
    retriever: HBMRetriever,
    eval_samples: Optional[List[EvalSample]] = None,
    k_values: List[int] = [1, 3, 5],
) -> EvalResult:
    samples = eval_samples or HBM_EVAL_SAMPLES
    result  = EvalResult(n_queries=len(samples))
    hits    = {k: 0 for k in k_values}
    rr_sum  = 0.0
    total_t = 0.0
    details = []

    for sample in samples:
        max_k = max(k_values)
        try:
            t0       = time.perf_counter()
            retrieved = retriever.retrieve(sample.query, k=max_k)
            elapsed  = time.perf_counter() - t0
        except FileNotFoundError:
            logger.error("벡터스토어 없음 — 평가 중단")
            break

        total_t += elapsed
        rr       = _reciprocal_rank(retrieved, sample.relevant_sources)
        rr_sum  += rr
        detail   = {"query": sample.query, "reciprocal_rank": round(rr, 4),
                    "latency": round(elapsed, 4), "hits": {}}

        for k in k_values:
            hit = _is_hit(retrieved[:k], sample.relevant_sources)
            hits[k] += int(hit)
            detail["hits"][f"@{k}"] = hit
        details.append(detail)

    n = result.n_queries
    result.hit_rate_at_k = {k: round(hits[k]/n, 4) for k in k_values} if n else {}
    result.mrr            = round(rr_sum/n, 4) if n else 0.0
    result.avg_latency    = round(total_t/n, 4) if n else 0.0
    result.details        = details
    return result


def _is_hit(retrieved: List[Tuple[Document, float]], relevant: List[str]) -> bool:
    srcs = {Path(doc.metadata.get("source", "")).name for doc, _ in retrieved}
    return any(Path(r).name in srcs for r in relevant)


def _reciprocal_rank(retrieved: List[Tuple[Document, float]], relevant: List[str]) -> float:
    names = {Path(r).name for r in relevant}
    for rank, (doc, _) in enumerate(retrieved, 1):
        if Path(doc.metadata.get("source", "")).name in names:
            return 1.0 / rank
    return 0.0


def print_eval_report(result: EvalResult) -> None:
    print("\n" + "="*50)
    print("📊 RAG 검색 품질 평가 결과")
    print("="*50)
    print(f"  평가 쿼리 수 : {result.n_queries}")
    for k, hr in result.hit_rate_at_k.items():
        status = "✅" if (k == 5 and hr >= 0.80) else ("⚠️" if k == 5 else " ")
        print(f"  Hit Rate@{k} : {hr:.4f}  {status}")
    print(f"  MRR         : {result.mrr:.4f}  {'✅' if result.mrr >= 0.70 else '⚠️'}")
    print(f"  Avg Latency : {result.avg_latency:.4f}s")
    print("-"*50)
    for d in result.details:
        hs = "  ".join(f"{k}={'O' if v else 'X'}" for k, v in d["hits"].items())
        print(f"  [{d['reciprocal_rank']:.2f}] {d['query'][:40]:<40} | {hs}")
    print("="*50 + "\n")


# public alias (main.ipynb import 호환)
NaiveRetriever    = _NaiveRetriever
HyDERetriever     = _HyDERetriever
MultiQueryRetriever = _MultiQueryRetriever

# 기존 build_hybrid_retriever / retrieve_with_scores 호환 인터페이스 유지
def build_hybrid_retriever(vectorstore, all_chunks, k=5, **kwargs) -> HBMRetriever:
    """구버전 호환 — HBMRetriever 반환."""
    ret = HBMRetriever(use_reranker=False)
    ret._faiss = vectorstore
    all_docs   = list(vectorstore.docstore._dict.values()) or all_chunks
    ret._bm25  = BM25Retriever.from_documents(all_docs, k=k * 2)
    ret._bm25.k = k * 2
    return ret


def retrieve_with_scores(retriever, query: str, k: int = 5) -> List[Dict]:
    """구버전 호환 — AgentState 형식 반환."""
    if hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(query, k=k)
    else:
        docs = [(d, 0.5) for d in retriever.invoke(query)[:k]]
    return [
        {"chunk": doc.page_content, "summary": doc.page_content[:150] + "...",
         "source": doc.metadata.get("source", "unknown"), "score": round(float(sc), 2)}
        for doc, sc in docs
    ]


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s | %(message)s")
    r = HBMRetriever(use_reranker=True)
    result = evaluate_retriever(r)
    print_eval_report(result)