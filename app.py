"""
app.py — SK Hynix HBM 경쟁사 기술 동향 분석 시스템
CLI 실행 진입점

사용 예:
  python app.py
  python app.py --query "Samsung HBM4 Hybrid Bonding 기술 동향"
  python app.py --query "Micron CXL 전략" --no-rag
"""
import argparse
import sys
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── 경로 설정 ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from graph.workflow import build_graph, get_initial_state
from agents.rag_agent import init_rag_retriever
from utils.pdf_exporter import export_to_pdf
from utils.pretty_print import pretty_print_messages, print_workflow_summary


def setup_rag(data_dir: str = "data", force_rebuild: bool = False):
    """RAG 파이프라인 초기화"""
    print("🔧 RAG 파이프라인 초기화 중...")
    try:
        from rag.vectorstore import build_vectorstore, load_vectorstore
        from rag.retriever import build_hybrid_retriever
        from rag.vectorstore import _get_sample_documents, split_documents, load_documents

        vs_path = os.path.join(data_dir, "vectorstore")

        if force_rebuild or not Path(vs_path).exists():
            vectorstore = build_vectorstore(data_dir=data_dir)
        else:
            vectorstore = load_vectorstore(save_path=vs_path)

        # BM25용 청크 로드
        docs = load_documents(data_dir)
        if not docs:
            docs = _get_sample_documents()
        all_chunks = split_documents(docs) if docs else _get_sample_documents()

        retriever = build_hybrid_retriever(vectorstore, all_chunks, k=5)
        init_rag_retriever(retriever, all_chunks)
        print("  ✅ RAG 파이프라인 준비 완료\n")
        return True

    except Exception as e:
        print(f"  ⚠️  RAG 초기화 실패: {e}")
        print("  → LLM 폴백 모드로 실행합니다.\n")
        return False


def run_workflow(query: str, verbose: bool = True) -> dict:
    """워크플로우 실행 및 최종 State 반환"""
    print(f"\n{'='*70}")
    print(f"🚀 HBM 경쟁사 분석 워크플로우 시작")
    print(f"   분석 주제: {query}")
    print(f"{'='*70}\n")

    graph = build_graph(use_checkpointer=True)
    initial_state = get_initial_state(query)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final_state = None

    for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
        if verbose:
            pretty_print_messages(chunk, last_message=True)
        # 마지막 상태 추적
        for node_name, node_update in chunk.items():
            if isinstance(node_update, dict):
                if final_state is None:
                    final_state = initial_state.copy()
                final_state.update(node_update)

    return final_state or {}


def save_outputs(final_state: dict) -> str:
    """보고서 저장 (PDF 또는 Markdown)"""
    report = final_state.get("report", "")
    if not report:
        print("⚠️  생성된 보고서가 없습니다.")
        return ""

    query = final_state.get("query", "분석보고서")
    safe_query = query[:30].replace(" ", "_").replace("/", "-")
    filename = f"HBM_{safe_query}"

    saved_path = export_to_pdf(report, filename=filename)
    return saved_path


def main():
    parser = argparse.ArgumentParser(
        description="SK Hynix HBM 경쟁사 기술 동향 분석 시스템"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Samsung과 Micron의 HBM4 및 PIM 기술 동향을 분석하고 TRL 기반으로 비교해주세요.",
        help="분석 주제 쿼리",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="RAG 파이프라인 초기화 건너뜀 (LLM 폴백 모드)",
    )
    parser.add_argument(
        "--rebuild-rag",
        action="store_true",
        help="벡터스토어 강제 재빌드",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="내부 분석 문서 디렉토리 (default: data/)",
    )
    parser.add_argument(
        "--quiet", "-q2",
        action="store_true",
        help="중간 노드 출력 숨김",
    )
    args = parser.parse_args()

    # RAG 초기화
    if not args.no_rag:
        setup_rag(data_dir=args.data_dir, force_rebuild=args.rebuild_rag)

    # 워크플로우 실행
    final_state = run_workflow(
        query=args.query,
        verbose=not args.quiet,
    )

    # 결과 요약
    print_workflow_summary(final_state)

    # 보고서 저장
    saved_path = save_outputs(final_state)
    if saved_path:
        print(f"\n📄 저장 경로: {saved_path}")

    return final_state


if __name__ == "__main__":
    main()
