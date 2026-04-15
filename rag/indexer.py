# 작성자: Kyungeun Shin
# 목적: PDF/TXT/MD/JSON 문서를 로드하고, 문서 유형별 청킹 후 FAISS 벡터스토어를 생성한다.
# 작성일: 2026-04-15
# 변경사항:
# - PDF/TXT/MD/JSON 분리 로딩
# - JSON 항목 단위 Document 변환
# - 문서 유형별 chunking 분리
# - metadata 강화
# - 빈 파일 및 예외 처리 강화

"""
RAG Indexer — 문서 청킹·임베딩·FAISS 색인

실행:
    python -m rag.indexer
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

PDF_CHUNK_SIZE = 900
PDF_CHUNK_OVERLAP = 120

TEXT_CHUNK_SIZE = 800
TEXT_CHUNK_OVERLAP = 100

JSON_CHUNK_SIZE = 320
JSON_CHUNK_OVERLAP = 40


def _normalize_text(text: str) -> str:
    """공백을 정리한다."""
    return " ".join((text or "").split()).strip()


def _safe_read_json(path: Path) -> Any:
    """JSON 파일을 안전하게 읽는다."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_to_documents(data: Any, source_name: str) -> list[Document]:
    """JSON 구조를 검색 친화적인 Document 목록으로 변환한다."""
    docs: list[Document] = []

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_key = f"{prefix}.{key}" if prefix else str(key)

                if isinstance(value, (dict, list)):
                    walk(value, current_key)
                else:
                    text = _normalize_text(str(value))
                    if not text:
                        continue

                    content = f"항목: {current_key}\n값: {text}"
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": source_name,
                                "doc_type": "json",
                                "term": current_key,
                            },
                        )
                    )

        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                current_key = f"{prefix}[{idx}]"
                if isinstance(item, (dict, list)):
                    walk(item, current_key)
                else:
                    text = _normalize_text(str(item))
                    if not text:
                        continue

                    content = f"항목: {current_key}\n값: {text}"
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": source_name,
                                "doc_type": "json",
                                "term": current_key,
                            },
                        )
                    )

    walk(data)
    return docs


def load_documents() -> list[Document]:
    """data/ 디렉토리의 문서를 로드한다."""
    docs: list[Document] = []

    for path in DATA_DIR.glob("**/*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                if path.stat().st_size == 0:
                    logger.warning(f"빈 PDF 파일 건너뜀: {path}")
                    continue

                loader = PyPDFLoader(str(path))
                pdf_docs = loader.load()

                for d in pdf_docs:
                    d.page_content = _normalize_text(d.page_content)
                    d.metadata["source"] = path.name
                    d.metadata["doc_type"] = "pdf"
                    d.metadata["path"] = str(path.relative_to(PROJECT_ROOT))

                docs.extend([d for d in pdf_docs if d.page_content])
                logger.info(f"PDF 로드: {path}")

            elif suffix in {".txt", ".md"}:
                if path.stat().st_size == 0:
                    logger.warning(f"빈 텍스트 파일 건너뜀: {path}")
                    continue

                loader = TextLoader(str(path), encoding="utf-8")
                text_docs = loader.load()

                for d in text_docs:
                    d.page_content = _normalize_text(d.page_content)
                    d.metadata["source"] = path.name
                    d.metadata["doc_type"] = "text"
                    d.metadata["path"] = str(path.relative_to(PROJECT_ROOT))

                docs.extend([d for d in text_docs if d.page_content])
                logger.info(f"텍스트 로드: {path}")

            elif suffix == ".json":
                if path.stat().st_size == 0:
                    logger.warning(f"빈 JSON 파일 건너뜀: {path}")
                    continue

                data = _safe_read_json(path)
                json_docs = _json_to_documents(data, source_name=path.name)

                for d in json_docs:
                    d.metadata["path"] = str(path.relative_to(PROJECT_ROOT))

                docs.extend(json_docs)
                logger.info(f"JSON 로드: {path} ({len(json_docs)}개 항목)")

        except Exception as e:
            logger.warning(f"문서 로드 실패 {path}: {e}")

    return docs


def _split_by_type(docs: list[Document]) -> list[Document]:
    """문서 유형별로 분리 청킹한다."""
    pdf_docs = [d for d in docs if d.metadata.get("doc_type") == "pdf"]
    text_docs = [d for d in docs if d.metadata.get("doc_type") == "text"]
    json_docs = [d for d in docs if d.metadata.get("doc_type") == "json"]

    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    json_splitter = RecursiveCharacterTextSplitter(
        chunk_size=JSON_CHUNK_SIZE,
        chunk_overlap=JSON_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ": ", ", ", " ", ""],
    )

    chunks: list[Document] = []
    chunks.extend(pdf_splitter.split_documents(pdf_docs))
    chunks.extend(text_splitter.split_documents(text_docs))
    chunks.extend(json_splitter.split_documents(json_docs))

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    return chunks


def build_index() -> FAISS | None:
    """문서를 청킹하여 FAISS 벡터스토어를 구성한다."""
    logger.info("문서 로드 시작...")
    docs = load_documents()

    if not docs:
        logger.warning(f"data/ 디렉토리에 색인 가능한 문서가 없습니다. 경로: {DATA_DIR}")
        return None

    chunks = _split_by_type(docs)
    logger.info(f"총 {len(chunks)}개 청크 생성 (원본 문서: {len(docs)}개)")

    logger.info(f"임베딩 모델 로드: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("FAISS 색인 구성 중...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    logger.info(f"벡터스토어 저장 완료: {VECTORSTORE_PATH}")

    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_index()