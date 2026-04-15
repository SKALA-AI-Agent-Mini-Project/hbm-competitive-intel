"""
FAISS 벡터스토어 설정 및 문서 인덱싱
설계서 2-2: FAISS 기반 벡터스토어
"""
import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings

# 벡터스토어 저장 경로
VECTORSTORE_PATH = Path(__file__).parent.parent / "data" / "vectorstore"


def load_documents(data_dir: str = "data") -> List[Document]:
    """data/ 디렉토리에서 문서 로드 (PDF, TXT)"""
    docs = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"[경고] {data_dir} 디렉토리가 없습니다.")
        return docs

    # PDF 로드
    pdf_files = list(data_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs.extend(loader.load())
            print(f"  ✓ PDF 로드: {pdf_file.name} ({len(docs)}개 페이지)")
        except Exception as e:
            print(f"  ✗ PDF 로드 실패: {pdf_file.name} - {e}")

    # TXT 로드
    txt_files = list(data_path.glob("*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs.extend(loader.load())
            print(f"  ✓ TXT 로드: {txt_file.name}")
        except Exception as e:
            print(f"  ✗ TXT 로드 실패: {txt_file.name} - {e}")

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """문서 청킹 — HBM 기술 용어 기준 최적화"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


def build_vectorstore(
    data_dir: str = "data",
    save_path: str = None,
    model_name: str = "BAAI/bge-m3",
) -> FAISS:
    """벡터스토어 빌드 및 저장"""
    print("📂 문서 로딩 중...")
    docs = load_documents(data_dir)

    if not docs:
        print("⚠️  문서가 없습니다. 샘플 문서로 초기화합니다.")
        docs = _get_sample_documents()

    print(f"✂️  문서 청킹 중... (총 {len(docs)}개 문서)")
    chunks = split_documents(docs)
    print(f"  → {len(chunks)}개 청크 생성")

    print(f"🔢 임베딩 생성 중... (모델: {model_name})")
    embeddings = get_embeddings(model_name)

    print("🗄️  FAISS 인덱스 빌드 중...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 저장
    save_dir = save_path or str(VECTORSTORE_PATH)
    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(save_dir)
    print(f"✅ 벡터스토어 저장 완료: {save_dir}")

    return vectorstore


def load_vectorstore(
    save_path: str = None,
    model_name: str = "BAAI/bge-m3",
) -> FAISS:
    """저장된 벡터스토어 로드"""
    save_dir = save_path or str(VECTORSTORE_PATH)
    embeddings = get_embeddings(model_name)

    if not Path(save_dir).exists():
        print(f"⚠️  저장된 벡터스토어가 없습니다. 새로 빌드합니다.")
        return build_vectorstore(model_name=model_name)

    vectorstore = FAISS.load_local(
        save_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ 벡터스토어 로드 완료: {save_dir}")
    return vectorstore


def _get_sample_documents() -> List[Document]:
    """샘플 HBM 기술 문서 (data/ 디렉토리 비어있을 때 사용)"""
    sample_texts = [
        {
            "content": """HBM (High Bandwidth Memory) 개요
HBM은 DRAM 다이를 수직으로 적층하여 TSV(Through-Silicon Via)로 연결하는 고대역폭 메모리 기술이다.
SK Hynix는 HBM3E 12-Hi(36GB) 제품을 NVIDIA H200·B200에 공급하며 세계 1위 점유율을 유지하고 있다.
HBM4는 16단 이상 적층을 목표로 하며, Hybrid Bonding 기술 도입으로 기존 MR-MUF 대비 
전력·열 효율을 개선할 것으로 예상된다.""",
            "source": "HBM_기술개요.txt",
        },
        {
            "content": """PIM (Processing-In-Memory) 기술 현황
PIM은 메모리 내부에 연산 유닛을 배치하여 DRAM↔CPU 간 데이터 이동을 최소화하는 기술이다.
SK Hynix는 AiMX(AI Memory eXtension)를 출시하며 초기 상용화 단계(TRL 5~6)에 진입했다.
주요 기술적 도전: 열 집중 문제, 전력 밀도 증가, 수율 저하.
Samsung은 PIM 관련 특허를 SK Hynix 대비 3~4배 많이 출원한 상태이다.""",
            "source": "PIM_기술현황.txt",
        },
        {
            "content": """CXL (Compute Express Link) 기술 현황
CXL은 CPU와 메모리·가속기 간 고속 인터커넥트 표준으로, 메모리 풀링을 가능하게 한다.
SK Hynix CMM-DDR5(CXL 2.0 지원)를 출시하여 TRL 7~8 수준에 도달했다.
CXL 3.0은 메모리 풀링 및 P2P 통신을 지원하며 AI 데이터센터 핵심 인프라로 부상 중이다.
Micron은 CXL 컨소시엄에서 표준 제정을 주도하여 시장 표준 선점을 시도하고 있다.""",
            "source": "CXL_기술현황.txt",
        },
        {
            "content": """Hybrid Bonding 기술
Hybrid Bonding은 Cu-Cu 직접 접합 방식으로 기존 솔더 범프 대비 피치를 획기적으로 줄이는 기술이다.
HBM4 이상에서 MR-MUF(Mass Reflow-Molded Underfill)를 대체할 핵심 공정으로 부상했다.
SK Hynix는 AMAT·BESI 장비 기업과 협력하여 Hybrid Bonding 도입을 추진 중이다.
TSV 수율과 함께 HBM4 양산성을 결정하는 핵심 공정 파라미터로 평가된다.""",
            "source": "HybridBonding_기술.txt",
        },
        {
            "content": """TRL (Technology Readiness Level) 프레임워크
TRL은 기술 성숙도를 1~9단계로 평가하는 프레임워크이다.
- TRL 1~3: 기초 연구 및 개념 검증
- TRL 4~6: 개발 및 시제품 검증 (비공개 구간 → 간접지표로만 추정 가능)
- TRL 7~9: 실증 및 양산 배치
HBM 도메인에서 TRL 4~6 구간의 수율·공정 파라미터는 영업비밀로 직접 확인이 불가하다.
특허 출원 패턴, 학회 발표 빈도, 채용공고 키워드를 통한 간접 추정이 최선이다.""",
            "source": "TRL_프레임워크.txt",
        },
    ]

    docs = []
    for item in sample_texts:
        doc = Document(
            page_content=item["content"],
            metadata={"source": item["source"]},
        )
        docs.append(doc)
    return docs
