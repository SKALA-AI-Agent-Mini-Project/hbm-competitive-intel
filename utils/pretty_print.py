"""출력 유틸리티 — 예제 코드 스타일 참고"""
from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent: bool = False):
    """개별 메시지 포맷 출력"""
    try:
        pretty = message.pretty_repr(html=False)
    except Exception:
        pretty = str(message)

    if not indent:
        print(pretty)
        return
    indented = "\n".join("\t" + line for line in pretty.split("\n"))
    print(indented)


def pretty_print_messages(update: dict, last_message: bool = False):
    """그래프 스트림 업데이트 출력"""
    is_subgraph = False

    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"📍 서브그래프 [{graph_id}] 업데이트:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        label = f"📌 [{node_name}] 노드 업데이트:"
        if is_subgraph:
            label = "\t" + label
        print(label + "\n")

        if not isinstance(node_update, dict):
            continue

        messages = convert_to_messages(node_update.get("messages", []))
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print()


def print_workflow_summary(final_state: dict):
    """워크플로우 실행 결과 요약 출력"""
    print("\n" + "=" * 70)
    print("🏁 워크플로우 실행 완료")
    print("=" * 70)
    print(f"  분석 주제  : {final_state.get('query', 'N/A')}")
    print(f"  RAG 결과   : {len(final_state.get('rag_results') or [])}건")
    print(f"  Web 결과   : {len(final_state.get('web_results') or [])}건")
    has_report = bool(final_state.get("report"))
    print(f"  보고서     : {'✅ 생성 완료' if has_report else '❌ 생성 실패'}")
    if has_report:
        report_len = len(final_state["report"])
        print(f"  보고서 길이: {report_len:,}자")
    print("=" * 70)
