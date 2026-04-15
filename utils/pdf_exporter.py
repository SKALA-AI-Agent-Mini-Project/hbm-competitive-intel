"""
Markdown → PDF 변환 유틸리티
최종 산출물: outputs/ 디렉토리에 보고서 저장

폴백 순서:
  1. weasyprint + markdown2  (권장 — 한글 테이블 완벽 지원)
  2. HTML 저장               (브라우저에서 인쇄 → PDF 가능)
  3. Markdown 저장           (최후 폴백)
"""
import os
import re
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def export_to_pdf(
    markdown_content: str,
    filename: str = None,
    output_dir: str = None,
) -> str:
    """
    Markdown 보고서를 PDF(또는 HTML/MD)로 저장

    Returns: 저장된 파일 경로 (str)
    """
    save_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = filename or f"HBM_경쟁사분석보고서_{ts}"

    # 1순위: weasyprint
    try:
        return _export_weasyprint(markdown_content, save_dir, base_name)
    except ImportError:
        print("  ⚠️  weasyprint 미설치 → HTML 저장으로 전환")
        print("  💡 PDF가 필요하다면: pip install weasyprint markdown2")
    except Exception as e:
        print(f"  ⚠️  weasyprint 오류: {e} → HTML 저장으로 전환")

    # 2순위: HTML 저장 (브라우저 인쇄 → PDF 변환 가능)
    try:
        return _export_html(markdown_content, save_dir, base_name)
    except Exception as e:
        print(f"  ⚠️  HTML 저장 오류: {e} → Markdown 저장으로 전환")

    # 최후 폴백: .md 저장
    return _export_markdown(markdown_content, save_dir, base_name)


# ────────────────────────────────────────────────────────────
# 1. weasyprint (PDF 직접 생성)
# ────────────────────────────────────────────────────────────
def _export_weasyprint(content: str, save_dir: Path, base_name: str) -> str:
    import markdown2
    from weasyprint import HTML, CSS

    html_body = markdown2.markdown(
        content,
        extras=["tables", "fenced-code-blocks", "strike", "footnotes"],
    )
    css = CSS(string="""
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
        body {
            font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
            font-size: 11pt; line-height: 1.7; color: #1a1a1a; margin: 2cm 2.5cm;
        }
        h1 { font-size: 18pt; color: #003087; border-bottom: 2px solid #003087; padding-bottom: 6px; }
        h2 { font-size: 14pt; color: #003087; margin-top: 24px; }
        h3 { font-size: 12pt; color: #444; }
        table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
        th { background: #003087; color: white; padding: 8px 10px; text-align: center; }
        td { border: 1px solid #ccc; padding: 6px 10px; }
        tr:nth-child(even) { background: #f5f7fb; }
        code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 9pt; }
        @page {
            size: A4; margin: 2cm 2.5cm;
            @bottom-right { content: "- " counter(page) " -"; font-size: 9pt; color: #888; }
            @bottom-left  { content: "SK Hynix | HBM 경쟁사 기술 동향 분석 보고서"; font-size: 9pt; color: #888; }
        }
    """)
    full_html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"></head>
<body>{html_body}</body></html>"""

    out_path = save_dir / f"{base_name}.pdf"
    HTML(string=full_html).write_pdf(str(out_path), stylesheets=[css])
    print(f"  ✅ PDF 저장 완료 (weasyprint): {out_path}")
    return str(out_path)


# ────────────────────────────────────────────────────────────
# 2. HTML 저장 (한글 완벽 지원, 브라우저 인쇄로 PDF 변환 가능)
# ────────────────────────────────────────────────────────────
def _export_html(content: str, save_dir: Path, base_name: str) -> str:
    # markdown2 있으면 사용, 없으면 간단 변환
    try:
        import markdown2
        html_body = markdown2.markdown(
            content,
            extras=["tables", "fenced-code-blocks", "strike"],
        )
    except ImportError:
        html_body = _markdown_to_html_simple(content)

    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HBM 경쟁사 기술 동향 분석 보고서</title>
  <style>
    body {{
      font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif;
      font-size: 13px; line-height: 1.8; color: #1a1a1a;
      max-width: 960px; margin: 40px auto; padding: 0 30px;
    }}
    h1 {{ font-size: 22px; color: #003087; border-bottom: 2px solid #003087; padding-bottom: 8px; }}
    h2 {{ font-size: 17px; color: #003087; margin-top: 32px; border-left: 4px solid #003087; padding-left: 10px; }}
    h3 {{ font-size: 14px; color: #333; margin-top: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 12px; }}
    th {{ background: #003087; color: white; padding: 10px 12px; text-align: center; }}
    td {{ border: 1px solid #ccc; padding: 8px 12px; }}
    tr:nth-child(even) {{ background: #f5f7fb; }}
    blockquote {{ border-left: 4px solid #003087; margin: 0; padding-left: 16px; color: #555; font-style: italic; }}
    code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 11px; }}
    pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
    .print-hint {{
      background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
      padding: 12px 16px; margin-bottom: 24px; font-size: 12px; color: #856404;
    }}
    @media print {{
      .print-hint {{ display: none; }}
      body {{ margin: 0; padding: 0; }}
    }}
  </style>
</head>
<body>
  <div class="print-hint">
    💡 <strong>PDF로 저장하려면</strong>: 브라우저에서 <strong>Ctrl+P (또는 Cmd+P)</strong> →
    프린터를 <strong>"PDF로 저장"</strong> 선택 → 저장
  </div>
  {html_body}
</body>
</html>"""

    out_path = save_dir / f"{base_name}.html"
    out_path.write_text(full_html, encoding="utf-8")
    print(f"  ✅ HTML 저장 완료: {out_path}")
    print(f"  💡 PDF 변환: 브라우저에서 파일 열기 → Ctrl+P → 'PDF로 저장'")
    return str(out_path)


# ────────────────────────────────────────────────────────────
# 3. Markdown 저장 (최후 폴백)
# ────────────────────────────────────────────────────────────
def _export_markdown(content: str, save_dir: Path, base_name: str) -> str:
    out_path = save_dir / f"{base_name}.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"  ✅ Markdown 저장 완료: {out_path}")
    print(f"  💡 PDF 변환 옵션:")
    print(f"     - pip install weasyprint markdown2  (자동 변환)")
    print(f"     - VS Code의 'Markdown PDF' 확장 사용")
    return str(out_path)


# ────────────────────────────────────────────────────────────
# 내부 유틸 — markdown2 없을 때 간단 HTML 변환
# ────────────────────────────────────────────────────────────
def _markdown_to_html_simple(md: str) -> str:
    """markdown2 미설치 시 기본 Markdown → HTML 변환"""
    lines = md.split("\n")
    html_lines = []
    in_table = False

    for line in lines:
        # 제목
        if line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        # 테이블
        elif line.startswith("|"):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            if re.match(r'^\|[-| :]+\|$', line):
                continue  # 구분선 행 스킵
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if not any("<td>" in l for l in html_lines[-5:]) else "td"
            row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            # 굵기
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            line = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', line)
            line = re.sub(r'`(.+?)`',       r'<code>\1</code>', line)
            line = re.sub(r'^---+$',        r'<hr>', line)
            if line.strip():
                html_lines.append(f"<p>{line}</p>")
            else:
                html_lines.append("<br>")

    if in_table:
        html_lines.append("</table>")

    return "\n".join(html_lines)
