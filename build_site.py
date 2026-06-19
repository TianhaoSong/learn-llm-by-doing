import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COURSES = ROOT / "courses"
MD = COURSES / "course-outlines.md"
PRIMERS = COURSES / "primers"   # 手写的交互式 primer（HTML 源），build 时原样拷进 site/
SITE = ROOT / "site"
SITE.mkdir(exist_ok=True)
INDEX_HTML = SITE / "index.html"

MODULES = [
    ("a", "A-M0", "m0-pytorch", "PyTorch 与训练循环基础"),
    ("a", "A-M1", "m1-mygpt", "Transformer 与 nanoGPT 单卡复现"),
    ("a", "A-M2", "m2-ddp", "单机多卡 DDP + Mixed Precision"),
    ("a", "A-M3", "m3-fsdp", "显存账本与 ZeRO/FSDP 渐进升级"),
    ("a", "A-M4", "m4-large-run", "100M–1B 多卡训练实跑（线 1 终点）"),
    ("b", "B-M0", "m0-baseline", "推理基础与 baseline"),
    ("b", "B-M1", "m1-engine", "手写 KV cache + continuous batching"),
    ("b", "B-M2", "m2-paged", "手写 PagedAttention"),
    ("b", "B-M3", "m3-tp", "Tensor parallel + 多机部署"),
    ("b", "B-M4", "m4-compare", "与真实 vLLM/SGLang 对比（线 2 终点）"),
    ("c", "C-M0", "m0-mini-agent", "Agent 基础与工具使用"),
    ("c", "C-M1", "m1-architecture", "Multi-agent 架构决策（核心）"),
    ("c", "C-M2", "m2-eval", "Agent eval 体系"),
    ("c", "C-M3", "m3-comparison", "Single-agent vs Multi-agent 对照实验"),
    ("c", "C-M4", "m4-slack-tools", "Slack agent 工具设计（辅助）"),
]
COURSE_META = {
    "a": ("课程 A · 5 个模块 · 串行", "手写训练",
          "从 nanoGPT 单卡到 100M–1B 多卡训练；脱口讲清 DDP/FSDP/ZeRO 取舍、NCCL 通信模式、显存账本、gradient accumulation 与 micro-batch 的 trade-off，并有自己的 profiling 数据。"),
    "b": ("课程 B · 5 个模块 · 前置 A-M1", "简化版 vLLM",
          "从 naive batched inference 到手写 PagedAttention + tensor parallel；每个阶段都有 throughput/latency benchmark；脱口讲清 KV cache 设计、continuous batching 调度策略、prefill vs decode 的资源差异，并能与真实 vLLM 对比说出差距与原因。"),
    "c": ("课程 C · 5 个模块 · 与 A/B 并行", "Agent",
          "multi-agent 架构决策三件事——什么时候该拆 agent / handoff 怎么设计 / 怎么 eval；并有一个 single-agent vs multi-agent 实验数字作为简历版佐证。"),
}

md_text = MD.read_text()

def get_module_section(code):
    pattern = rf"### {re.escape(code)}[\s\S]*?(?=\n### [A-CB]-M\d|\n## |\Z)"
    m = re.search(pattern, md_text)
    return m.group(0) if m else ""

def parse_module(code):
    section = get_module_section(code)
    topic_m = re.search(r"\*\*Topic\*\*：(.+?)(?=\n\n)", section, re.DOTALL)
    topic = topic_m.group(1).strip() if topic_m else ""
    topic_preview = topic.split("——")[0].split("。")[0][:60]
    if len(topic) > 60: topic_preview = topic_preview + "…"
    
    obj_m = re.search(r"1\. \*\*学习目标\*\*\n((?:   - .+\n)+)", section)
    objectives = []
    if obj_m:
        for line in obj_m.group(1).strip().split("\n"):
            objectives.append(line.strip("- ").strip())
    
    setup_m = re.search(r"0\. \*\*环境准备\*\*\n\n(.+?)(?=\n1\. \*\*学习目标\*\*)", section, re.DOTALL)
    setup_md = setup_m.group(1).strip() if setup_m else ""
    
    # Capture the whole readings block: from the "2. **学习材料...**" header line
    # (which may carry trailing text like "— 按顺序：") up to the next section
    # ("3. **") or a "   >" note line or end-of-section.
    readings_m = re.search(r"2\. \*\*学习材料[^\n]*\n(.+?)(?=\n3\. \*\*|\n   >|\Z)", section, re.DOTALL)
    readings = []
    if readings_m:
        # A new entry begins with "   - " (or "- "). Lines that are indented but
        # do NOT start with a dash are continuation/description lines and get
        # folded into the previous entry's note (this is the multi-line
        # "self-study syllabus" format: title+url on one line, why-to-read below).
        for raw in readings_m.group(1).split("\n"):
            if not raw.strip():
                continue
            is_item = re.match(r"\s*-\s+", raw)
            if is_item:
                line = re.sub(r"^\s*-\s+", "", raw).strip()
                # http(s) link, or a local relative page (e.g. nccl-primer.html#anchor).
                # URL stops at whitespace or a fullwidth paren so a trailing CJK
                # annotation like "（只读 §2-3）" isn't swallowed into the href.
                url_m = re.match(r"(.+?):\s*(https?://[^\s（）]+|[\w./-]+\.html(?:#[\w-]+)?)\s*(.*)", line)
                if url_m:
                    readings.append([url_m.group(1).strip(), url_m.group(2).strip(), url_m.group(3).strip()])
                else:
                    readings.append([line, None, ""])
            elif readings:
                # continuation line → append to the current entry's note
                cont = raw.strip()
                readings[-1][2] = (readings[-1][2] + " " + cont).strip()
        readings = [tuple(r) for r in readings]
    
    return {"topic": topic, "topic_preview": topic_preview, "objectives": objectives,
            "setup_md": setup_md, "readings": readings}

def md_inline(text):
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r'<a href="\2">\1</a>', text)
    return text

def render_setup(setup_md):
    if not setup_md: return "<p>(setup pending)</p>"
    out = []
    parts = re.split(r"\*\*(快速 setup|值得理解)\*\*[^\n]*\n", setup_md)
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i+1] if i+1 < len(parts) else ""
        out.append(f'<h4>{header}</h4>')
        out.append("<ul>")
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                out.append(f"<li>{md_inline(line[2:])}</li>")
        out.append("</ul>")
    return "\n".join(out) if out else "<p>(setup pending)</p>"

def render_readings(readings):
    out = ['<ol class="reading-list">']
    for text, url, *rest in readings:
        note = rest[0] if rest else ""
        note_html = f' <span class="reading-note">{md_inline(note)}</span>' if note else ""
        if url:
            out.append(f'<li><a href="{url}" target="_blank" rel="noopener">{md_inline(text)}</a>{note_html}</li>')
        else:
            out.append(f'<li>{md_inline(text)}{note_html}</li>')
    out.append('</ol>')
    return "\n".join(out)

CSS_BASE = '''  :root {
    --bg: #ffffff; --bg-soft: #fafafa; --fg: #111111; --fg-2: #2a2a2a;
    --muted: #6a6a6a; --border: #e2e2e2; --border-strong: #111111; --accent: #c0392b;
    --sans: 'IBM Plex Sans', -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0e0e0e; --bg-soft: #161616; --fg: #e8e8e8; --fg-2: #c4c4c4;
      --muted: #888888; --border: #2a2a2a; --border-strong: #e8e8e8; --accent: #e57367; }
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; font-family: var(--sans); background: var(--bg); color: var(--fg); line-height: 1.7; font-size: 15px; -webkit-font-smoothing: antialiased; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  code { font-family: var(--mono); background: var(--bg-soft); padding: 1px 5px; border-radius: 2px; font-size: 0.9em; }
  ::selection { background: var(--accent); color: var(--bg); }'''

# ============= INDEX PAGE =============

INDEX_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>learn-llm-by-doing · 课程索引</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
__CSS_BASE__
  .layout { display: grid; grid-template-columns: 240px 1fr; max-width: 1180px; margin: 0 auto; }
  .toc { position: sticky; top: 0; align-self: start; height: 100vh; overflow-y: auto; padding: 32px 20px 32px 28px; border-right: 1px solid var(--border); font-size: 13px; }
  .toc-title { font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
  .toc-section { margin-bottom: 20px; }
  .toc-course { font-size: 13px; font-weight: 600; color: var(--fg); margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
  .toc-course .num { font-family: var(--mono); color: var(--muted); margin-right: 6px; font-weight: 500; }
  .toc-list { list-style: none; padding: 0; margin: 0; }
  .toc-list a { display: grid; grid-template-columns: 38px 1fr; gap: 4px; padding: 3px 0; color: var(--fg-2); text-decoration: none; font-size: 13px; align-items: baseline; }
  .toc-list a > span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .toc-list a:hover { color: var(--accent); }
  .toc-list .code { font-family: var(--mono); color: var(--muted); font-size: 11px; }
  .toc-list a:hover .code { color: var(--accent); }
  main { padding: 40px 40px 60px; min-width: 0; max-width: 820px; }
  h1.page-title { font-size: 28px; font-weight: 600; margin: 0 0 6px 0; letter-spacing: -0.01em; }
  .subtitle { font-size: 14px; color: var(--muted); margin-bottom: 24px; }
  .preface { margin-bottom: 32px; padding: 16px 18px; background: var(--bg-soft); border-left: 2px solid var(--accent); font-size: 13.5px; line-height: 1.7; color: var(--fg-2); }
  .preface p { margin: 6px 0; }
  .preface strong { color: var(--fg); }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } .toc { display: none; } main { padding: 28px 20px 48px; } }
  .course { margin-top: 48px; }
  .course-head { border-top: 2px solid var(--border-strong); padding-top: 14px; margin-bottom: 20px; }
  .course-head .label { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.05em; margin-bottom: 4px; }
  .course-head h2 { font-size: 22px; font-weight: 600; margin: 0 0 8px 0; letter-spacing: -0.01em; }
  .course-head .outcome { font-size: 14px; color: var(--fg-2); line-height: 1.6; margin: 0; }
  .course-head .outcome strong { color: var(--accent); font-weight: 500; }
  .module-card { margin: 18px 0; padding: 18px 20px; border: 1px solid var(--border); transition: border-color 0.15s; }
  .module-card:hover { border-color: var(--border-strong); }
  .module-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }
  .module-head .code { font-family: var(--mono); font-size: 12px; color: var(--accent); font-weight: 500; letter-spacing: 0.03em; }
  .module-head h3 { font-size: 17px; font-weight: 600; margin: 0; letter-spacing: -0.005em; }
  .topic { padding: 10px 14px; background: var(--bg-soft); border-left: 2px solid var(--accent); font-size: 13px; line-height: 1.6; color: var(--fg-2); margin: 10px 0 14px 0; }
  .topic .label { font-family: var(--mono); font-size: 10px; color: var(--accent); letter-spacing: 0.06em; font-weight: 500; margin-right: 6px; }
  .objectives { margin: 14px 0; }
  .objectives .label { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.05em; margin-bottom: 6px; }
  .objectives ul { margin: 0; padding-left: 22px; font-size: 14px; line-height: 1.7; }
  .objectives li { margin-bottom: 3px; }
  .readings-section { margin: 14px 0; }
  .readings-section .label { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.05em; margin-bottom: 6px; }
  .reading-list { list-style: none; padding: 0; counter-reset: rd; margin: 6px 0 0 0; }
  .reading-list li { padding: 5px 0 5px 30px; counter-increment: rd; position: relative; font-size: 13.5px; line-height: 1.55; }
  .reading-list li::before { content: counter(rd, decimal-leading-zero); position: absolute; left: 0; top: 5px; font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .reading-list a { color: var(--accent); border-bottom: 1px solid currentColor; word-break: break-word; }
  .reading-list a:hover { background: var(--accent); color: var(--bg); text-decoration: none; }
  .reading-note { color: var(--muted); font-size: 12px; }
  .start-doing { display: inline-block; margin-top: 14px; padding: 8px 14px; border: 1px solid var(--accent); color: var(--accent); font-family: var(--mono); font-size: 12px; letter-spacing: 0.04em; transition: background 0.15s, color 0.15s; }
  .start-doing:hover { background: var(--accent); color: var(--bg); text-decoration: none; }
  .appendix { margin-top: 64px; padding-top: 24px; border-top: 2px solid var(--border-strong); }
  .appendix h2 { font-size: 22px; font-weight: 600; margin: 0 0 16px 0; }
  .appendix .label { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.05em; margin-bottom: 6px; }
  .alignment { width: 100%; border-collapse: collapse; font-size: 13px; }
  .alignment th, .alignment td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .alignment th { font-family: var(--mono); font-size: 11px; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border-strong); }
  .alignment .modules { font-family: var(--mono); font-size: 12px; color: var(--accent); white-space: nowrap; }
  footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); font-family: var(--mono); }
</style>
</head>
<body>

<div class="layout">
  <nav class="toc">
    <div class="toc-title">目录</div>
__TOC__
  </nav>

  <main>
    <h1 class="page-title">learn-llm-by-doing — 课程索引</h1>
    <div class="subtitle">三条工作线 · 15 个模块 · 学习与执行分页</div>

    <div class="preface">
      <p>本页是<strong>学习入口</strong>——只列每模块的 Topic / 学习目标 / canonical 学习材料；先看完材料、想清楚为什么做。</p>
      <p>每个模块底部 "<strong>→ 开始做</strong>" 跳到该模块的<strong>执行页</strong>，含环境准备 + Doing 任务（DOING.md 全文）+ 过关标准（QUIZ.md 全文）。学习与做分开。</p>
      <p>Out-of-Scope：chatbot/RAG 上层应用、深度数学推导、大段博客或文档撰写、死磕收敛/调参。</p>
    </div>

__COURSES__

    <section class="appendix" id="alignment">
      <div class="label">附录</div>
      <h2>对齐核对表</h2>
      <table class="alignment">
        <thead><tr><th style="width: 46%;">成功标准（来自 requirements.md）</th><th style="width: 22%;">覆盖模块</th><th>备注</th></tr></thead>
        <tbody>
__ALIGNMENT__
        </tbody>
      </table>
    </section>

    <footer>learn-llm-by-doing · 学习与执行分页</footer>
  </main>
</div>

</body>
</html>
'''

# Build per-course sections for index
course_sections = []
for course in ("a", "b", "c"):
    label, name, outcome = COURSE_META[course]
    cards = []
    for c, code, mod_dir, title in MODULES:
        if c != course: continue
        parsed = parse_module(code)
        obj_html = "\n".join(f'      <li>{md_inline(o)}</li>' for o in parsed["objectives"])
        readings_html = render_readings(parsed["readings"])
        page = f"{code.lower()}.html"
        cards.append(f'''  <div class="module-card" id="{code.lower()}">
    <div class="module-head">
      <span class="code">{code.replace('-', '·')}</span>
      <h3>{title}</h3>
    </div>
    <div class="topic"><span class="label">TOPIC</span>{md_inline(parsed["topic"])}</div>
    <div class="objectives">
      <div class="label">学习目标</div>
      <ul>
{obj_html}
      </ul>
    </div>
    <div class="readings-section">
      <div class="label">学习材料 (canonical)</div>
{readings_html}
    </div>
    <a href="{page}" class="start-doing">→ 开始做（环境 / 任务 / 过关）</a>
  </div>''')
    
    course_sections.append(f'''  <section class="course" id="course-{course}">
    <div class="course-head">
      <div class="label">{label}</div>
      <h2>{name}</h2>
      <p class="outcome"><strong>学完后能讲清楚</strong>：{outcome}</p>
    </div>
{chr(10).join(cards)}
  </section>''')

alignment_rows = [
    ("多 GPU 训练：DDP / FSDP / ZeRO 取舍", "A·M2 · A·M3 · A·M4", "M2 给出 DDP 实测，M3 渐进升 FSDP 并对比 ZeRO 概念，M4 在多卡上跑通"),
    ("通信模式（all-reduce / reduce-scatter / all-gather）", "A·M2 · A·M3 · B·M3", "A·M2 单独 benchmark；A·M3 在 FSDP 中讲 forward all-gather + backward reduce-scatter；B·M3 推理侧 TP all-reduce"),
    ("显存占用计算", "A·M3", "mem_calc 脚本 + 实测验证"),
    ("gradient accumulation 与 micro-batch", "A·M2", "含 <code>no_sync</code> 用法与 step time 对比"),
    ("KV cache 设计", "A·M1 · B·M1 · B·M2", "A·M1 直觉，B·M1 连续 KV cache 实现，B·M2 paged 实现"),
    ("为什么需要 PagedAttention", "B·M2", "含 fragmentation / over-provisioning 解释 + 自实现 + 显存对比"),
    ("Continuous batching 调度策略", "B·M1", "自写 scheduler + 与 static batching 对比"),
    ("Prefill vs decode 资源差异", "B·M0 · B·M1", "B·M0 baseline 给出 TTFT/TPOT 区别，B·M1 调度区分两类请求"),
    ("Multi-agent：什么时候该拆 agent", "C·M1", "拆分判定标准 + 项目决策记录"),
    ("Multi-agent：handoff 怎么设计", "C·M1", "5 个 handoff 场景 + 结构化 schema 评审"),
    ("Multi-agent：怎么 eval", "C·M2 · C·M3", "C·M2 搭 harness，C·M3 single vs multi 对照实验"),
]
alignment_html = "\n".join(
    f'        <tr><td>{a}</td><td class="modules">{b}</td><td style="color: var(--muted); font-size: 12px;">{c}</td></tr>'
    for a, b, c in alignment_rows
)

# Build TOC HTML — 人工短名，避免按字符硬截断切出断头残词
TOC_SHORT = {
    "A-M0": "PyTorch 基础",
    "A-M1": "Transformer / nanoGPT",
    "A-M2": "DDP 多卡",
    "A-M3": "FSDP / ZeRO 显存",
    "A-M4": "1B 多卡实跑",
    "B-M0": "推理 baseline",
    "B-M1": "KV cache + CB",
    "B-M2": "PagedAttention",
    "B-M3": "Tensor Parallel",
    "B-M4": "对比 vLLM",
    "C-M0": "Agent 基础",
    "C-M1": "Multi-agent 架构",
    "C-M2": "Agent eval",
    "C-M3": "single vs multi",
    "C-M4": "工具设计",
}

def short_title(title, code=None):
    if code and code in TOC_SHORT:
        return TOC_SHORT[code]
    return title.split("（")[0]

toc_parts = []
for course in ("a", "b", "c"):
    course_label = {"a": "手写训练", "b": "简化版 vLLM", "c": "Agent"}[course]
    items = []
    for c, code, mod_dir, title in MODULES:
        if c != course: continue
        items.append(f'        <li><a href="#{code.lower()}"><span class="code">{code.replace("-", "·")}</span><span>{short_title(title, code)}</span></a></li>')
    toc_parts.append(f'''    <div class="toc-section">
      <div class="toc-course"><span class="num">{course.upper()}</span>{course_label}</div>
      <ul class="toc-list">
{chr(10).join(items)}
      </ul>
    </div>''')
toc_parts.append('''    <div class="toc-section">
      <div class="toc-course"><span class="num">§</span>附录</div>
      <ul class="toc-list">
        <li><a href="#alignment"><span class="code">附录</span><span>对齐核对表</span></a></li>
      </ul>
    </div>''')
toc_html = "\n".join(toc_parts)

index_html = INDEX_TEMPLATE.replace("__CSS_BASE__", CSS_BASE) \
                           .replace("__TOC__", toc_html) \
                           .replace("__COURSES__", "\n".join(course_sections)) \
                           .replace("__ALIGNMENT__", alignment_html)

INDEX_HTML.write_text(index_html)
print(f"Wrote {INDEX_HTML} ({len(index_html)} bytes)")

# ============= MODULE PAGES =============

MODULE_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__CODE__ · __TITLE__ — learn-llm-by-doing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
__CSS_BASE__
  .layout { max-width: 880px; margin: 0 auto; padding: 32px 32px 60px; }
  .back { font-family: var(--mono); font-size: 12px; color: var(--muted); display: inline-block; margin-bottom: 16px; padding: 4px 8px; border: 1px solid var(--border); }
  .back:hover { color: var(--accent); border-color: var(--accent); text-decoration: none; }
  .module-head { display: flex; align-items: baseline; gap: 12px; padding-bottom: 10px; border-bottom: 2px solid var(--border-strong); margin-bottom: 16px; flex-wrap: wrap; }
  .module-head .code { font-family: var(--mono); font-size: 13px; color: var(--accent); font-weight: 500; letter-spacing: 0.03em; }
  .module-head h1 { font-size: 24px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
  .topic { padding: 12px 14px; background: var(--bg-soft); border-left: 2px solid var(--accent); font-size: 13px; line-height: 1.65; color: var(--fg-2); margin: 0 0 24px 0; }
  .topic .label { font-family: var(--mono); font-size: 10px; color: var(--accent); letter-spacing: 0.06em; font-weight: 500; margin-right: 6px; }
  .section { margin: 32px 0; }
  .section-head { display: flex; align-items: baseline; gap: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border-strong); margin-bottom: 16px; }
  .section-head .num { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.04em; }
  .section-head h2 { font-size: 17px; font-weight: 600; margin: 0; letter-spacing: -0.005em; }
  .setup h4 { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.05em; margin: 14px 0 6px 0; font-weight: 500; }
  .setup h4:first-child { margin-top: 0; }
  .setup ul { margin: 0; padding-left: 22px; font-size: 14px; line-height: 1.7; }
  .setup li { margin-bottom: 4px; }
  .nav-pills { display: flex; gap: 8px; padding: 8px 0; margin-bottom: 16px; position: sticky; top: 0; background: var(--bg); z-index: 10; border-bottom: 1px solid var(--border); }
  .nav-pills a { font-family: var(--mono); font-size: 11px; padding: 6px 10px; border: 1px solid var(--border); color: var(--muted); letter-spacing: 0.04em; }
  .nav-pills a:hover { color: var(--accent); border-color: var(--accent); text-decoration: none; }

  /* Markdown rendered */
  .md-rendered { font-size: 14px; line-height: 1.7; color: var(--fg-2); }
  .md-rendered h1 { display: none; }
  .md-rendered h2 { font-size: 16px; font-weight: 600; color: var(--fg); margin: 22px 0 10px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border-strong); letter-spacing: -0.005em; }
  .md-rendered h3 { font-size: 14px; font-weight: 600; color: var(--fg); margin: 16px 0 8px 0; font-family: var(--mono); letter-spacing: 0.02em; }
  .md-rendered h4 { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.05em; margin: 12px 0 4px 0; font-weight: 500; }
  .md-rendered p { margin: 8px 0; }
  .md-rendered ul, .md-rendered ol { margin: 6px 0 10px 0; padding-left: 24px; }
  .md-rendered li { margin-bottom: 4px; }
  .md-rendered ul ul, .md-rendered ol ol { margin: 4px 0; }
  .md-rendered blockquote { margin: 10px 0; padding: 10px 14px; border-left: 2px solid var(--muted); background: var(--bg-soft); color: var(--fg-2); font-size: 13.5px; }
  .md-rendered blockquote p { margin: 4px 0; }
  .md-rendered code { font-family: var(--mono); background: var(--bg-soft); padding: 1px 5px; border-radius: 2px; font-size: 0.88em; color: var(--fg); }
  .md-rendered pre { background: var(--bg-soft); border: 1px solid var(--border); padding: 12px 14px; overflow-x: auto; font-size: 12.5px; line-height: 1.55; margin: 10px 0; }
  .md-rendered pre code { background: transparent; padding: 0; font-size: 12.5px; }
  .md-rendered a { color: var(--accent); border-bottom: 1px solid currentColor; word-break: break-word; }
  .md-rendered a:hover { background: var(--accent); color: var(--bg); text-decoration: none; }
  .md-rendered strong { color: var(--fg); font-weight: 600; }
  .md-rendered table { border-collapse: collapse; margin: 10px 0; font-size: 13px; width: 100%; }
  .md-rendered th, .md-rendered td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; vertical-align: top; }
  .md-rendered th { background: var(--bg-soft); font-family: var(--mono); font-size: 11px; color: var(--muted); font-weight: 500; }
  .md-rendered hr { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
  footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); font-family: var(--mono); }
</style>
</head>
<body>

<div class="layout">
  <a href="index.html#__CODE_LOWER__" class="back">← 回索引</a>
  
  <div class="module-head">
    <span class="code">__CODE_DOT__</span>
    <h1>__TITLE__</h1>
  </div>
  <div class="topic"><span class="label">TOPIC</span>__TOPIC__</div>

  <nav class="nav-pills">
    <a href="#setup">0 · 环境准备</a>
    <a href="#doing">1 · Doing 任务</a>
    <a href="#quiz">2 · 过关 Quiz</a>
  </nav>

  <section class="section" id="setup">
    <div class="section-head"><span class="num">0</span><h2>环境准备</h2></div>
    <div class="setup">
__SETUP__
    </div>
  </section>

  <section class="section" id="doing">
    <div class="section-head"><span class="num">1</span><h2>Doing 任务（DOING.md）</h2></div>
    <div class="md-rendered" id="doing-content">Loading…</div>
  </section>

  <section class="section" id="quiz">
    <div class="section-head"><span class="num">2</span><h2>过关 Quiz（QUIZ.md）</h2></div>
    <div class="md-rendered" id="quiz-content">Loading…</div>
  </section>

  <footer>__CODE__ · 执行页</footer>
</div>

<script type="text/markdown" id="md-doing">__DOING_MD__</script>
<script type="text/markdown" id="md-quiz">__QUIZ_MD__</script>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
  if (window.marked) {
    marked.setOptions({ breaks: false, gfm: true });
    document.getElementById('doing-content').innerHTML =
      marked.parse(document.getElementById('md-doing').textContent);
    document.getElementById('quiz-content').innerHTML =
      marked.parse(document.getElementById('md-quiz').textContent);
  }
</script>

</body>
</html>
'''

# Generate 15 module pages
for course, code, mod_dir, title in MODULES:
    parsed = parse_module(code)
    setup_html = render_setup(parsed["setup_md"])
    
    spec_dir = COURSES / code.lower()  # courses/a-m0, courses/b-m1, etc.
    doing_md = (spec_dir / "DOING.md").read_text()
    quiz_md = (spec_dir / "QUIZ.md").read_text()
    # Escape for embedding in <script type="text/markdown">
    doing_md = doing_md.replace("</script>", "<\\/script>")
    quiz_md = quiz_md.replace("</script>", "<\\/script>")
    
    page_html = MODULE_TEMPLATE.replace("__CSS_BASE__", CSS_BASE) \
                               .replace("__CODE__", code) \
                               .replace("__CODE_DOT__", code.replace("-", "·")) \
                               .replace("__CODE_LOWER__", code.lower()) \
                               .replace("__TITLE__", title) \
                               .replace("__TOPIC__", md_inline(parsed["topic"])) \
                               .replace("__SETUP__", setup_html) \
                               .replace("__DOING_MD__", doing_md) \
                               .replace("__QUIZ_MD__", quiz_md)
    
    out_path = SITE / f"{code.lower()}.html"
    out_path.write_text(page_html)
    print(f"  wrote {out_path.name} ({len(page_html)} bytes)")

# ============= PRIMERS (copy手写交互页) =============
# courses/primers/*.html 是手写的交互式 primer，build 时原样拷进 site/。
# 这样 site/ 100% 是产物（可随时清空重建），所有源都在 courses/ 下。
if PRIMERS.is_dir():
    for p in sorted(PRIMERS.glob("*.html")):
        shutil.copy2(p, SITE / p.name)
        print(f"  copied primer {p.name} ({p.stat().st_size} bytes)")

print("Done.")
