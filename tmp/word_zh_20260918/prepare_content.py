from pathlib import Path
import re, json, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
WORK = Path(__file__).resolve().parent
PANDOC = "D:/Programme/Anaconda/Scripts/pandoc.exe"


def argument(text, start):
    assert text[start] == "{"
    depth, pos = 1, start + 1
    while depth:
        if text[pos] == "{" and text[pos - 1] != "\\": depth += 1
        elif text[pos] == "}" and text[pos - 1] != "\\": depth -= 1
        pos += 1
    return text[start + 1:pos - 1], pos


def expand(text):
    return re.sub(r"\\input\{([^}]+)\}", lambda m: expand((DOCS / m[1]).read_text(encoding="utf-8")), text)


source = (WORK / "manuscript_source.tex").read_text(encoding="utf-8")
source = source.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
source = expand(source).replace(r"\maketitle", "WORDTITLE\n\n")
source = source.replace(r"\method{}", "同伴序列修复与融合").replace(r"\method", "同伴序列修复与融合")
source = source.replace(r"\begin{abstract}", r"\section*{摘要}")
source = source.replace(r"\end{abstract}", "\n\n\\textbf{关键词：} 时序基础模型；历史缺失；同伴序列；岭回归；预测融合\n\n")
source = re.sub(r"\\bibliography\{[^}]+\}", r"\\section*{参考文献}\n\nBIBLIOGRAPHYPLACEHOLDER\n", source)
source = re.sub(r"\\bibliographystyle\{[^}]+\}", "", source)
aux = (DOCS / "iclr2027/tsfm_fais_iclr2027.aux").read_text(encoding="utf-8")
label_numbers = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux))
source = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + label_numbers[m[1]] + ")", source)
source = re.sub(r"\\ref\{([^}]+)\}", lambda m: label_numbers[m[1]], source)

# Keep the manuscript section numbers as ordinary heading text so Word's
# existing template numbering cannot renumber the scientific cross-references.
section, subsection, appendix = 0, 0, False
out, cursor = [], 0
for match in re.finditer(r"\\appendix|\\(subsection|section)(\*)?\{", source):
    out.append(source[cursor:match.start()])
    if match.group(0) == r"\appendix":
        appendix, section, subsection = True, 0, 0
        cursor = match.end()
        continue
    title, end = argument(source, match.end() - 1)
    level, star = match[1], bool(match[2])
    if not star:
        if level == "section": section, subsection = section + 1, 0
        else: subsection += 1
        number = chr(64 + section) if appendix else str(section)
        if level == "subsection": number += "." + str(subsection)
        elif appendix: number = "附录 " + number
        title = number + "　" + title
    out.append("\\" + level + "{" + title + "}")
    cursor = end
out.append(source[cursor:])
source = "".join(out)

source = re.sub(r"\\paragraph\{([^}]+)\}", r"\\textbf{\1}", source)
source = source.replace("拟合目标为\n\\begin{equation}", "取 $\\lambda=10^{-3}$，拟合目标为\n\\begin{equation}", 1)

def convert_float(match):
    env, content = match[1], match[2]
    label = re.search(r"\\label\{([^}]+)\}", content)[1]
    number = label_numbers[label]
    cstart = content.index(r"\caption{") + len(r"\caption")
    caption, cend = argument(content, cstart)
    if env == "figure":
        asset = WORK / "diagram.png" if label == "fig:method" else DOCS / "paper_zh/history_repair_fusion_zh.png"
        return "\n\n\\includegraphics[width=14cm]{" + asset.as_posix() + "}\n\nFIGCAP " + "图 " + number + "　" + caption + "\n\n"
    start = content.index(r"\begin{tabular}")
    end = content.index(r"\end{tabular}") + len(r"\end{tabular}")
    return "\n\nTABLECAP " + "表 " + number + "　" + caption + "\n\n" + content[start:end] + "\n\n"

source = re.sub(r"\\begin\{(figure|table)\}(?:\[[^]]*\])?(.*?)\\end\{\1\}", convert_float, source, flags=re.S)

def convert_equation(match):
    body = match[2]
    lab = re.search(r"\\label\{([^}]+)\}", body)
    number = label_numbers[lab[1]]
    if lab[1] == "eq:ridge":
        body = body.replace(r",\qquad \lambda=10^{-3}.", ".")
    body = re.sub(r"\\label\{[^}]+\}|\\nonumber", "", body).strip()
    # texmath's OMML writer supports aligned arrays and cases directly.
    if match[1] == "align": body = r"\begin{aligned}" + body + r"\end{aligned}"
    return "\n\n\\[" + body + r"\qquad\text{(" + number + r")}\]" + "\n\n"

source = re.sub(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", convert_equation, source, flags=re.S)
source = re.sub(r"\\label\{[^}]+\}|\\clearpage", "", source)
(WORK / "content.tex").write_text(source, encoding="utf-8")
subprocess.run([PANDOC, "-f", "latex", "-t", "json", str(WORK / "content.tex"), "--citeproc",
                "--bibliography", str(DOCS / "iclr2027/current_references.bib"), "-M", "lang=zh-CN",
                "-o", str(WORK / "content.json")], check=True, cwd=DOCS)
data = json.loads((WORK / "content.json").read_text(encoding="utf-8"))

def plain(value):
    if isinstance(value, dict):
        if value.get("t") == "Str": return value["c"]
        return plain(value.get("c", []))
    if isinstance(value, list): return "".join(plain(x) for x in value)
    return ""

references = [b for b in data["blocks"] if b["t"] == "Div" and b["c"][0][0] == "refs"]
assert len(references) == 1
blocks = []
for block in data["blocks"]:
    if block is references[0]: continue
    if plain(block) == "BIBLIOGRAPHYPLACEHOLDER": blocks.append(references[0])
    else: blocks.append(block)
data["blocks"] = blocks
(WORK / "content.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"blocks": len(blocks), "tables": sum(b["t"] == "Table" for b in blocks),
                  "headings": sum(b["t"] == "Header" for b in blocks), "reference_entries": len(references[0]["c"][1])}))
