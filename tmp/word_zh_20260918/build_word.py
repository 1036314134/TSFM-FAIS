from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from collections import Counter
from lxml import etree as E
import hashlib, json, re, sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
REFERENCE = ROOT / "docs/3-B-FAIS-面向时间序列基础模型的预测感知块级填补算法选择-0828.docx"
CONTENT = WORK / "content.docx"
OUTPUT = ROOT / "docs/TSFM-FAIS-同伴序列修复与预测融合-中文版-0918.docx"
EXPECTED = "6731cee5a0b433513f176613634f1198dc4504c41bd95023aade3e85649274b7"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "m": M, "r": R}


def tag(name): return "{" + W + "}" + name
def val(node, name, default=None): return node.get(tag(name), default)
def text(node): return "".join(node.xpath(".//w:t/text()", namespaces=NS))
def xml(node): return E.tostring(node, encoding="UTF-8", xml_declaration=True, standalone=True)


assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED
with ZipFile(REFERENCE) as z: parts = {n: z.read(n) for n in z.namelist()}
with ZipFile(CONTENT) as z: incoming = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts["word/document.xml"])
body = doc.find("w:body", NS)
templates = list(body)
source_section = deepcopy(templates[-1])
incoming_doc = E.fromstring(incoming["word/document.xml"])
incoming_body = incoming_doc.find("w:body", NS)
style_ids = {val(s, "styleId") for s in E.fromstring(parts["word/styles.xml"]).findall("w:style", NS)}
roles = Counter()


def replace_ppr(paragraph, template_index):
    for current in paragraph.findall("w:pPr", NS): paragraph.remove(current)
    props = templates[template_index].find("w:pPr", NS)
    if props is not None: paragraph.insert(0, deepcopy(props))


def run_properties(run, size=None, bold=None):
    props = run.find("w:rPr", NS)
    if props is None:
        props = E.Element(tag("rPr")); run.insert(0, props)
    # Fonts and point sizes come from the retained source styles. Retain only
    # semantic emphasis and hyperlink styling from the content conversion.
    for child in list(props):
        local = E.QName(child).localname
        if local in {"rFonts", "sz", "szCs", "color", "spacing", "position", "kern", "lang", "highlight"}:
            props.remove(child)
        elif local == "rStyle":
            identifier = val(child, "val")
            if identifier in {"Hyperlink", "a3"}: child.set(tag("val"), "a3")
            elif identifier not in style_ids: props.remove(child)
    if size is not None:
        for local in ["sz", "szCs"]:
            E.SubElement(props, tag(local)).set(tag("val"), str(size))
    if bold is not None:
        for local in ["b", "bCs"]:
            for previous in props.findall("w:" + local, NS): props.remove(previous)
            if bold: E.SubElement(props, tag(local))


def normalize_runs(paragraph, size=None, bold=None):
    for run in paragraph.findall(".//w:r", NS): run_properties(run, size=size, bold=bold)
    for run in paragraph.findall(".//m:r", NS):
        props = run.find("w:rPr", NS)
        if props is None:
            props = E.Element(tag("rPr"))
            run.insert(1 if run.find("m:rPr", NS) is not None else 0, props)
        for local in ["sz", "szCs", "rFonts"]:
            for old in props.findall("w:" + local, NS): props.remove(old)
        fonts = E.SubElement(props, tag("rFonts"))
        for name in ["ascii", "hAnsi", "cs"]: fonts.set(tag(name), "Cambria Math")
        for name in ["sz", "szCs"]: E.SubElement(props, tag(name)).set(tag("val"), str(size or 24))


def remove_prefix(paragraph, prefix):
    assert text(paragraph).startswith(prefix)
    remaining = len(prefix)
    for node in paragraph.findall(".//w:t", NS):
        current = node.text or ""
        take = min(remaining, len(current))
        node.text = current[take:]
        remaining -= take
        if not remaining: break
    assert not remaining


def new_text_paragraph(template_index, value, size=None, bold=None):
    p = E.Element(tag("p")); replace_ppr(p, template_index)
    r = E.SubElement(p, tag("r")); run_properties(r, size=size, bold=bold)
    t = E.SubElement(r, tag("t")); t.text = value
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


def normalize_table(table, index):
    count = len(table.find("w:tr", NS).findall("w:tc", NS))
    template_index = 52 if count == 7 else (119 if count == 2 else 153)
    prototype = templates[template_index]
    size = {7: 18, 5: 19, 2: 21}[count]
    if count == 7:
        widths = [2706, 934, 933, 934, 933, 933, 933]
    elif count == 2: widths = [650, 7656]
    elif index == 5: widths = [2000, 1300, 1300, 1853, 1853]
    else: widths = [1100, 2600, 1500, 1200, 1906]
    assert sum(widths) == 8306
    for local in ["tblPr", "tblGrid"]:
        for old in table.findall("w:" + local, NS): table.remove(old)
    table.insert(0, deepcopy(prototype.find("w:tblPr", NS)))
    grid = E.Element(tag("tblGrid"))
    for width in widths: E.SubElement(grid, tag("gridCol")).set(tag("w"), str(width))
    table.insert(1, grid)
    rows = table.findall("w:tr", NS)
    if count == 7:
        # Use the reference's one-row grid header, explicitly retaining every
        # panel/horizon label from the source's two-row header.
        table.remove(rows[0]); rows = rows[1:]
        scopes = ["自然缺测 H24", "人工缺测 H24", "原始缺失 H96"] if index in [2, 4] else ["自然缺口", "人工缺测", "原生网格"]
        headers = ["方法"] + [scope + "\n" + metric for scope in scopes for metric in ["MAE", "MSE"]]
        for cell, value in zip(rows[0].findall("w:tc", NS), headers):
            for child in list(cell):
                if child.tag != tag("tcPr"): cell.remove(child)
            p = E.SubElement(cell, tag("p")); r = E.SubElement(p, tag("r"))
            for j, line in enumerate(value.split("\n")):
                if j: E.SubElement(r, tag("br"))
                E.SubElement(r, tag("t")).text = line
    for row_index, row in enumerate(rows):
        ref_row = prototype.findall("w:tr", NS)[0 if row_index == 0 else 1]
        for old in row.findall("w:trPr", NS): row.remove(old)
        row.insert(0, deepcopy(ref_row.find("w:trPr", NS)))
        for column, cell in enumerate(row.findall("w:tc", NS)):
            ref_cell = ref_row.findall("w:tc", NS)[min(column, len(ref_row.findall("w:tc", NS)) - 1)]
            for old in cell.findall("w:tcPr", NS): cell.remove(old)
            props = deepcopy(ref_cell.find("w:tcPr", NS))
            props.find("w:tcW", NS).set(tag("w"), str(widths[column]))
            cell.insert(0, props)
            for p in cell.findall("w:p", NS):
                for old in p.findall("w:pPr", NS): p.remove(old)
                ppr = deepcopy(ref_cell.find("w:p/w:pPr", NS))
                ppr.find("w:jc", NS).set(tag("val"), "center" if row_index == 0 or (column > 0 and count != 2) else "left")
                p.insert(0, ppr)
                normalize_runs(p, size=size, bold=True if row_index == 0 else None)
    return table


new_children = []
in_references = False
table_index = 0
for item in incoming_body:
    item = deepcopy(item)
    if item.tag == tag("sectPr"): continue
    if item.tag == tag("tbl"):
        new_children.append(normalize_table(item, table_index)); table_index += 1
        continue
    if item.tag != tag("p"):
        new_children.append(item); continue
    value = text(item)
    style = item.find("w:pPr/w:pStyle", NS)
    style_id = val(style, "val") if style is not None else ""
    if value == "WORDTITLE":
        item = new_text_paragraph(0, "面向传感器历史缺失的同伴序列修复与预测融合", size=32)
        roles["title"] += 1
    elif style_id in {"1", "Heading1", "2", "Heading2"}:
        if value == "参考文献":
            in_references = True; replace_ppr(item, 81)
        elif value.startswith("附录"):
            in_references = False
            replace_ppr(item, 115 if value.startswith("附录 A") else 132)
        else: replace_ppr(item, 4 if style_id in {"1", "Heading1"} else 11)
        normalize_runs(item)
        roles["heading"] += 1
    elif value.startswith("FIGCAP "):
        remove_prefix(item, "FIGCAP "); replace_ppr(item, 38)
        normalize_runs(item, size=21); roles["figure_caption"] += 1
    elif value.startswith("TABLECAP "):
        remove_prefix(item, "TABLECAP "); replace_ppr(item, 51)
        normalize_runs(item, size=21); roles["table_caption"] += 1
    elif item.find(".//w:drawing", NS) is not None:
        replace_ppr(item, 37); normalize_runs(item); roles["figure"] += 1
    elif item.find("m:oMathPara", NS) is not None:
        replace_ppr(item, 19); normalize_runs(item); roles["display_equation"] += 1
    elif in_references:
        replace_ppr(item, 82); normalize_runs(item); roles["reference"] += 1
    else:
        replace_ppr(item, 2); normalize_runs(item); roles["body"] += 1
    new_children.append(item)

for child in list(body): body.remove(child)
for child in new_children: body.append(child)
body.append(source_section)

# Import only relationships referenced by the new body into a copy of the
# retained package. Existing template relationships and package parts survive.
relations = E.fromstring(parts["word/_rels/document.xml.rels"])
incoming_relations = {x.get("Id"): x for x in E.fromstring(incoming["word/_rels/document.xml.rels"])}
used_ids = {attribute for node in body.iter() for key, attribute in node.attrib.items() if key in {"{" + R + "}id", "{" + R + "}embed", "{" + R + "}link"}}
existing_ids = {r.get("Id") for r in relations}
relationship_map = {}
added_media = []
for source_id in sorted(used_ids):
    if source_id not in incoming_relations: continue
    relationship = incoming_relations[source_id]
    # sectPr references must keep their original template bindings.
    if relationship.get("Type", "").endswith(("/header", "/footer")): continue
    candidate = "rCurrent" + str(len(relationship_map) + 1)
    assert candidate not in existing_ids
    copied = deepcopy(relationship); copied.set("Id", candidate)
    if copied.get("Type", "").endswith("/image"):
        target = copied.get("Target"); source_path = "word/" + target
        extension = Path(target).suffix.lower()
        destination = f"word/media/current_figure_{len(added_media) + 1}{extension}"
        assert destination not in parts
        parts[destination] = incoming[source_path]
        copied.set("Target", destination.removeprefix("word/")); added_media.append(destination)
    relations.append(copied); relationship_map[source_id] = candidate
for element in new_children:
    for node in element.iter():
        for key in ["{" + R + "}id", "{" + R + "}embed", "{" + R + "}link"]:
            old = node.get(key)
            if old in relationship_map: node.set(key, relationship_map[old])

parts["word/document.xml"] = xml(doc)
parts["word/_rels/document.xml.rels"] = xml(relations)
core = E.fromstring(parts["docProps/core.xml"])
dc = "http://purl.org/dc/elements/1.1/"
title = core.find("{" + dc + "}title")
if title is None: title = E.SubElement(core, "{" + dc + "}title")
title.text = "面向传感器历史缺失的同伴序列修复与预测融合"
parts["docProps/core.xml"] = xml(core)
with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as z:
    for name, value in parts.items(): z.writestr(name, value)

with ZipFile(REFERENCE) as ref, ZipFile(OUTPUT) as final:
    preserve = [n for n in ref.namelist() if n not in {"word/document.xml", "word/_rels/document.xml.rels", "docProps/core.xml"}]
    assert all(ref.read(n) == final.read(n) for n in preserve)
    assert final.testzip() is None
assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED
assert table_index == 6 and roles["figure"] == 2 and roles["reference"] == 13 and roles["display_equation"] == 7
all_text = text(body)
assert "WORDTITLE" not in all_text and "TABLECAP" not in all_text and "FIGCAP" not in all_text and "BIBLIOGRAPHYPLACEHOLDER" not in all_text
all_math = len(body.findall(".//m:oMath", NS))
assert all_math == len(incoming_body.findall(".//m:oMath", NS)) - 6  # H labels moved into the one-row grid headers.
report = {"output": str(OUTPUT), "reference_sha256": EXPECTED, "preserved_parts": preserve,
          "roles": dict(roles), "tables": table_index, "native_equations": all_math,
          "added_media": added_media, "styles_sha256": hashlib.sha256(parts["word/styles.xml"]).hexdigest()}
(WORK / "format_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
