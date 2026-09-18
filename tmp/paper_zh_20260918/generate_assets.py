from pathlib import Path
import importlib.util
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EN = ROOT / "docs/iclr2027/current_results"
ZH = ROOT / "docs/paper_zh"
LABELS = {
    "Native, 192 h": "原生预测，192 小时",
    "Native, long": "原生预测，长历史",
    "Native, long, targets only": "原生长历史，仅目标",
    "Native, long, target only": "原生长历史，仅目标",
    "VAR": "VAR",
    "Native, long + VAR": "原生长历史＋VAR",
    "Native, long, targets + VAR": "原生长历史，仅目标＋VAR",
    "Target KNN, 192 h + VAR": "目标 KNN，192 小时＋VAR",
    "Gaussian repair, long": "高斯修复，长历史",
    "Gaussian repair, long + VAR": "高斯长历史修复＋VAR",
    "KNN repair, long + VAR": "KNN 长历史修复＋VAR",
    "Local ridge, long, targets only": "局部岭回归，长历史，仅目标",
    "Corruption-trained LoRA, 192 h": "缺失扰动 LoRA，192 小时",
    r"\method{}": r"\method{}",
    "LOCF": "LOCF",
    "Linear interpolation": "线性插值",
    "Seasonal lag": "季节滞后",
    "Multivariate KNN": "多变量 KNN",
    "SAITS": "SAITS",
    "TimeMixer++": "TimeMixer++",
    "MoTM": "MoTM",
    "Eight-candidate mean": "八候选均值",
    "Eight-candidate median": "八候选中位数",
    "Peer ridge": "同伴岭回归",
    "Peer ridge + AR residual": "同伴岭回归＋AR 残差",
    "Peer ridge + linear residual": "同伴岭回归＋线性残差",
    "Natural-history LoRA": "原历史 LoRA",
    "Corruption-trained LoRA": "缺失扰动 LoRA",
    "Median repair, long, target only": "中位数长历史修复，仅目标",
    "Median repair, long, with peers": "中位数长历史修复，含同伴",
    "Auxiliary-query mask, time": "辅助查询掩码，时间",
    "Auxiliary-query mask, time/group": "辅助查询掩码，时间与分组",
    "Nine-candidate median, 192 h": "九候选中位数，192 小时",
    "Aotizhongxin": "奥体中心", "Changping": "昌平", "Dingling": "定陵",
    "Dongsi": "东四", "Guanyuan": "官园", "Gucheng": "古城", "Huairou": "怀柔",
    "Nongzhanguan": "农展馆", "Shunyi": "顺义", "Tiantan": "天坛", "Wanliu": "万柳",
    "Wanshouxigong": "万寿西宫",
}
checks = []
for name in ["main_rows.tex", "hdb_rows.tex", "baseline_rows.tex", "station_rows.tex"]:
    lines = (EN / name).read_text(encoding="utf-8").splitlines()
    translated = []
    for line in lines:
        if " & " not in line:
            translated.append(line)
            continue
        label, values = line.split(" & ", 1)
        translated.append(LABELS[label] + " & " + values)
    result = "\n".join(translated) + "\n"
    # Every numeric cell and its emphasis stays byte-identical to the English source.
    old_values = [line.split(" & ", 1)[1] for line in lines if " & " in line]
    new_values = [line.split(" & ", 1)[1] for line in translated if " & " in line]
    assert old_values == new_values
    (ZH / name).write_text(result, encoding="utf-8")
    checks.append({"table": name, "matching_rows": len(old_values)})

font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name()
plt.rcParams.update({"font.family": font, "font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 10, "axes.unicode_minus": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
summary = pd.read_csv(EN / "all_panel_results.csv", float_precision="round_trip")
values = summary[summary.panel == "beijing/natural_outage_h24"].set_index("method")
curves = [
    ("native_peer", "native_long_prefix_peer", "原生预测", "#777777", "o", "-"),
    ("peer_ridge", "long_repair_peer_ridge_peer", "同伴修复", "#2F5597", "s", "-"),
    ("half_var_native_peer", "half_var_native_long_prefix_peer", "原生预测＋VAR", "#A96A22", "^", "--"),
    ("half_var_peer_ridge", "half_var_long_repair_peer_ridge_peer", "同伴修复＋VAR", "#257657", "D", "--"),
]
fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.9))
for metric, ax in zip(["mae", "mse"], axes):
    for short, long, label, color, marker, linestyle in curves:
        ax.plot([0, 1], [values.loc[short, metric], values.loc[long, metric]],
                label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.5, markersize=5)
    ax.set_xticks([0, 1], ["192 小时", "8192 小时"])
    ax.set_xlim(-0.15, 1.15)
    ax.set_xlabel("预测器历史长度")
    ax.set_ylabel("标准化 " + metric.upper())
    ax.set_title("（a）绝对误差" if metric == "mae" else "（b）平方误差", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.subplots_adjust(left=0.09, right=0.985, top=0.89, bottom=0.29, wspace=0.36)
fig.savefig(ZH / "history_repair_fusion_zh.pdf")
fig.savefig(ZH / "history_repair_fusion_zh.png", dpi=200)
plt.close(fig)

en_files = [ROOT / "docs/iclr2027/tsfm_fais_iclr2027.tex"] + list((ROOT / "docs/iclr2027/sections").glob("current_*.tex"))
zh_files = [ROOT / "docs/tsfm_fais_current_zh.tex"] + [ZH / name for name in ["introduction.tex", "method.tex", "experiments.tex", "appendix.tex"]]
en_text = "\n".join(p.read_text(encoding="utf-8") for p in en_files)
zh_text = "\n".join(p.read_text(encoding="utf-8") for p in zh_files)
get_labels = lambda text: set(re.findall(r"\\label\{([^}]+)\}", text))
get_cites = lambda text: {key.strip() for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", text) for key in group.split(",")}
assert get_labels(en_text) == get_labels(zh_text)
assert get_cites(en_text) == get_cites(zh_text)
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", zh_text))
assert refs <= get_labels(zh_text)
report = {"tables": checks, "matched_labels": len(get_labels(zh_text)),
          "matched_references": len(get_cites(zh_text)), "new_experiments": False}
(ROOT / "tmp/paper_zh_20260918/translation_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
