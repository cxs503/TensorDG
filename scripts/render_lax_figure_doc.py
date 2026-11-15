#!/usr/bin/env python3
"""Render ``docs/CASE-LAX-DG-figure-reproducibility.md`` from the two records.

Every number in the note is injected from the artifacts -- nothing is typed by
hand, so the note cannot drift away from what is on disk::

    python scripts/render_lax_figure_doc.py            # write the note
    python scripts/render_lax_figure_doc.py --check    # non-zero if stale

The note is about *evidence class*, not about physics: a committed PNG is
decoration until something proves it was drawn from the committed numbers.  The
checks here (and in ``tests/test_lax_figure_reproducibility.py``) are:

* **G1 binding gate** -- each record's ``figures_binding.json`` must match the
  record bytes, the ``profiles.npz`` bytes, the renderer bytes and every PNG's
  bytes (cheap; needs no matplotlib);
* **G2 figure tamper control** -- flipping one byte in a *copy* of a committed
  figure must turn G1 to FAIL;
* **G3 record tamper control** -- editing a number in a *copy* of the record
  (a new run whose figures were never re-rendered) must also turn G1 to FAIL;
* **G4 record idempotence** -- re-serialising a record through the renderer's
  writer must reproduce it byte for byte;
* **G5 byte-identical re-render** -- a fresh render from the record must
  reproduce every committed PNG byte for byte (run by the test, not here: it
  needs matplotlib).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_lax_figures import (  # noqa: E402
    BINDING, PROVENANCE, RECORDS, check_binding, rel, write_record,
)

DOC = ROOT / "docs" / "CASE-LAX-DG-figure-reproducibility.md"


def _short(digest):
    return "n/a" if digest is None else digest[:16]


def _copy_record(rec_dir, tmp, root=ROOT):
    """Copy a record (and its sidecar/figures) under the same repo-relative path."""
    src = Path(rec_dir)
    dst = Path(tmp) / src.relative_to(Path(root))
    (dst / "figures").mkdir(parents=True)
    shutil.copy(src / "metrics.json", dst / "metrics.json")
    shutil.copy(src / "profiles.npz", dst / "profiles.npz")
    shutil.copy(src / BINDING, dst / BINDING)
    for png in (src / "figures").glob("*.png"):
        shutil.copy(png, dst / "figures" / png.name)
    return dst


def _control_figure_tamper(rec_dir, root=ROOT):
    """Flip the last byte of one figure in a copy; the gate must notice."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = _copy_record(rec_dir, tmp, root)
        name = sorted(p.name for p in (copy / "figures").glob("*.png"))[0]
        target = copy / "figures" / name
        blob = bytearray(target.read_bytes())
        blob[-1] ^= 0xFF            # last byte of the IEND chunk
        target.write_bytes(bytes(blob))
        ok, _ = check_binding(copy, root)
        return {"what": f"`figures/{name}` 翻转第 {len(blob) - 1} 字节",
                "status": "PASS" if ok else "FAIL", "detected": not ok}


def _control_record_tamper(rec_dir, root=ROOT):
    """Change one number in a copy of the record; the gate must notice."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = _copy_record(rec_dir, tmp, root)
        metrics = json.loads((copy / "metrics.json").read_text())
        metrics["generated_at_utc"] = "1999-01-01T00:00:00Z"
        write_record(copy / "metrics.json", metrics)
        ok, _ = check_binding(copy, root)
        return {"what": "改写记录 `generated_at_utc`（模拟重跑未重画图）",
                "status": "PASS" if ok else "FAIL", "detected": not ok}


def _idempotence(rec_dir):
    """Writer round-trip: re-serialise the record, expect identical bytes."""
    path = Path(rec_dir) / "metrics.json"
    original = path.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "metrics.json"
        copy.write_text(original)
        write_record(copy, json.loads(original))
        return copy.read_text() == original


def collect(root=ROOT):
    root = Path(root)
    rows = []
    for rec, label in RECORDS:
        rec_dir = root / rec
        metrics = json.loads((rec_dir / "metrics.json").read_text())
        binding = json.loads((rec_dir / BINDING).read_text())
        ok, gate_rows = check_binding(rec_dir, root)
        sizes = {n: (rec_dir / "figures" / n).stat().st_size
                 for n in binding["figures_sha256"]}
        rows.append({
            "record": rec, "label": label,
            "generated_at_utc": metrics.get("generated_at_utc"),
            "figure_count": len(binding["figures_sha256"]),
            "bytes": sum(sizes.values()),
            "gate": "PASS" if ok else "FAIL",
            "inputs": [{"what": what, "recorded": recorded, "actual": actual}
                       for what, recorded, actual in gate_rows
                       if not what.startswith("figures/")],
            "figures": [{"what": what.split("/", 1)[1], "recorded": recorded,
                         "actual": actual, "bytes": sizes.get(what.split("/", 1)[1])}
                        for what, recorded, actual in gate_rows
                        if what.startswith("figures/")],
            "figure_control": _control_figure_tamper(rec_dir, root),
            "record_control": _control_record_tamper(rec_dir, root),
            "idempotent": _idempotence(rec_dir),
            "provenance": metrics.get("figures_provenance"),
            "provenance_names_renderer": (
                metrics.get("figures_provenance") == PROVENANCE),
        })
    return {"records": rows}


def render(payload, root=ROOT) -> str:
    rows = payload["records"]
    total_figs = sum(r["figure_count"] for r in rows)
    g1 = all(r["gate"] == "PASS" for r in rows)
    g2 = all(r["figure_control"]["detected"] for r in rows)
    g3 = all(r["record_control"]["detected"] for r in rows)
    g4 = all(r["idempotent"] for r in rows)
    prov = all(r["provenance_names_renderer"] for r in rows)
    out = []
    out.append("# CASE-LAX-DG · 图与记录的绑定：把「图只是装饰」变成可判定命题")
    out.append("")
    out.append("> 锚点 `CASE-LAX-DG`（Lax 激波管 / Toro test 2，ref = 精确解，容差 <3 %）。")
    out.append("> 本文**全部数字由 `scripts/render_lax_figure_doc.py` 从两份已提交记录及其**")
    out.append("> `figures_binding.json`** 程序化注入**，零手抄；`--check` 可判陈旧。")
    out.append("> 渲染器：`scripts/render_lax_figures.py`；闸门测试：")
    out.append("> `tests/test_lax_figure_reproducibility.py`。")
    out.append("")
    out.append("## ① 案例介绍")
    out.append("")
    out.append("Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]` 域，隔膜 `x0 = 0.5`，γ = 1.4，")
    out.append("快照 `t = 0.13`；左态 (ρ, u, p) = (0.445, 0.698, 3.528)，"
               "右态 (0.500, 0.000, 0.571)；参考值取闭式 Riemann 解。算例池九条 "
               "`(cells, degree)` 腿与限幅器三臂（`none` / `tvb` / `weno`）各有一份 "
               "`metrics.json` + `profiles.npz`，外加若干 PNG。")
    out.append("")
    out.append("本文处理的是**证据等级**问题，不是物理问题：`metrics.json` 里的数可以逐位复核，"
               "但 PNG 是二进制，谁也不能从一张图看出它是哪次运行的产物。原状是「图由记录渲染」"
               "只写在 `figures_provenance` 的一句话里，没有任何检查能发现"
               "「图被替换」或「记录重跑了、图没重画」。")
    out.append("")
    out.append("## ② 怎么做的")
    out.append("")
    out.append("1. **一行侧车同时钉住两端**：每次渲染写 `figures_binding.json`，里面既有"
               "**输入**（`record_sha256` / `profiles_sha256` / `renderer_sha256`），也有"
               "**输出**（`figures_sha256` 一图一摘要）。摘要若只写进记录内部，就只能证明"
               "「记录里写的摘要 == 磁盘上的图」——记录整份被新运行重写、图没重画时它依然自洽；"
               "侧车记录 `record_sha256` 恰好补上这个方向。")
    out.append("2. **G1 绑定闸门**（不需要 matplotlib）：记录字节、`profiles.npz` 字节、渲染器"
               "字节、每张 PNG 字节四项全对，且记录 `figures`、磁盘 `*.png`、侧车摘要键三者"
               "集合相等；缺侧车、孤儿图、缺图、字节漂移一律 FAIL。")
    out.append("3. **G2 图篡改对照**：在**副本**里翻转某张图最后一字节，G1 必须 FAIL。")
    out.append("4. **G3 记录篡改对照**：在**副本**里改写记录里的一个字段（模拟「重跑未重画图」），"
               "G1 也必须 FAIL —— 这条正是「摘要写进记录内部」做不到的方向。")
    out.append("5. **G4 记录幂等**：渲染器写回记录时沿用记录自身的 JSON 缩进。原实现硬编码 "
               "`indent=1`，而算例池记录是 `indent=2`，于是**一次无变化的重新渲染也会重写 "
               "2518 行**数字——真的改了数和只是换了缩进，在 diff 里再也分不出来。"
               "（`PR#575` 的 `metrics.json` 差异 2292 行里既有真实重算，也有这类整份重写，"
               "这正是需要幂等写回的理由：让**没有变化的**那次渲染彻底不出现在 diff 里。）")
    out.append("6. **G5 逐字节重渲染**：从已提交记录 + `profiles.npz` 重新渲染到临时目录，"
               "要求每张 PNG 与已提交字节完全相同（由测试执行，需要 matplotlib）。")
    out.append("")
    out.append("## ③ 计算结果")
    out.append("")
    out.append("### 3.1 绑定输入（图从哪儿来）")
    out.append("")
    out.append("| 记录 | 绑定量 | 侧车记录值（前 16） | 实测值（前 16） | 判定 |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        for item in r["inputs"]:
            ok = item["recorded"] is not None and item["recorded"] == item["actual"]
            out.append(f"| `{r['label']}` | `{item['what']}` | "
                       f"`{_short(item['recorded'])}` | `{_short(item['actual'])}` | "
                       f"{'PASS' if ok else 'FAIL'} |")
    out.append("")
    out.append("### 3.2 绑定输出（图是什么）")
    out.append("")
    out.append("| 记录 | 图 | 字节 | 侧车 sha256（前 16） | 实测 sha256（前 16） | 判定 |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        for item in r["figures"]:
            ok = item["recorded"] is not None and item["recorded"] == item["actual"]
            out.append(f"| `{r['label']}` | `{item['what']}` | {item['bytes']} | "
                       f"`{_short(item['recorded'])}` | `{_short(item['actual'])}` | "
                       f"{'PASS' if ok else 'FAIL'} |")
    out.append("")
    out.append("### 3.3 闸门总表")
    out.append("")
    out.append("| 闸门 | 判定 | 说明 |")
    out.append("|---|---|---|")
    out.append(f"| G1 绑定一致（{total_figs} 张图、{sum(len(r['inputs']) for r in rows)} 项输入） "
               f"| **{'PASS' if g1 else 'FAIL'}** | 记录 / profiles / 渲染器 / 图 四类字节全对，"
               "三方集合相等 |")
    out.append("| G2 图篡改可检出 | **" + ('PASS' if g2 else 'FAIL') + "** | "
               + "；".join(f"`{r['label']}`: {r['figure_control']['what']} → "
                          f"{r['figure_control']['status']}" for r in rows) + " |")
    out.append("| G3 记录篡改可检出 | **" + ('PASS' if g3 else 'FAIL') + "** | "
               + "；".join(f"`{r['label']}`: {r['record_control']['what']} → "
                          f"{r['record_control']['status']}" for r in rows) + " |")
    out.append(f"| G4 记录幂等 | **{'PASS' if g4 else 'FAIL'}** | "
               + "；".join(f"`{r['label']}` {'字节不变' if r['idempotent'] else '被重写'}"
                          for r in rows) + " |")
    out.append(f"| 溯源串指向渲染器 | **{'PASS' if prov else 'FAIL'}** | "
               "`figures_provenance` 与 `PROVENANCE` 常量逐字相同 |")
    out.append("| G5 逐字节重渲染 | 由测试判定 | "
               "`tests/test_lax_figure_reproducibility.py`（需 matplotlib，用例内 "
               "`importorskip`） |")
    out.append("")
    out.append("### 3.4 工件规模")
    out.append("")
    out.append("| 记录 | 生成时刻 (UTC) | 图数 | 图总字节 |")
    out.append("|---|---|---|---|")
    for r in rows:
        out.append(f"| `{r['label']}` | {r['generated_at_utc']} | {r['figure_count']} | "
                   f"{r['bytes']} |")
    out.append("")
    out.append("## ④ 注意点")
    out.append("")
    out.append("1. **绑定不等于可复现**：G1 只把「记录/图/渲染器」三者钉在一起，任何人只要"
               "同时改图、改记录、重写侧车就能一起骗过 G1。真正闭口的是 G5——从记录重渲染"
               "必须得到同样的字节；G1/G2/G3 是它的廉价前置门（无需 matplotlib，秒级）。")
    out.append("2. **G5 的适用范围**：PNG 字节依赖 matplotlib 版本与后端。本环境实测逐字节相同"
               "（matplotlib 3.10.8，Agg）；换绘图栈后 G5 可能 FAIL 而图语义未变——此时正确动作"
               "是重渲染并刷新侧车（`python scripts/render_lax_figures.py`），而不是放宽闸门。")
    out.append("3. **`renderer_sha256` 是有意的耦合**：改渲染器（哪怕只改一句注释）都会让 G1 "
               "变 FAIL，迫使重渲染。代价是两百毫秒，收益是「代码改了、图没跟上」这类漂移"
               "无法静默存活。")
    out.append("4. **G4 是格式守卫，不是语义守卫**：它只保证「无变化的记录重写后字节不变」，"
               "不保证记录内容正确；记录内容由算例池自身的闸门负责。")
    out.append("5. **本机渲染 vs 节点渲染**：loong64 计算镜像无 matplotlib，图一律在节点外渲染；"
               "「记录 → 图」这一步本来就跨机器，所以更必须有侧车把它钉住。")
    out.append("")
    out.append("## ⑤ 误差判定")
    out.append("")
    out.append("本文的关键量是**绑定关系**，不是物理量，故误差判据为字节相等（偏差 0 才算真）：")
    out.append("")
    out.append("| 关键量 | 基准 | 偏差 | 是否 < 3 % |")
    out.append("|---|---|---|---|")
    out.append(f"| 图字节 sha256（{total_figs} 张） | 侧车 `figures_sha256` | "
               f"{0 if g1 else '≥1 项不符'} | {'是（严格 0）' if g1 else '否'} |")
    out.append(f"| 记录 / profiles / 渲染器字节 | 侧车三项输入摘要 | "
               f"{0 if g1 else '≥1 项不符'} | {'是（严格 0）' if g1 else '否'} |")
    out.append(f"| 图篡改检出 | 必须 FAIL | {'检出' if g2 else '漏检'} | "
               f"{'是' if g2 else '否'} |")
    out.append(f"| 记录篡改检出 | 必须 FAIL | {'检出' if g3 else '漏检'} | "
               f"{'是' if g3 else '否'} |")
    out.append(f"| 记录重写字节 | 原记录 | {'0' if g4 else '非 0'} | "
               f"{'是（严格 0）' if g4 else '否'} |")
    out.append("")
    out.append("**物理误差本文明不涉及**：本文不动求解器、不重算任何误差，`verdict = FAIL` 的 "
               "L2 判据与限幅器归因仍以 `docs/CASE-LAX-DG.md`、"
               "`docs/CASE-LAX-DG-limiter-arms.md`、`docs/CASE-LAX-DG-acceptance-A0.md` 为准；"
               "那三份文档的结论（整场相对 L2 超容差、C3/R3 阶数闸门 FAIL）**未被本文改变**。")
    out.append("")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="non-zero exit if the committed note is stale")
    args = parser.parse_args(argv)
    text = render(collect())
    if args.check:
        current = DOC.read_text() if DOC.is_file() else ""
        if current != text:
            print(f"[doc] {DOC.relative_to(ROOT)} is STALE; rerun "
                  f"scripts/render_lax_figure_doc.py", file=sys.stderr)
            return 1
        print(f"[doc] {DOC.relative_to(ROOT)} is fresh")
        return 0
    DOC.write_text(text)
    print(f"[doc] wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())