"""Build the round-17 report payload (figure + markdown) for CASE-DG-EULER-VORTEX-ERR."""
import base64
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
labels = ["recorded", "replayed", "unverified", "rejected"]
r16 = [21, 16, 3, 0]
r17 = [21, 18, 1, 0]
x = np.arange(len(labels))
w = 0.38
axes[0].bar(x - w / 2, r16, w, label="round 16", color="#9aa7b4")
axes[0].bar(x + w / 2, r17, w, label="round 17", color="#2b6cb0")
for i, (a, b) in enumerate(zip(r16, r17)):
    axes[0].text(i - w / 2, a + 0.3, str(a), ha="center", fontsize=9)
    axes[0].text(i + w / 2, b + 0.3, str(b), ha="center", fontsize=9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("certificate records")
axes[0].set_title("C11 provenance classification")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

jobs = ["cj4184\nn64_t2, 1thr", "cj4185\nn32_t10, 1thr", "cj4186\nn64_t10, 4thr"]
delta = [0.0, 1.128e-12, None]
colors = ["#2f855a", "#2f855a", "#c53030"]
bars = axes[1].bar(jobs, [1, 1, 1], color=colors, alpha=0.85)
axes[1].set_yticks([])
axes[1].set_ylim(0, 1.25)
for bar, d in zip(bars, delta):
    text = ("certificate landed\nreplay_delta = 0.0" if d == 0.0 else
            f"certificate landed\nreplay_delta = {d:.2e}" if d else
            "no certificate\n(only failure)")
    axes[1].text(bar.get_x() + bar.get_width() / 2, 0.5, text, ha="center",
                 va="center", color="white", fontsize=9, weight="bold")
axes[1].set_title("in-flight replay jobs: evidence recovered from artifacts")
fig.suptitle("CASE-DG-EULER-VORTEX-ERR round 17: unverified 3 -> 1 "
             "(the evidence already existed)", fontsize=11)
fig.tight_layout()
fig.savefig("/tmp/fig_round17_provenance.png", dpi=140)

png = base64.b64encode(open("/tmp/fig_round17_provenance.png", "rb").read()).decode()
markdown = open("docs/CASE-DG-EULER-VORTEX-ERR-round17-forensic-landing.md").read()
report = {
    "title": "CASE-DG-EULER-VORTEX-ERR round 17：集群在飞证书的取证收账（unverified 3→1）",
    "summary": "取回 round16 遗留的三条在飞重放作业工件，发现其中两条证书其实已算完"
               "（replay_delta 0.0 / 1.13e-12）只是从未入库；经严格搬运器入库后 C11 unverified "
               "由 3 条压到 1 条（剩余 dg_d2_n64_t10_cfl0.2 已按 1 线程重放递交 cj4263），"
               "整包仍如实记为 INCOMPLETE。",
    "scenario_key": "CASE-DG-EULER-VORTEX-ERR",
    "mission_id": 22612,
    "tags": ["DG", "Euler", "isentropic vortex", "provenance", "C11",
             "honest-INCOMPLETE", "Yee2000"],
    "markdown": markdown,
    "figures": [{"name": "fig_round17_provenance.png",
                 "caption": "左：C11 溯源分类 round16→round17（unverified 3→1）；"
                            "右：三条在飞作业的取证结局（两条有证书，4 线程那条无）。",
                 "png_base64": png}],
}
json.dump(report, open("/tmp/report.json", "w"), ensure_ascii=False)
print("report payload bytes:", len(markdown))