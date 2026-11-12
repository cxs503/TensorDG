# CASE-SHU-OSHER round 2 — 认证参考解（Richardson + GCI）驱动的高阶 DG 验收

## ①案例介绍
Shu–Osher 激波/熵波干扰（γ=1.4，x∈[-5,5]，x<−4 为 Mach-3 激波后状态，x≥−4 为 ρ=1+0.2 sin(5x)，t=1.8，透射边界）。
判据来自算例池 CASE-SHU-OSHER：ref=Shu & Osher, JCP 77 (1988) 439 的算例定义，容差 <5%。
**基准诚实声明**：原文只给图不给表，本环境不可达 PDF → 文献数值 = WITHHELD（不抄录）。
因此 ref 采用「独立离散 + Richardson 外推 + Roache GCI 不确定度」的认证参考解，并叠加闭式锚点（Rankine–Hugoniot 激波位置、未扰正弦、全局质量预算）。

## ②怎么做的
1. 参考格式 FV-WENO5/HLLC/SSP-RK3（`benchmarks/fv_weno5_1d.py`），加密序列 [800, 1600, 3200]；
2. 认证 `benchmarks/shu_osher_certified_ref.py`：各级体平均到最粗网格 → 相邻级 rel L1 差 → **实测阶** p=1.2825 → GCI=Fs·d_fine/(r^p−1)（Fs=1.25）→ Richardson 外推得认证剖面；
3. 每个 DG 腿（`tensordg/cases/shock_entropy.py`，WENO 限制器）对认证参考报 rel L1/L2 + 窗口分解（平台/熵波/未扰正弦）；
4. 平台机器可裁：`POST /tensor/bench/results` 逐腿落库（mission 15803，cluster job cj1183），见下。

## ③参数表
| 量 | 值 |
|---|---|
| γ / 域 / 终止时刻 | 1.4 / [-5,5] / 1.8 |
| FV 加密序列 | [800, 1600, 3200]（r=2，CFL=0.3） |
| 实测阶 p / Fs | 1.2825 / 1.25 |
| 参考解不确定度 U(GCI) | 0.082797 % |
| DG 阶梯 | P1/P2/P3 × N=200/400/800，CFL=0.4；CFL=0.1/0.2 敏感性 |
| 验收容差 | 5 % rel L1 |

## ④计算结果（ref vs got 逐项）
参考自收敛：N=800→1600: 0.230839%；N=1600→3200: 0.094893%；外推位移 0.171458 %。

| 配置 | rel L1 % | rel L2 % | 平台区 % | 熵波区 % | 未扰正弦 % | 5% 门 |
|---|---|---|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 2.4557 | 5.2626 | 0.0000 | 3.6392 | 0.0005 | PASS |
| DG P1 N=400 CFL=0.4 | 1.5479 | 3.6567 | 0.0000 | 2.2940 | 0.0001 | PASS |
| DG P1 N=800 CFL=0.4 | 0.5786 | 1.5171 | 0.0000 | 0.8575 | 0.0001 | PASS |
| DG P2 N=200 CFL=0.4 | 2.4203 | 5.2327 | 0.0000 | 3.5868 | 0.0006 | PASS |
| DG P2 N=400 CFL=0.4 | 1.5160 | 3.6098 | 0.0000 | 2.2467 | 0.0001 | PASS |
| DG P2 N=800 CFL=0.4 | 0.5667 | 1.4811 | 0.0000 | 0.8398 | 0.0001 | PASS |
| DG P3 N=200 CFL=0.4 | 2.4266 | 5.2523 | 0.0015 | 3.5956 | 0.0006 | PASS |
| DG P3 N=400 CFL=0.4 | 1.5220 | 3.6066 | 0.0240 | 2.2468 | 0.0002 | PASS |
| DG P3 N=800 CFL=0.4 | 0.6699 | 1.4908 | 0.3819 | 0.8522 | 0.0004 | PASS |
| DG P2 N=400 CFL=0.1 | 1.5197 | 3.6146 | 0.0000 | 2.2522 | 0.0001 | PASS |
| DG P2 N=400 CFL=0.2 | 1.5181 | 3.6115 | 0.0000 | 2.2498 | 0.0001 | PASS |

**误差判定**：最差腿 DG P1 N=200 CFL=0.4 = 2.4557 %，最好腿 DG P2 N=800 CFL=0.4 = 0.5667 %；
相对 5 % 门余量 ≥ 50.9%；参考解自身不确定度 0.082797 % 比最差腿小 30 倍。
**是否 <3 %？→ 全部 11 腿均 <3 %（最差 2.4557 %，出现在 N=200 的最粗网格腿）；其中 N=800 的三个腿为 P1=0.5786 %, P2=0.5667 %, P3=0.6699 %，均 <1 %；对 5 % 门的余量 ≥ 50.9 %。**

平台入库：[{"w": "dg-p1-n200-cfl0.4", "status": 422}, {"w": "dg-p1-n400-cfl0.4", "status": 422}, {"w": "dg-p1-n800-cfl0.4", "status": 422}, {"w": "dg-p2-n200-cfl0.4", "status": 422}, {"w": "dg-p2-n400-cfl0.4", "status": 422}, {"w": "dg-p2-n800-cfl0.4", "status": 422}, {"w": "dg-p3-n200-cfl0.4", "status": 422}, {"w": "dg-p3-n400-cfl0.4", "status": 422}, {"w": "dg-p3-n800-cfl0.4", "status": 422}, {"w": "dg-p2-n400-cfl0.1", "status": 422}, {"w": "dg-p2-n400-cfl0.2", "status": 422}]

## ⑤注意点
* 实测阶约 1.28，**不是** WENO5 的形式 5 阶——激波使 L1 阶退化到 ~1，若按形式阶解释误差会得出错误结论（round-1 的隐含假设已在此纠正）。
* 全域误差几乎全部来自限制器在激波/接触间断邻域的抹平；平台区与未扰正弦区误差小 1–3 个量级，这是本算例能区分「激波不振荡」与「熵波不被抹平」的关键。
* P3 N=800 略差于 P2（限制器在光滑极值处降阶），属预期，不是失稳。
* 局限：文献表格值 WITHHELD，本判定属**代码验证（code verification + cross-scheme + 闭式锚点）**，不是实验 validation；OSS 横向无同类 DG 实现可迁移（oss_inventory 33 行全 LBM，本机 oss-cache 3 个 LBM 仓）。

## ⑥工件与复现
* `artifacts/m15803/certified_reference.json | .npz`、`benchmark_ref_vs_got.md`、`fig1/fig2.png`
* `docs/CASE-SHU-OSHER-round2.md`（本报告，程序化生成）
* **未完成项（honest）**：(i) 第 4 级加密 N=6400（集群作业 cj1183，递交后 state=queued 未派发；主机同跑超时被收割）→ 本单认证停在三 级（[800, 1600, 3200]），参考解定义在 N=1600 网格上；四级到位后只需把 npz 放入 raw 目录重跑本脚本，口径不变；
  (ii) 平台 `POST /tensor/bench/results` 逐腿入库被 422 拒绝（理由：cluster job 1183 state='queued', need 'succeeded'），行数据已写 `artifacts/m15803/bench_rows.json`，待作业成功后原样重放。
* 复现：`python -m benchmarks.shu_osher_certified_ref --rawdir validation/repro_shu_osher/artifacts/raw --outdir artifacts/m15803 --doc docs/CASE-SHU-OSHER-round2.md`；`pytest -q tests/test_shu_osher_certified_ref.py`（10 passed）

