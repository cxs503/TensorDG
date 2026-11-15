# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 功能量级认证（验证欠账复核）

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open）
> 类型：capability_extension / 验证欠账（已合并功能的文献级基准复核）
> 报告：平台 report id 20004 · 文档 `docs/XVAL-DG-ROUND2-FEATURE-CERTIFICATION.md`

## 1. 做了什么

round-2 认证的是**整条密度剖面**（4 级 FV-WENO5 + Richardson + GCI，11 条 DG 腿 L1 ≤ 2.46 %
过 3 % 门槛）。L1 范数被激波主导，激波后放大熵波波列——本算例真正的物理内容——从未被量化
（既有 `wave_amplitude` 量的是 x ≥ 2.6 的未扰动区）。本单补这一层：

1. **引擎扩展**（可复用，case 无关）：`tensordg.certified_reference.certify_functional` /
   `judge_functional` —— Roache 方法搬到标量功能量，复用三判据（非渐近拒绝 / U/τ ≤ 1/3 /
   band-aware PASS_MARGINAL），显式拒绝收敛到零的功能量（`WITHHELD_ZERO_FUNCTIONAL`），保留符号。
2. **审计**：`benchmarks/xval_dg_round2_feature_certification.py`，5 个功能量
   （amplitude/crest 极值型，rms/l1 积分型，peak_x 相位型），解析锚定的统计窗
   x∈[−2.7031, 2.2894]，预声明判据 G0–G4。
3. **结论**：只有 `amplitude` 通过两族（WENO5+WENO7）认证且门槛可用
   —— 0.827972，U = 0.1169 %（占 3 % 门槛 3.9 %），p = 1.60，跨族差 0.049 %（Δ/(U5+U7)=0.32 覆盖）；
   其余 4 个功能量被引擎拒绝（非单调梯次 / 负阶），如实 WITHHELD。
   11 条"L1 全过"的 DG 腿在波列幅度上误差中位 41.4 %、最差 68.4 %（11/11 FAIL）→
   **全局 L1 门槛对该算例不充分**（G4 = NO，11 个反例）。

## 2. 归因（为什么不是度量伪影）

- 随 N 单调改善：N=200/400/800 → ≈ 68/41/18 %；
- 同网格 N=800：FV-WENO5 2.13 %、FV-WENO7 0.27 %，DG P1/P2/P3 17.4–18.4 % → 格式/限制器效应；
- CFL 0.1/0.2/0.4（DG P2 N=400）极差 0.18 pp → 时间积分不是原因；
- 单族认证的其它功能量（rms 中位 34.7 %、l1 中位 28.9 %，均 11/11 FAIL）→ 结论不依赖度量选择。

## 3. 测试

`tests/test_certified_functional.py` 17 项（解析幂律梯次回收阶与极限 / 符号 / Roache 带宽恒等式 /
6 条拒绝路径 / 判定语义）与 `tests/test_xval_dg_round2_feature_certification.py` 13 项
（解析正弦 amplitude·rms·l1 闭式回收 / 网格不变性 / 窗口锚定 / 结论内一致性 / 归因断言）全绿；
`tests/test_certified_reference.py`、`tests/test_xval_dg_round2_*` 无回归。

已知**与本单无关**的既有红灯：`tests/test_public_api_contract.py` 5 项（89 个未导出符号 + 10 个
`prefinement_2d*` 未入 `__all__`），在暂存本单 `tensordg/__init__.py` 改动后同样失败，属他单欠账。

## 4. 产物

`tensordg/certified_reference.py`（+2 函数）· `benchmarks/xval_dg_round2_feature_certification.py` ·
`validation/xval_dg_round2/artifacts/feature_certification/`（audit.json/md + 3 图）·
`docs/XVAL-DG-ROUND2-FEATURE-CERTIFICATION.md` · 两个新测试文件。
