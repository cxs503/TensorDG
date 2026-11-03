# 二维高阶 Euler DG：验证欠账闭环（m2143 honest FAIL → m2400 PASS）

- 任务：m2400（models 席自找活），锚点 = `docs/verify_euler_dg_2d_attempt.md` §7「给下一次自找活的建议」，
  以及 `expand_audit_tensordg.md` §6 backlog 第 3 项「2D Euler 高阶」。
- 分支：`agent/m2400`　日期：2026-11-03　Python 3.11 / torch 2.7.1 / pytest 9.2.0
- 一句话：m2143 定位到「面通量投影」后停滞的装配缺陷**已修复**（根因是一行遗漏的 `torch.roll`），
  二维残差与独立的 1D DG 实现逐格逐模一致到 **1e-15 相对**；光滑涡取得 **EOC≈2.95（P2 设计阶 3）**，
  挤压 Sod 对精确 Riemann 解 **0.76% / 0.71% < 3%**；模块状态由 EXPERIMENTAL 转 **VERIFIED**。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

验证对象：`tensordg/euler_dg_2d.py`（张量积 Legendre 模态 DG + HLLC 法向通量 + SSP-RK3
+ positivity 回退 + 可选 TVB 限制器），二维可压缩 Euler 方程，理想气体 γ=1.4。

| 算例 | 几何 | 参数 | 判据 |
|---|---|---|---|
| A 跨码残差一致性 | x∈[0,1]，y 单层（ny=1），y 不变数据，64 格 | P1/P2；x 模态全填（tanh 剖面 L2 投影） | 二维残差 = 独立 1D DG 残差（相对 <1e-12） |
| B 挤压 Sod | 同上，100 格，隔膜 x=0.5 | 左 (1,0,1)，右 (0.125,0,0.1)，t=0.2，CFL 0.2，P1 | vs Sod 精确解 <3% |
| C 等熵涡 | [-5,5]²，周期 | β=5，背景速度 (1,1)，t=1.0，P2，h=10/16、10/24 | EOC≈3 且误差 <3% |
| D 四象限 Riemann | [0,1]²，48²（另有 32² 网格自收敛） | Lax & Liu (1998) config 3，t=0.25，P1 + TVB | 正性 + 有限性（无解析解，如实声明） |

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **复现**（`python -m benchmarks.diagnose_euler_dg_2d`）：界面 HLLC 质量通量 4.3026e-01 正确，
   但投影后为 0、二维密度/能量残差恒为 0 —— 与 m2143 记录一致。
2. **定位**：对 `_hllc` 打桩后发现其收到的 `ur[49] = 1.0`（应是右单元 0.125），即
   `_axis_fluxes` 在**非周期**分支里 `neighbour = trace_left.clone()` **漏了 `torch.roll`**，
   于是 HLLC 两侧同状态、退化为物理通量（Rusanov 分支同理），迎风信息全部丢失；
   单元内通量差随即相消 → 残差恒零。周期分支有 roll，所以光滑周期算例不会暴露（m2143 §5.3 的观察得到解释）。
   仓库里**已合并且已通过验证**的 1D 实现 `euler_dg_1d._rhs` 用的正是「先 roll、再只覆盖最外侧面」
   这一正确写法，可作为交叉参照。
3. **修复**（最小改动，2 处）：两个方向的 neighbour 一律先 `torch.roll(-1)`，
   非周期时只把最外侧面覆盖为内部迹（outflow）。同时校正了诊断脚本里一个度量口径错误
   （此前把「点值通量」与「投影后的面积分通量」直接相除，必然得 100% 损失）。
4. **顺带修复的第二个真实缺陷（影响所有光滑算例的阶）**：初值只写入 P0 模态
   （`_initial_averages`），光滑初值因此只有一阶精度。新增 `initial_projection="l2"`
   （对**全部模态**做 L2 投影，默认开启）并保留 `"cell-average"` 复刻旧行为，供与 1D 求解器做
   apples-to-apples 的跨码比较（1D 求解器仍用 P0 初值）。
5. **验证**：新增 `benchmarks/verify_euler_dg_2d.py`（A/B/C/D 四查 + JSON 落盘）、
   `benchmarks/report_euler_dg_2d_verification.py`（出图/出数）、
   回归位 `tests/test_euler_dg_2d.py`（6 条，取代原 `xfail(strict=True)` 钉扎）。

## 3. 计算结果（程序化读数，见 `validation/euler_dg_2d/evidence.json`）

### 3.1 A 跨码残差一致性（m2143 缺陷的直接回归）

| 阶 | max｜2D−1D｜ | 参考量级 | 相对差 |
|---|---:|---:|---:|
| P1 | 8.527e-14 | 77.35 | **1.10e-15** |
| P2 | 1.681e-13 | 56.27 | **2.99e-15** |

（`y` 动量残差 <1e-10，即 y 不变数据不产生 y 伪源。）m2143 时该相对差为 **100%**。

### 3.2 B 挤压 Sod vs 精确 Riemann 解（P1，100 格，t=0.2，1041 步）

| 量 | 相对 L2 | 是否 <3% |
|---|---:|---|
| 密度 | **0.761%** | ✅ 是 |
| 压力 | **0.707%** | ✅ 是 |

### 3.3 C 等熵涡（P2，t=1.0）

| 网格 | h | ρ 相对 L2 | p 相对 L2 |
|---|---:|---:|---:|
| 16² | 0.625 | 3.976e-04 | 5.625e-04 |
| 24² | 0.417 | 1.203e-04 | 1.685e-04 |

观测收敛阶（16²→24²）：**ρ: 2.95，p: 2.97**，对照 P2 设计阶 3（误差 1.7%/1.0%）。

### 3.4 D 四象限 Riemann（Lax & Liu config 3，P1+TVB，48²，t=0.25）

正性保持、全场有限；32²→48² 的网格自收敛指标见 `verify.json`（无解析解，仅作网格无关性指示）。

## 4. 结果图

- `validation/euler_dg_2d/figures/verification_summary.png`：左=涡的误差-网格 log-log（含 slope 3 参考线），
  右=Sod 密度剖面 vs 精确解。
- `validation/euler_dg_2d/figures/vortex_field_p2_24.png`：P2 24² 的数值密度场与解析解密度场对照。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **非周期的邻元 roll 是最易漏的一行**：周期算例永远不暴露（roll 与 clone 在 ny=1 周期下等价），
   光滑算例也不暴露（残差为 0 时解「看起来还在动」）。挤压/激波 + 非周期是唯一有效探针。
2. 本机 torch 默认线程数很高，二维 DG 小网格务必 `torch.set_num_threads(1..4)`，否则把算法误判成慢。
3. **P0 初值会把光滑算例的收敛阶压到 1**：这是本轮第二个独立缺陷；凡做光滑收敛研究必须用 L2 投影初值。
4. 局限：无粘（无 NS 粘性通量）、结构网格（非结构未做）、限制器只到方向性 minmod/TVB；
   四象限算例无解析解，只能给正性 + 网格自收敛证据，不能给误差比。
5. 本轮未做：`tensordg/__init__.py` 的公开导出（保持实验边界，建议下一单连同 API 文档一起做）；
   三维推广；WENO 限制器的二维版本。

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---:|---|
| 挤压 Sod 密度 L2 | Sod 精确 Riemann 解 | 0.761% | 7.61e-03 | ✅ 是 |
| 挤压 Sod 压力 L2 | 同上 | 0.707% | 7.07e-03 | ✅ 是 |
| 等熵涡 ρ L2 (24²) | 解析等熵涡 | 1.203e-04 | 1.20e-04 | ✅ 是 |
| 等熵涡 p L2 (24²) | 解析等熵涡 | 1.685e-04 | 1.69e-04 | ✅ 是 |
| 跨码残差（P1/P2） | 独立 1D DG 残差 | 1.1e-15 / 3.0e-15 | — | ✅ 是（远优） |
| 涡的收敛阶 | 设计阶 p+1=3 | 2.95（ρ）/2.97（p） | 1.7% / 1.0% | ✅ 是 |
| 四象限算例 | **无解析解** | 正性/有限性通过 | 不可定义 | — 声明原因 |

**判定：本轮能力达到 3% 门，m2143 的 honest FAIL 正式翻案为 PASS。** 局限已在 §5 如实列出。

## 7. 复现命令

```bash
PYTHONPATH=. python -m benchmarks.diagnose_euler_dg_2d          # m2143 缺陷（现已闭合）
PYTHONPATH=. python -m benchmarks.verify_euler_dg_2d --threads 4 # A/B/C/D + JSON
PYTHONPATH=. python -m benchmarks.report_euler_dg_2d_verification # 图 + evidence.json
PYTHONPATH=. python -m pytest -q tests/test_euler_dg_2d.py      # 6 passed
```