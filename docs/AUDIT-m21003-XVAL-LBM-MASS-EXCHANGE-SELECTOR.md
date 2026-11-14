# 交叉审计：m21003 · XVAL-LBM-MASS-EXCHANGE-SELECTOR（TensorDG / TensorLBM）

- **审计单**：m21040（交差核验，非重做）
- **审计对象**：mission 21003，`TensorLBM` 子包，HEAD `6000a43` / 实现提交 `a43d251`
- **工作树**：`/root/tp-agent/work/m21003`（只读审计；审计方副本 `audit/refs/TensorLBM_m21003`，sha256 见 `audit/artifacts/sha256_subject.txt`）
- **独立性**：审计方未参与 m21003 实现。所有数字均为本次审计自行重算（重跑 pytest、重跑交付 benchmark、自写收敛探针），不引用 m21003 报告中的数值作为证据。
- **裁定**：**L2（独立功能实现）**。L3 证据不足（无集成调用方）；L4 拒绝（基准表无该 anchor 条目、无 run 工件、repro_cmd 在 MRT 分支崩溃、交付配置下指标 11% > 3%）。

## 1. 交付三件套核验

| 件 | 要求 | 实测 | 结论 |
|---|---|---|---|
| run_artifacts | 运行工件（结果 JSON/日志/场量） | 交付树内 `benchmarks/results/` **不存在**；`main()` 在 MRT 分支抛错，写盘语句（`benchmarks/poiseuille.py:140-142`）从未执行；报告 `figures: []` | **缺失** |
| benchmark ref vs got | 基准表参考值 + 实测值 | `tensor_benchmark_cases` 中 `anchor_slug='XVAL-LBM-MASS-EXCHANGE-SELECTOR'` 行数 = **0**（`ILIKE '%MASS-EXCHANGE%'` 亦为 0）→ 无容差、无裁判；交付报告自述 Poiseuille 11% vs 自设 3% 门槛 | **无基准判据** |
| repro_cmd | 可执行复现命令 | `cd TensorLBM && python3 benchmarks/poiseuille.py` → BGK 完成，**MRT 处 `RuntimeError: expected m1 and m2 to have the same dtype, but got: double != float`，exit=1**（`audit/artifacts/repro_failure.log`） | **不可端到端执行** |

repro 崩溃的根因是交付代码自身的 dtype 缺陷，不是环境问题：
`benchmarks/poiseuille.py:79` 用 `torch.ones(Ny, Nx)`（默认 float32）构造 `f_init`，而 `collision.py:58-68` 的 `M` 与 `M_inv` 是 float64，`M @ f_flat` 因此报 dtype 冲突。单元测试不受影响，因为测试里显式传 float64 场（`tests/test_mass_exchange_selector.py:16-23` 的 `small_field` 夹具显式 float64）。m21003 报告"注意点 3"提到了该 bug，但仍在同一提交里交付了会崩溃的入口。

## 2. 报告六段核验

`TensorLBM/report_m21003.json` 六段齐全（1 案例介绍 / 2 方法 / 3 计算结果 / 4 结果图 / 5 注意点 / 6 误差判定）。
缺陷：第 4 段为"无可视化场图"占位，`figures: []` —— 本算例有场量与解析解，属可出图而未出图；报告对结论的表述是诚实 FAIL（自报 11%），未粉饰，这点予以确认。

## 3. 复现与独立重算

### 3.1 单元测试（独立重跑）
`python3 -m pytest tests/ -q` → **8 passed in 4.25s**（`audit/artifacts/pytest_rerun.log`）。报告"8/8 PASS"**属实**。
测试断言的是物理不变量（质量漂移 < 1e-12、x/y 动量漂移 < 1e-10、feq 反演宏观量、周期 streaming 质量守恒），不是"能跑通"式空断言。

### 3.2 交付配置（Nx=64, Ny=32, τ=0.8, F=1e-5, n_steps=2000）
审计方复算：BGK rel-L2 = **10.9985%**，TRT = **10.9452%**，质量漂移 0.0；MRT 崩溃（同上）。报告"~11%"**属实且可复现**。

### 3.3 关键发现：11% 主要来自"没跑到稳态"，不是格式误差
稳态扩散时标 H²/ν = 31²/0.1 ≈ 9.6×10³ 步，交付配置 2000 步仅约 20%。审计方续算：

| steps | rel-L2 vs 节点中心解析解 | rel-L2 vs 半程反弹壁面解析解 | u_max/解析 |
|---|---|---|---|
| 2000（交付） | 10.998% | 5.428% | 0.852 |
| 5000 | 2.787% | 3.568% | 0.969 |
| 10000 | 2.449% | 3.900% | 0.974 |
| 16000 | **2.447%** | 3.900% | 0.974 |

数据：`audit/artifacts/poiseuille_convergence.json`；脚本 `audit/probe_poiseuille_convergence.py`（不改交付代码，仅改步数）。
结论（可复现，非推测）：交付报告的 11% 是**基准配置欠收敛**导致的，收敛后同一实现在其自设指标（节点中心抛物线）下为 **2.45% < 3%**；若改用物理上更严格的半程反弹壁面参考，则为 **3.90% > 3%**。因此"<3%"是**指标敏感的**，不能无条件宣称。

### 3.4 三方案一致性（实现是否只是壳）
float64 场、6000 步：BGK rel-L2 = 2.5648%，MRT = **2.5936%**（u_max/解析 0.9727 / 0.9724），TRT 在 2000 步为 10.9452%，与 BGK 收敛态同量级。MRT 在给定 float64 场时**能跑且物理正确**，说明三种碰撞算子是真实实现，不是 MOCK（`audit/artifacts/scheme_parity.txt`）。
MRT 的 D2Q9 变换矩阵 `collision.py:58-68` 与 Lallemand–Luo 标准矩基逐行一致；TRT 的 magic number 关系 `(τ₊-½)(τ₋-½)=Λ` 在 `mass_exchange_selector.py:135-137` 中自洽。

## 4. 严重度排序的发现

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | **阻塞** | 基准表无该 anchor 条目 → 按"基准判据是唯一裁判"，本单**没有任何可机器复核的通过判据**，L4 无从成立 | `SELECT count(*) ... = 0`；`tensor_benchmark_cases` 全表 ILIKE '%MASS-EXCHANGE%' = 0 |
| F2 | **阻塞** | 交付入口 `benchmarks/poiseuille.py` 在 MRT 分支崩溃（dtype），写盘从未执行 → 三件套的 run_artifacts 不存在，复现命令 exit=1 | `audit/artifacts/repro_failure.log`；`poiseuille.py:82` × `collision.py:59` |
| F3 | 高 | 报告 11% FAIL 归因不准：真实原因是 n_steps=2000 欠收敛（稳态需 ~1e4 步），不是报告所称的"边界一阶精度 + Guo 体力未实现"为主因；收敛后同实现 2.45%（自设指标） | 3.3 节复算表 |
| F4 | 中 | 精度结论指标敏感：节点中心 2.45% vs 半程反弹壁面 3.90%，跨越 3% 门槛；交付报告未声明用的是哪一种壁面位置约定 | `poiseuille_convergence.json` 两列 |
| F5 | 中 | 无集成调用方：仓内 (`tensordg/`、`tools/`、平台路由) 对 `tensorlbm` / `MassExchangeSelector` 的引用 **0 处**（TensorLBM 目录外 grep 全仓） → L3 不成立 | grep 输出为空 |
| F6 | 中 | external anchor（Maarten-vd-Sande/lbm）**未建立**：无上游文件/版本号/逐函数对照或数值对照工件，仅文档声明"implements the pattern" | 交付树内无 upstream 引用工件 |
| F7 | 低 | 报告第 4 段无图（`figures: []`），本算例完全可出图 | `report_m21003.json` |
| F8 | 低 | 文档 `docs/mass_exchange_selector.md` 与实际行为不一致处：宣称 3 方案可用，但 MRT 在默认 benchmark 路径上直接崩 | `docs/mass_exchange_selector.md:47-70` vs repro 日志 |
| F9 | 低 | 包 docstring 宣称 `(BGK, MRT, entropic)` 三种交换方案，实际 `Scheme` 只有 BGK/MRT/TRT，entropic 未实现（文档超前） | `src/tensorlbm/__init__.py:4` × `mass_exchange_selector.py:19-23` |

未发现：造假数据、硬编码结果、报告数字与代码不符（报告自报 8/8、~11% 均经独立重跑证实）。

## 5. 判级

| 级别 | 判据 | 裁定 |
|---|---|---|
| L0 文档 | — | 超过 |
| L1 数学/单元验证 | 守恒律与算子级断言 | **满足**（8 个物理不变量测试，独立重跑通过） |
| L2 独立功能实现 | 真代码：输入→计算→输出；无硬编码 | **满足**：三种碰撞算子 + D2Q9 算子 + streaming；输出随物理参数变化（3.2/3.4 节），MRT/TRT 非空壳 |
| L3 系统集成 | 接入主求解器/平台 | **不满足**：零调用方（F5） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 + 注册基准 | **不满足**：无基准条目（F1）、入口崩溃（F2）、交付配置 11% > 3% |
| L5 科研工程成果 | — | 不适用 |

**裁定：L2。** 报告页数与图不改变级别；级别由功能实质决定。

## 6. 返修清单（reopened 时应满足）

1. 修 `benchmarks/poiseuille.py:79` 的 float32/float64 不一致（`torch.ones(..., dtype=torch.float64)`），并把 `main()` 改为逐方案 try/except，保证结果 JSON 落盘；
2. 基准配置改为收敛判据（建议 `|Δu_max|/u_max < 1e-4` 或 n_steps ≥ 2×10⁴ @ Ny=32），并在报告中显式声明壁面位置约定（节点中心 / 半程反弹）；
3. 在 `tensor_benchmark_cases` 注册 `XVAL-LBM-MASS-EXCHANGE-SELECTOR` 用例（metric = Poiseuille rel-L2，tolerance = 0.03，声明参考解约定），使门禁可机器裁定；
4. 补 external anchor 对照工件（上游 commit + 选择器语义对照表）；
5. 增加一个真实调用方（TensorLBM 求解器循环或平台 benchmark 路由）方可申报 L3；
6. 补结果图（剖面 + 收敛曲线）。

## 7. 未执行的检查（不得视为通过）

- 与上游 `Maarten-vd-Sande/lbm` 的 API/数值逐项对照：未执行（无工件、无网络取证条件），标记 `not_run`。
- D3Q19/D3Q27、复杂几何、非周期性边界：交付即声明未实现，未测。
- entropic 碰撞：包 docstring 列出但 `Scheme` 枚举中不存在，未实现，未测。
- Guo 体力方案 `_apply_guo_force`（`mass_exchange_selector.py:191`）在求解器路径中未被使用（benchmark 用的是简化 `w·3e_x F`），其正确性**未验证**。
- 长时稳定性（>10⁵ 步）、多核/GPU 一致性：未测。

## 8. 处置

**request changes（reopened）**。L2 予以确认并落表；L3/L4 拒绝；返修项见第 6 节。m21003 报告的诚实 FAIL 表述本身合格，问题在于"以欠收敛配置作为基线"与"无注册基准"两项结构性缺陷。

## 复现命令

```bash
# 审计方副本（不改交付代码）
cd audit/refs/TensorLBM_m21003
python3 -m pytest tests/ -q                      # 8 passed
python3 benchmarks/poiseuille.py                 # exit=1, MRT dtype RuntimeError
cd ../../ && python3 audit/probe_poiseuille_convergence.py   # 收敛探针, 75 s
python3 audit/make_figures.py                    # audit/artifacts/fig_poiseuille_convergence.png
```
## 9. 落表与门禁状态

- 落表：`tensor_maturity_verdicts#872`，`repo=TensorDG`，`scope='m21003 XVAL-LBM-MASS-EXCHANGE-SELECTOR'`，`mission_id=21040`，`level=2`，`verdict_by='board-audit'`（`audit/insert_verdict.py`，幂等）。
- 门禁：`maturity_gate.py`（pid 1885，30 min 周期）在 23:15:58 的周期未摄入本 verdict（`CYCLE ... total 0`）——该守护只处理 **审计单已 succeeded** 的 verdict，本单仍在执行中，故 `gate_state=pending` 属预期，非失败。
- 预演：用守护自身的判据函数 `maturity_gate.judge(level, evidence, 'TensorDG', 21040)` 对落表 evidence 逐键复核 → `(2, [])`（L2 全过）；`judge(3, …)` → `L2 缺 integration_callers`；`judge(4, …)` → `L2 缺 integration_callers, repro_cmd 不成立`。与审计结论一致：**gate_level=2 confirmed**，L3/L4 不成立。
