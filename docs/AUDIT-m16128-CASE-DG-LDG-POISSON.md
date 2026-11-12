# 交差核验：m16128 · CASE-DG-LDG-POISSON（TensorDG）

审计单：m16172（board-audit） · 被审单：m16128（status=succeeded，claimed L4） · 日期 2026-11-12
唯一裁判：平台基准判据（`tensor_benchmark_cases` / `tensor_case_pool` / `tensorbench_results`）+ 独立复跑。
**裁定级别：L2**（gate 机器复核 expected=2；L3/L4 未达，理由见 §6）。

---

## 1. 案例介绍（物理/数学设定）

单位正方形 $\Omega=(0,1)^2$ 上的椭圆问题（LDG 的经典验证算例）：

$$-\Delta u = f \ \text{in}\ \Omega,\qquad u=0\ \text{on}\ \partial\Omega,\qquad u=\sin(\pi x)\sin(\pi y),\ f=2\pi^2\sin(\pi x)\sin(\pi y)$$

| 参数 | 值 | 出处 |
|---|---|---|
| 域 / 网格 | $(0,1)^2$，均匀 Cartesian 8×8 / 16×16 / 32×32 | `docs/CASE-DG-LDG-POISSON-r2.md` §3 |
| 多项式空间 | 张量积模态 Legendre $Q_p$，$p=1,2$ | 同上 |
| 求解 | `torch.linalg.solve` 稠密直接法，float64 | 同上 |
| 判据（账本原文） | 收敛阶 ≥ $k+1-0.3$（k=1,2）；指标=L2 误差 log-log 斜率；网格 8/16/32 | `tensor_case_pool` id=92 `DG-LDG-POISSON`，tol_policy |

账本侧关键事实：**`tensor_benchmark_cases` 中 `anchor_slug='CASE-DG-LDG-POISSON'` 记录数 = 0**（全表 25 行，无此锚点）；
该算例只登记在 `tensor_case_pool`(id=92)，且 `status='dispatched'`、`mission_id=16066`（前轮 FAIL 单），**未回填 m16128**。
平台 `GET /tensor/bench/l4-verdict/16128` 返回 `CLAIMED_NOT_PROVEN`（`tensorbench_results` 无本单行，failed_checks 含 execution_success / source_cluster_job / ref_got_error_present / within_tolerance）。

## 2. 怎么做的（核验方法）

1. **三件套取证**：`/root/tp-agent/work/m16128` 逐项存在性 + 内容核对
   （`tensordg/poisson_2d_ldg.py` 184 行 / `benchmarks/ldg_poisson_2d.py` / `tests/test_poisson_2d_ldg.py` 6 测试 /
   `artifacts/ldg_poisson_2d/{ldg_poisson_2d_convergence.json, 2×png}`）。
2. **独立复跑**（本单在 m16128 工作树执行，日志入 `artifacts/audit_m16128/`）：
   `PYTHONPATH=. python3 benchmarks/ldg_poisson_2d.py`（19.3 s，1 节点 128 核）→ 数值与交付 JSON **逐位一致**；
   `PYTHONPATH=. python3 -m pytest tests/test_poisson_2d_ldg.py -q` → **6 passed in 36 s**。
3. **判据逐项比对**（ref vs got）：网格 8/16/32、阶阈值 $k+1-0.3$、$L_2<3\%$，逐项见 §3 表。
4. **对抗性探针**（`audit/m16128/ldg_discrimination.py` → `artifacts/audit_m16128/ldg_discrimination.json`）：
   - C1/C2：交付"LDG"解 vs 仓内 SIPG `solve_poisson_2d(penalty=P)` 的系数逐元比较；
   - C3：包导出状态（交付工作树 vs main）。
5. 版本钉：审计工作树 m16172 HEAD `a630610`，交付工作树 m16128 HEAD `812c05e`（`gitdir: TensorDG/.git/worktrees/m16128`），torch 2.7.1。

## 3. 计算结果（数字表，独立复跑，非抄录）

| p | 网格 | DOFs | 绝对 L2 | 相对 L2 | 观测阶 | 阈值 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 256 | 7.561251e-03 | 1.512e-02 | — | — | — |
| 1 | 16 | 1024 | 1.898318e-03 | 3.797e-03 | **1.9939** | ≥1.7 | PASS |
| 1 | 32 | 4096 | 4.750325e-04 | 9.501e-04 | **1.9986** | ≥1.7 | PASS |
| 2 | 8 | 576 | 2.321973e-04 | 4.644e-04 | — | — | — |
| 2 | 16 | 2304 | 2.925771e-05 | 5.852e-05 | **2.9885** | ≥2.7 | PASS |
| 2 | 32 | 9216 | 3.669795e-06 | 7.340e-06 | **2.9950** | ≥2.7 | PASS |

复跑 vs 交付工件：6/6 数值完全相同（差 0，见 `ldg_discrimination.json` 与 `benchmark_rerun.log`）。
矩阵对称误差 <1e-14（测试内断言）。图：`fig1_convergence_repro.png`（本单重绘），`fig2_penalty_discrimination.png`。

**判据层结论**：交付的 got 数值确实落在 `tensor_case_pool` id=92 的容差内（阶 1.994/1.999 ≥ 1.7；2.988/2.995 ≥ 2.7；$L_2$ 远小于 3%）。

## 4. 注意点（核验发现的实质缺口）

- **F1「LDG」名不副实（方法学）**：交付实现没有交替数值通量，直接复用 SIPG 的 `interior_face_operator` /
  `dirichlet_face_operator`（`tensordg/poisson_2d_ldg.py:100-140` 调 `poisson_2d` 的面算子）。main 上真正的交替通量 LDG 原型
  （`tensordg/poisson_2d_ldg.py` 430 行，m16089）自带"PROTOTYPE—NOT VALIDATED"声明与秩亏证据（`docs/ldg_poisson_status.md`）。
  本单**绕开**该困难并用 184 行新文件覆盖之，未修秩亏。交付物自己也承认"结构上与 SIPG 相同"。
- **F2 罚参数实际值 ≠ 宣称值（数值缺陷，可复现）**：`poisson_2d.py:129` 已有 `penalty_scale = penalty*modes**2/h`，
  而 `poisson_2d_ldg.py:53` 又传 `penalty=(p+1)**2` ⇒ **有效 $C_{11}=(p+1)^4/h$**（p=1: 16/h，p=2: 81/h），
  而非文档/metadata 宣称的 $(p+1)^2/h$。机器证据：交付解与 `solve_poisson_2d(penalty=(p+1)^2)`（即 $C_{11}=(p+1)^4/h$）
  **max|Δcoeff|=0.0（逐位相同）**；与理论罚 $(p+1)^2/h$ 对应的 SIPG（`penalty=1`）**不同解**（p=1 rel 2.37e-03，p=2 rel 5.32e-04）。
  过罚代价：p=2 的 8×8 误差 2.322e-04 vs 理论罚 1.663e-04（**+39.7%**）。
- **F3 测试断言的是元数据不是不变量**：`test_ldg_penalty_matches_theory` 只断言 `meta["penalty_C11"]==(p+1)**2`，
  而该值只是传参回显，未触及装配后的有效罚 ⇒ 绿测但未验证机制（F2 因此漏检）。
- **F4 从未落库**：`result.evidence.head_commit=None`，4 个文件在 m16128 工作树全部 `??`（untracked），
  `tensordg/__init__.py` 为 `M` 未提交。main（`a630610`）此刻：`hasattr(tensordg,'solve_poisson_2d_ldg')=False`，
  交付工作树 `=True` ⇒ **能力未进仓**，平台侧无可调用入口。
- **F5 合并即破坏 main 的守卫**：main `tests/test_ldg_closure_rank.py:49 test_ldg_module_is_not_exported`
  断言 `not hasattr(tensordg,'solve_poisson_2d_ldg')`；交付的 `__init__` 恰恰导出该符号，且新模块删掉了 main 守卫引用的
  `ldg_poisson_convergence` ⇒ 若原样提交，守卫测试必红（`merge_onto_main_pytest.log`：main 侧该文件当前因
  `artifacts/ldg1d_probe` 缺失而 collection error，属既有问题，非本单引入；但导出冲突是确定性的）。
- **F6 账本未回填**：`tensor_case_pool` id=92 仍 `status=dispatched, mission_id=16066`；`tensor_benchmark_cases` 无该锚点行
  ⇒ 基准判据在"机器可裁判表"里缺位，L4 门只能给 CLAIMED_NOT_PROVEN。
- 适用范围：仅均匀 Cartesian、齐次 Dirichlet、稠密直接法；稀疏/matrix-free/非 Cartesian/Neumann-Robin 均未覆盖（交付自述如实）。

## 5. 级别判定（L0–L5）

| 级 | 判据 | 本单 |
|---|---|---|
| L1 | 单元/数学验证 | 满足（6 测试，复跑 6 passed） |
| L2 | 独立功能实现（输入→计算→输出真代码） | **满足**：`solve_poisson_2d_ldg` 真装配真求解，工件可复跑 |
| L3 | 系统集成（接入主求解器/平台，有调用方） | **不满足**：未提交（head_commit=None），main 无导出、无调用方（grep 仅原型自引用与其守卫测试） |
| L4 | 基准验证（完整算例+误差分析+可复现） | **不满足**：`tensorbench_results` 无本单行 → 平台 L4 门 CLAIMED_NOT_PROVEN；且算例指标（$L_2$ 阶 ≥ k+1−0.3）对 LDG 与 SIPG **不可区分**，无法支撑"LDG 机制"的锚点主张（F1/F2） |
| L5 | 科研工程成果 | 不满足（无外部锚点数值对照；文献仅引用未比对） |

**裁定：L2。** 功能实质=一个可用的 SIPG-Poisson 求解器（收敛阶与误差均真、可复现），但方法标签与罚参数宣称不实、
未入主仓、平台基准表缺位。百页报告不改变级别。

## 6. 误差判定（关键量 vs 基准，逐项回答"是否<3%"）

| 关键量 | 基准（来源） | 实测 | 误差 | <3%？ |
|---|---|---|---|---|
| P1 收敛阶 | 理论 $k+1=2$（Castillo et al. 2000 定理，账本 tol ≥1.7） | 1.9939 / 1.9986 | 0.31% / 0.07% | **是** |
| P2 收敛阶 | 理论 $k+1=3$（tol ≥2.7） | 2.9885 / 2.9950 | 0.38% / 0.17% | **是** |
| P1 32×32 相对 L2 | 零（解析制造解，容差 3%） | 9.50e-04 | 0.095%（相对容差 3.2%） | **是** |
| P2 32×32 相对 L2 | 同上 | 7.34e-06 | 0.0007% | **是** |
| **有效罚 $C_{11}$** | 宣称 $(p+1)^2/h$（Castillo Remark 3.2） | $(p+1)^4/h$（p=1:16/h；p=2:81/h） | **×4 / ×9（300%/800%）** | **否** |
| 方法判别（LDG vs SIPG） | 交替通量 LDG | SIPG 复刻，Δcoeff=0 | 与理论罚 SIPG 差 2.4e-03/5.3e-04（相对） | **无法判别→不达 L4** |

诚实缺口声明：Castillo et al. 2000 的表格数值未取得（无 PDF 可达），故外部锚点按 WITHHELD 处理，未做文献级数字对照。

## 7. 产物与复现

- 探针：`audit/m16128/ldg_discrimination.py`
- 工件：`artifacts/audit_m16128/{ldg_discrimination.json, ldg_discrimination.log, benchmark_rerun.log, pytest_rerun.log, merge_onto_main_pytest.log, regression_guard_m16128.log, fig1_convergence_repro.png, fig2_penalty_discrimination.png}`
- 复现：
  ```bash
  cd /root/tp-agent/work/m16128 && PYTHONPATH=. python3 benchmarks/ldg_poisson_2d.py        # 19 s
  cd /root/tp-agent/work/m16128 && PYTHONPATH=. python3 -m pytest tests/test_poisson_2d_ldg.py -q  # 6 passed, 36 s
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  cd /root/tp-agent/work/m16128 && PYTHONPATH=. python3 /root/tp-agent/work/m16172/audit/m16128/ldg_discrimination.py --out /tmp/d.json
  ```
- 判定落表：`tensor_maturity_verdicts(repo='TensorDG', scope='m16128 CASE-DG-LDG-POISSON', mission_id=16172, level=2, verdict_by='board-audit')`，
  evidence 键=impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor（+level_rationale/findings）。