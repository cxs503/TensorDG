# TensorDG PR #100 CI 红修复：分支带回的「陈旧等温壁钉疤」（2026-11-05 / m4361）

修复对象：PR #100，分支 `agent/m3714`，head `4c4138f`（"m3714: matured mission deliverables"）。
症状：CI `test=failure`。
落盘修复：分支树内 `tests/test_ns_heat_wall_closure.py` 采用 main 在 `c051cef`
（PR #74 / m3694）里已评审通过的修复版钉疤；产品代码 0 行改动。

## 1. 案例介绍

物理背景：`[0,1]^2` 上两块等温壁之间的静止流体纯热传导（`T_bottom=1.0`、
`T_top=0.5`，`p=1`，`mu=0.1`，`Pr=0.71`，`R=1`），强形式 BR1/SIPG 间断 Galerkin
离散、DG 阶 `p=1`、壁法向导数取默认 `"face"`（单边 DG 导数）。
`tensordg.ns_dg_2d.solve_heat_conduction_dg_2d` 是该模块唯一从未闭合验证的公开
入口，其壁面闭合与崩溃史见 `docs/reports/tensordg-ns-heat-conduction-verification.md`
与 `docs/reports/tensordg-ns-heat-wall-closure.md`；main 侧的同一红测已在
`docs/reports/tensordg-main-ci-rca-ns-heat-2026-11-04.md`（m3760）里诊断并翻正。

| 参数 | 值 |
|---|---|
| 域 | `[0,1]^2`，`nx x ny` 张量积网格，`nx=2` |
| 阶 | `p=1`（`legendre_operators(1, max(p+2,3))`） |
| 边界 | `ymin` 等温 `T=1.0`，`ymax` 等温 `T=0.5`，`xmin/xmax` 无（一维剖面问题） |
| 输运 | 无对流，纯导热；`mu=0.1`，`Pr=0.71`，`kappa = mu*cp/Pr` |
| 初值 | 精确导热态（线性温度剖面）→ 残差即壁面闭合缺陷 |

## 2. 怎么做的

1. 把工作树从 `origin/agent/m3714`（`4c4138f`）切出，**逐文件跑**全部 53 个测试
   文件复现红：只有 `tests/test_ns_heat_wall_closure.py` rc=1（2 failed / 3 passed），
   其余 52 个文件 rc=0（其中 3 个主文件在满载 400+ 的共享机上超时，见 §5）。
2. `git merge-tree` 确认 PR 合入 main **无文本冲突**，但该文件在 merge-base
   `619167f` 之后就同时被两侧改写：main 于 `c051cef` 翻正钉疤，分支保留更早的
   严判据副本。逐行 `diff` 定位两个已失效的钉：
   - `assert bottom_ratio < 1.1`（16/32 冷壁能量残差比），实测 1.1254；
   - `with pytest.raises(FloatingPointError)`（入口点按默认 `final_time=40` 崩溃），
     m3139 壁面修复落地后该崩溃不再发生。
3. 复算而非引用：`validation/ns_heat_wall_closure/pr100_repro.py` 用测试自带
   helper 从零重算 16/32/64 的残差与入口点解（float64/CPU），把「陈旧钉必失败、
   修复钉必通过」写成机器可判的 JSON（`pr100_repro.json`）。
4. 修复只改一条：分支树内该测试文件采用 main 的修复版（同一物理判据，边界由实测
   值加 6% 余量定；新增单调停滞与 O(1) floor 两条更强钉，并把崩溃钉换成
   「入口点跑完 + 物理性」）。产品代码 0 行改动，无任何无关文件改动。
5. 证据图/表由 `validation/ns_heat_wall_closure/pr100_figure.py` 程序化生成
   （`docs/figures/tensordg-pr100-wall-pin.png`、`pr100_table.md`）。

## 3. 计算结果

### 3.1 精确导热态上的能量残差随细化（`wall_gradient="face"`，`p=1`）

| ny | cold-wall $\max|R_E|$ | wall ratio vs prev | interior $\max|R_E|$ | interior ratio vs prev |
|---|---|---|---|---|
| 8 | 1.9561e-01 | -- | 2.9398e-01 | -- |
| 16 | 1.5739e-01 | 1.2429 | 1.7093e-01 | 1.7199 |
| 32 | 1.3985e-01 | 1.1254 | 9.2912e-02 | 1.8397 |
| 64 | 1.3143e-01 | 1.0640 | 4.8554e-02 | 1.9136 |

Stale pin `wall_ratio(16/32) < 1.1` against measured **1.1254006471** -> fails by 2.31 %.
Repaired pins (`< 1.2` at 16/32, `< 1.1` at 32/64, monotone, floor > 1e-2) with measured 1.0640333275 -> all hold.

冷壁残差比 1.2429 → 1.1254 → 1.0640（趋 1，停滞，floor 1.3143e-01 为 O(1)）；
内区残差比 1.7199 → 1.8397 → 1.9136（≈O(h)，逐次逼近 p+1 阶）。
即：缺陷是**壁面闭合**，不是全局离散误差。

### 3.2 入口点 `solve_heat_conduction_dg_2d(ncells=(2,8), final_time=0.5)`

| 量 | 值 |
|---|---|
| 步数 | 1070 |
| `p` 范围 | [7.3237698e-01, 9.2602155e-01] |
| `rho` 范围 | [9.1246211e-01, 1.3363162e+00] |
| `theta=p/rho` 热壁 → 冷壁 | 0.9841004 → 0.5616704（y 方向单调） |
| 与精确导热剖面最大偏差 | 1.23e-1（相对 12.3%） |

### 3.3 图

`docs/figures/tensordg-pr100-wall-pin.png`：左图冷壁 vs 内区 `max|R_E|` 的 log-log
细化曲线（带 O(h) 参考线与壁面比值标注），右图入口点 `theta(y)` 对精确导热直线。

## 4. 误差判定

| 关键量 | 基准 | 本次复算 | 相对偏差 | 是否 < 3% |
|---|---|---|---|---|
| 冷壁残差比 16/32 | main `c051cef` docstring 的 1.1254 | 1.1254006471 | < 1e-9 | **是** |
| 冷壁残差比 32/64 | main docstring 的 1.0640 | 1.0640333275 | < 1e-9 | **是** |
| 内区残差比 16/32 | main docstring 的 1.8397 | 1.8397111512 | < 1e-9 | **是** |
| 残差 sweep（ny=8/16/32/64） | main doc `tensordg-ns-heat-wall-closure.md` 表 | 见 §3.1 | 四位有效数字全同 | **是** |
| 入口点 `theta` 对精确导热 | 解析线性剖面 | 最大偏差 1.23e-1 | **12.3%** | **否（如实报告）** |

最后一行不是回归：壁面闭合本身仍是不一致的（§3.1 的 O(1) floor），故入口点瞬态
带 12.3% 偏移。修复版测试**故意不断言入口点精度**，只钉「跑完 + 物理性」，
文档里也写明 completion ≠ verification。本任务不改产品代码，因此不声称修好了
壁面闭合——那是另一个未闭合的科研缺口。

## 5. 注意点与坑

- **陷阱：分支陈旧钉 vs main 翻正**。同一个测试文件被两侧独立改写时，
  `git merge-tree` 显示「无冲突」并不等于语义正确：3-way 文本合并可能把一侧
  的旧判据与另一侧的新 docstring 拼在一起。本次合并结果恰取 main 版，故合并本身
  即修复；但仍逐测试复核后才下结论。
- **共享机负载**：99 控制面 load average 400+（多 agent 并发），
  `test_euler_2d_p0_verification` 单文件 900 s 超时（rc=124）属资源超时而非断言失败；
  该文件分支 0 改动（`git diff 619167f 4c4138f` 为空）。
- **计算集群不可用于本仓 pytest**：loong64 节点无 `torch`/`numpy`，
  52 个测试文件在 collection 阶段即 `ModuleNotFoundError`，重计算只能留在本机。
- 修复边界：只动一个测试文件（沿用 main 已评审版本）+ 新增两个取证脚本与本节文档；
  不碰 `tensordg/` 任何产品代码，不改其他测试阈值。

## 6. 复现

```bash
cd <work tree>
PYTHONPATH=. python -m pytest -q tests/test_ns_heat_wall_closure.py      # 5 passed
PYTHONPATH=. python validation/ns_heat_wall_closure/pr100_repro.py      # 陈旧钉 False / 修复钉 True
PYTHONPATH=. python validation/ns_heat_wall_closure/pr100_figure.py     # 重建图与表
```