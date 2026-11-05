# TensorDG 壁面诊断口径欠账结清：三处诊断调用点与生产算子不一致（m3524）

> 类型：**验证欠账**（诊断工具/基准的调用口径 vs 已合并生产算子）· 仓 TensorDG · 2026-11-04
> 上游欠账：`docs/reports/tensordg-ns-wall-gradient-fix.md`（m3415）§5.5 明写
> 「`tools/diag_ns_wall.wall_fluxes` 对下壁给出 25.4%，生产路径给 3.7%，是否把下壁当 `right`
> 传参未定论」。本单把它结清：**不是算例问题，是调用口径问题，且一共三处。**

## 0. 结论摘要

| 编号 | 结论 | 证据 |
|---|---|---|
| F1 | m3415 §5.5 的 25.4% 复现成功：它是**修正前置罚项符号**（`outward_sign=+1`，m2782 之前）的下壁值，与生产路径（−1，3.734%）差 6.8 倍 | `validation/ns_wall_diag/convention_audit.json` |
| F2 | `tools/diag_ns_wall.wall_fluxes` 的 **fd2 / br1 行是假的**：从未传 `centre=`，`_boundary_flux` 静默退回一阶面导数，因此 fd2/br1 行与 face 行**逐位相同**（mislabelled mode，不是测量） | 同上 + `tests/test_ns_wall_diag_binding.py` |
| F3 | 若把 `centre=` 接上但不改 `side_name`，下壁 fd2 通量**符号翻转**，误差 200.3–202.5 %（mb 级网格无关的常数倍），正确标注后 0.303–2.472 % | 同上 |
| F4 | `benchmarks/verify_ns_wall_gradient.py` 的下壁调用缺 `outward_sign=-1`（且带 m2879 `neighbour=` 原型），其发布数字 25.41/12.69/6.34/3.17 % 属**退休算子**；同配置生产口径 3.734/1.841/0.914/0.455 % | 同上 |
| F5 | 三处均已修，并用 5 条新测试把「诊断口径 ≡ 生产口径」钉成回归（含负控：错标注必须给出符号翻转） | `pytest tests/test_ns_wall_diag_binding.py` 5 passed |

## 1. 案例与判据（物理背景 / 几何 / 参数）

解析稳态 Couette + Brinkman 温度抛物线（`tools.diag_ns_wall.build_state`）：$\mu=0.1$，
$U=1.0$，$T_w=1.0$，$p=1.0$，则
$T(y)=T_w+\frac{\mu U^2}{2\kappa}y(1-y)$，壁面热流解析真值 $-\kappa\,\partial_yT|_0=-0.05$。
几何 $[0,1]^2$，x 周期、y 双壁，$2\times n_y$ 单元、模态 Legendre $P_1$，罚参数
$\sigma=2(p+1)^2=8$（fd2 走 `WALL_PENALTY`=0.0）。

判据（先于动手）：①修正后每个 mode 的取值必须与生产 `ns._axis_fluxes_ns` 的关键字集合逐位一致（1e-12）；
②同一 mode 的 `side_name="right"/"left"` 必须给出**互为相反数**的通量（负控）；
③fd2 行必须与 face 行可区分（否则证明工具在偷懒）；④发布数必须可复现（25.40824642499248 %）。

## 2. 怎么做的

1. 复现：按 m3415 的两种口径直接调 `ns._boundary_flux`（`side_name` × `outward_sign` × mode 的 8 种组合），
   定位 25.4% 的来源。
2. 修 `tools/diag_ns_wall.py::wall_fluxes`：fd2/br1 传 `centre=`、`side_name="left"`、`side_name` 语义写入 docstring，
   fd2 用 `ns.WALL_PENALTY`。
3. 修 `benchmarks/verify_ns_wall_gradient.py::_row`：默认改为生产口径（`outward_sign=-1, side_name="left"`），
   旧口径保留为 `_legacy_row` 负控。
4. 绑测试：`tests/test_ns_wall_diag_binding.py`（5 条），把工具↔生产↔发布数的三条链路钉死。

## 3. 计算结果（数字由 `validation/ns_wall_diag/convention_audit.json` 直读）

下壁热流相对误差 [%]，$2\times n_y$、P1、$\sigma=8$：

| ny | retired_err_pct | face_err_pct | taylor_err_pct | fd2_err_pct | fd2_wrong_side_err_pct |
|---:|---:|---:|---:|---:|---:|
| 8 | 25.408 | 3.734 | 14.519 | 2.472 | 202.472 |
| 16 | 12.690 | 1.841 | 7.253 | 1.222 | 201.222 |
| 32 | 6.345 | 0.914 | 3.626 | 0.608 | 200.608 |
| 64 | 3.173 | 0.455 | 1.813 | 0.303 | 200.303 |

观测阶：face 1.005 / fd2 1.004 / retired 1.0（64 格相邻）。
fd2 的剩余 $O(h)$ 来自 Dirichlet 迹误差经罚项放大（m3415 §5.3 同结论），不是 stencil 缺陷。

m2879 `neighbour=` 原型（`benchmarks/verify_ns_wall_gradient.py` 实测，生产默认不启用）：
修正符号后 8.478 %，旧符号 37.620 %（2×8）。工具修好后 fd2@2×8 =
-0.05123600（2.472 %）。

![图](figures/ns_wall_diag_convention.png)

左：四种口径的壁面热流误差随网格；右：fd2 在 `side_name` 两种标注下的误差（对数轴，符号翻转 ⇒ 200 %+）。

## 4. 注意点（坑 / 局限 / 适用范围）

1. **静默 fallback 比报错更危险**：`centre=None` 不是错误，它让 fd2 退化成一阶面导数，
   于是工具「跑得通、有数字、还很好看」。凡 mode 参数，必须有一条「不同 mode 必须给出不同数」的测试。
2. **`side_name` 默认值 `"right"` 是陷阱**：它只影响 fd2 stencil 方向，不影响 face 路径，
   因此只测 face 的套件永远发现不了（这正是 m3415 定不了论的原因）。
3. **诊断数字必须带口径**：本单把「25.4 %」拆成「退休算子 25.408 %」「生产 face 3.734 %」
   「m2879 原型 8.478 %」三个不同对象——引用历史数字时必须说明是哪一个。
4. 局限：只覆盖 Steady Couette/Brinkman 状态、Cartesian、常 $\mu$、P1；
   `_row` 的 `neighbour=` 原型仍非生产默认（已在新 docstring 与测试里标注，未做生产化）。

## 5. 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 基准 | 本单结果 | 相对误差 | <3%？ |
|---|---|---:|---:|---|
| 下壁热流，face，2×64 | −0.05 | -0.049772 | 0.455% | 是 |
| 下壁热流，face，2×32 | −0.05 | -0.049543 | 0.914% | 是 |
| 下壁热流，face，2×16 | −0.05 | -0.049080 | 1.841% | 是 |
| 下壁热流，face，2×8 | −0.05 | -0.048133 | 3.734% | **否（3.734%）** |
| 下壁热流，fd2，2×32 | −0.05 | -0.050304 | 0.608% | 是 |
| 下壁热流，fd2，2×8 | −0.05 | -0.051236 | 2.472% | 是 |
| 工具↔生产口径一致（5 种 mode） | 逐位相同 | 差 < 1e-12 | 0% | 是 |
| 负控：错标注符号翻转 | 比值 −1 | 逐网格 −1.000 | 0% | 是 |

本单不改任何数值算法：修的是**诊断/基准的调用口径**，生产求解器逐位不变（`face` 路径未触碰）。

## 6. 验证

```
OMP_NUM_THREADS=2 PYTHONPATH=. python -m pytest tests/test_ns_wall_diag_binding.py -q   # 5 passed
OMP_NUM_THREADS=2 PYTHONPATH=. python -m pytest tests/test_ns_dg_2d_wall_scope.py -q -k "flux or residual or scope or table"  # 10 passed
```

工件：`validation/ns_wall_diag/convention_audit.json`、`docs/figures/ns_wall_diag_convention.png`；
本文数字全部由 `validation/ns_wall_diag/convention_audit.json` 程序化注入（零手抄）。

## 7. 下一次自找活的建议

1. **`march`/`probe_wall_sign` 的 monkeypatched `channel_masked` 仍无 `neighbour`/`wall_adjoint` 分支**——
   与生产 `_boundary_flux` 的签名漂移同类，属同一族欠账（本期只修了 `wall_fluxes` 与 gradient 基准）。
2. 把 `neighbour=`（m2879 二阶迹导数外推）真正接进生产默认并复测设计阶，或明确退役该原型。
3. 用本单的「诊断≡生产」绑法回头审 `tools/spectrum_ns_wall.py`（它自带一份 `_boundary_flux` 复刻）。
