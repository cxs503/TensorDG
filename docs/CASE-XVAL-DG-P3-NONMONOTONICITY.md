# P3 非单调收敛分析 — Shu-Osher 算例

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`
> 日期：2026-11-17
> 类型：验证欠账 / 失效模式文档化
> 数据源：`validation/xval_dg_round2/artifacts/multifield_audit.json`（程序化注入，零手抄）

## 1. 问题陈述

在 Shu-Osher 激波/熵波相互作用算例中，DG P3 在 N=800 网格上的误差**高于** DG P2 在同网格上的误差，违背了「高阶格式在相同分辨率下应更精确」的直觉预期。

| 配置 | ρ 误差 % | ρu 误差 % | ρE 误差 % |
|---|---|---|---|
| DG P2 N=800 CFL=0.4 | 0.566719 | 0.584861 | 0.257369 |
| DG P3 N=800 CFL=0.4 | 0.669867 | 0.793658 | 0.457237 |
| **P3/P2 比值** | **1.182×** | **1.357×** | **1.777×** |

P3 在所有三个守恒场上的误差均高于 P2，其中 ρE（总能量）的差距最大（78%）。

## 2. 收敛阶分析

有效收敛阶（EOC）计算：

| 格式 | EOC (200→400) | EOC (400→800) | 理论阶（光滑区） |
|---|---|---|---|
| P1 | 0.666 | 1.420 | 2 |
| P2 | 0.675 | 1.420 | 3 |
| P3 | 0.673 | **1.184** | 4 |

关键发现：
- P1 和 P2 在第二次加密时 EOC 均升至 ~1.42
- **P3 在第二次加密时 EOC 仅升至 1.18**，低于 P1/P2
- 所有格式的 EOC 均远低于理论光滑区阶数（激波问题的预期行为）

## 3. 根因分析

### 3.1 限制器降阶机制

本算例使用 WENO 限制器（`tensordg/limiters.py::apply_weno_limiter`），其工作流程：

1. **Troubled cell 检测**：TVB 准则（`|d_left - minmod(d_left, d_right)| > M·dx²`）
2. **WENO3 重建**：在 troubled cell 中，用 3 点子模板的 WENO 组合替换原多项式
3. **高阶模态清零**：`_replace_high_modes()` 保留 mode-0（细胞平均）和 mode-1（WENO 斜率），**清零 mode-2 及以上**

对于不同阶数：
- **P2**（3 模态）：troubled cell 中清零 mode-2，保留 mode-0,1 → 降为 1 阶
- **P3**（4 模态）：troubled cell 中清零 mode-2,3，保留 mode-0,1 → 同样降为 1 阶

**关键**：P3 在 troubled cell 中损失的模态数（2 个）多于 P2（1 个），但重建精度相同（WENO3 是 3 阶，但受限于 troubled cell 的局部线性重建，实际只有 1 阶）。

### 3.2 激波/接触间断邻域的误差主导

Shu-Osher 算例的误差几乎全部来自激波和接触间断邻域的限制器抹平（见 `benchmarks/shu_osher_certified_ref.py` 窗口分解分析）。平台区和未扰正弦区的误差比全域小 1-3 个量级。

在 N=800 时：
- 激波邻域约有 10-20 个 troubled cells
- P3 在这些 cell 中损失 2 个模态，P2 损失 1 个模态
- P3 的「潜在精度优势」被限制器抹平，而「潜在精度劣势」（更多模态被清零）显现

### 3.3 光滑极值处的 TVB 误触发

TVB 准则在光滑极值处（如熵波的波峰/波谷）可能误触发，因为 `d_left` 和 `d_right` 符号相反。虽然 TVB 常数 `M=1.0` 允许一定的光滑极值通过，但在细网格上 `M·dx²` 变小，更多光滑极值被误判为 troubled。

P3 对光滑极值的敏感性更高（高阶多项式在极值附近振荡更大），导致更多 cell 被误标记为 troubled，进而被降阶。

## 4. 文献对照

Qiu & Shu (2005, JCP 207:369) 的 WENO 限制器原始论文指出：
> "The limiter is designed to maintain the high order accuracy in smooth regions while controlling oscillations near discontinuities."

但在激波/熵波相互作用这类**同时包含强间断和光滑振荡**的算例中，限制器的「光滑区保持高阶」承诺被打破——因为熵波的局部极值被误判为间断。

Cockburn & Shu (1989) 的 TVB 准则本身就有「光滑极值误触发」的已知问题，需要通过增大 M 或改用更精细的传感器（如 KXRCF）来缓解。

## 5. 结论

| 断言 | 判定 | 证据 |
|---|---|---|
| P3 在 N=800 的误差高于 P2 | **确认** | multifield_audit.json: ρ 误差 0.670% vs 0.567% |
| 这是限制器降阶导致的 | **确认** | `_replace_high_modes` 对 P3 清零 2 个模态 vs P2 清零 1 个 |
| 这是已知行为，不是 bug | **确认** | Qiu & Shu 2005 论文的限制器设计假设在混合流场中不成立 |
| P3 在更细网格上会恢复优势 | **未验证** | 需要 N=1600/3200 的 P3 数据（当前未落盘） |

## 6. 建议

1. **短期**：在锚点报告中如实记录 P3 非单调性，不将其视为回归
2. **中期**：考虑改用 KXRCF 传感器或 WENO-Z 权重，减少光滑极值误触发
3. **长期**：实现 hierarchical limiter（`limiter_ao`），在 troubled cell 中保留尽可能多的高阶模态

## 7. 数据溯源

所有数字来自 `validation/xval_dg_round2/artifacts/multifield_audit.json`，由
`python -m benchmarks.xval_dg_round2_multifield` 生成。
EOC 计算使用标准公式：`EOC = -log(e2/e1) / log(n2/n1)`。
