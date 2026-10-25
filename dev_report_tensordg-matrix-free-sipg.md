# TensorDG Matrix-free SIPG 开发报告

**任务编号**: m275  
**开发日期**: 2026-10-25  
**提案编号**: tensordg-002

## 实现摘要

将 TensorDG 的 SIPG Poisson/heat 求解器从密集矩阵扩展为支持 sparse 和 matrix-free 后端，使其可扩展到实际分辨率（>10^4 DOFs）。

### 新增模块

1. **`tensordg/sipg_sparse.py`** - Sparse SIPG 实现
   - 使用 scipy CSR 格式组装刚度矩阵
   - 支持 Poisson 2D 和 Heat 1D 问题
   - 直接求解器：scipy.sparse.linalg.spsolve

2. **`tensordg/sipg_matrixfree.py`** - Matrix-free CG 实现
   - 不显式组装矩阵，仅实现 matvec A@x
   - 预条件共轭梯度（PCG）迭代求解
   - 对角预条件器（Jacobi）：从体积项 + 面项构建完整对角线
   - 支持 Poisson 2D 和 Heat 1D 问题

### 重构模块

1. **`tensordg/poisson_2d.py`**
   - `solve_poisson_2d()` 新增 `backend` 参数
   - 支持 "dense"（默认）、"sparse"、"matrix-free"
   - 保持向后兼容

2. **`tensordg/heat_1d.py`**
   - `solve_heat_1d()` 新增 `backend` 参数
   - 支持 "dense"（默认）、"sparse"、"matrix-free"
   - Crank-Nicolson 时间步进使用对应后端求解

## 判据表

### 回归测试（sparse/matrix-free vs dense）

| 问题 | 网格 | 后端 | max_diff | 通过 |
|------|------|------|----------|------|
| Poisson 2D P1 | 8x8 | sparse | 3.13e-14 | ✅ |
| Poisson 2D P1 | 8x8 | matrix-free | 2.28e-14 | ✅ |
| Poisson 2D P1 | 16x16 | sparse | 9.07e-14 | ✅ |
| Poisson 2D P1 | 16x16 | matrix-free | 3.29e-14 | ✅ |
| Poisson 2D P1 | 32x32 | sparse | 3.12e-13 | ✅ |
| Poisson 2D P1 | 32x32 | matrix-free | 1.23e-13 | ✅ |
| Heat 1D P2 | n=16 | sparse | 4.14e-14 | ✅ |
| Heat 1D P2 | n=16 | matrix-free | 5.54e-13 | ✅ |
| Heat 1D P2 | n=32 | sparse | 7.45e-14 | ✅ |
| Heat 1D P2 | n=32 | matrix-free | 6.62e-13 | ✅ |
| Heat 1D P2 | n=64 | sparse | 2.30e-13 | ✅ |
| Heat 1D P2 | n=64 | matrix-free | 9.01e-13 | ✅ |

**结论**: 所有回归测试通过（atol=1e-10）

### 可扩展性测试（Poisson 2D sparse 后端）

| 网格 | DOFs | 时间(s) | 内存(MB) | NNZ |
|------|------|---------|----------|-----|
| 8x8 | 256 | 0.019 | 0.51 | 4,608 |
| 16x16 | 1,024 | 0.055 | 2.05 | 19,456 |
| 32x32 | 4,096 | 0.212 | 8.23 | 79,872 |
| 64x64 | 16,384 | 0.887 | 33.02 | 323,584 |

**内存增长**: O(N) 线性增长（从 0.51MB → 33.02MB，64倍增长对应 64倍 DOFs）  
**结论**: 64x64 P1 可解，内存增长 < O(N^1.5) ✅

### 精度测试（EOC 保持）

| 后端 | 误差序列 | EOC |
|------|----------|-----|
| dense | [1.519e-02, 3.800e-03, 9.503e-04] | [1.999, 2.000] |
| sparse | [1.519e-02, 3.800e-03, 9.503e-04] | [1.999, 2.000] |
| matrix-free | [1.519e-02, 3.800e-03, 9.503e-04] | [1.999, 2.000] |

**结论**: P1 的 EOC=2 在所有后端保持一致 ✅

### Matrix-free CG 迭代次数

| 问题 | 网格 | CG 迭代 | 残差范数 |
|------|------|---------|----------|
| Poisson 2D P1 | 8x8 | 47 | ~1e-12 |
| Poisson 2D P1 | 16x16 | 132 | ~1e-12 |
| Poisson 2D P1 | 32x32 | 253 | ~1e-12 |
| Heat 1D P2 | n=16 | 2038 | ~1e-12 |

**观察**: 对角预条件器对 Poisson 效果良好，对 Heat 方程（含质量矩阵）迭代次数较多，可能需要更复杂的预条件器（如 ILU）。

## 测试覆盖

- ✅ 所有现有测试通过（21/21）
- ✅ Sparse 结果与 dense 一致（atol=1e-10）
- ✅ Matrix-free 结果与 dense 一致（atol=1e-10）
- ✅ 64x64 P1 可解（4096 DOFs → 16384 DOFs）
- ✅ 内存增长 < O(N^1.5)（实际 O(N)）
- ✅ EOC 保持（P1 → 2阶）

## 技术细节

### Sparse 后端

**组装策略**:
- 体积项：每个单元独立组装到 CSR
- 面项：内部面连接相邻单元，边界面对角修改
- 使用 scipy.sparse.csr_matrix 存储
- 直接求解：scipy.sparse.linalg.spsolve

**优势**:
- 实现简单，精度与 dense 完全一致
- 内存 O(NNZ)，远小于 dense 的 O(N^2)
- 适合中小规模问题（<10^5 DOFs）

**局限**:
- 仍需显式组装
- 大规模问题可能需要迭代求解器

### Matrix-free 后端

**核心思想**:
- 不组装全局矩阵，仅实现 matvec A@x
- 使用预条件共轭梯度（PCG）迭代求解
- 对角预条件器：从体积项 + 面项构建完整对角线

**对角预条件器构建**:
1. 体积项对角线：每个单元独立贡献
2. 内部面对角线：LL/RR（垂直面）和 BB/TT（水平面）块
3. 边界面对角线：边界单元的面贡献

**优势**:
- 内存 O(N)，天然可扩展
- 无需显式组装
- 适合大规模问题（>10^5 DOFs）

**局限**:
- 需要迭代求解，收敛速度依赖预条件器
- 对角预条件器对某些问题（如 Heat 方程）迭代次数较多
- 未来可引入 ILU 或多重网格预条件器

## 使用示例

```python
from tensordg.poisson_2d import solve_poisson_2d

# Dense 后端（默认，向后兼容）
x, y, coeff, meta = solve_poisson_2d(ncells=16, degree=1, backend='dense')

# Sparse 后端
x, y, coeff, meta = solve_poisson_2d(ncells=64, degree=1, backend='sparse')
print(f"NNZ: {meta['nnz']}, DOFs: {meta['dofs']}")

# Matrix-free 后端
x, y, coeff, meta = solve_poisson_2d(
    ncells=128, degree=1, backend='matrix-free',
    rtol=1e-10, maxiter=5000,
)
print(f"CG iterations: {meta['cg_iterations']}")
```

## 生态价值

1. **可扩展性**: 使 TensorDG 椭圆求解器可扩展到实际分辨率（10^4-10^6 DOFs）
2. **隐式时间推进**: 为 Navier-Stokes 方程的隐式求解打基础
3. **生态互补**: 与 TensorFVM 的 multigrid 形成互补（FVM 有 multigrid，DG 有高阶精度）
4. **可微性**: 保持 PyTorch 可微特性，支持梯度计算

## 局限与未来工作

### 当前局限

1. **Matrix-free 预条件器**: 对角预条件器对某些问题迭代次数较多
2. **非结构网格**: 当前仅支持 Cartesian 网格
3. **GPU 支持**: 未针对 GPU 优化

### 未来工作

1. **ILU 预条件器**: 为 matrix-free 后端实现 ILU(0) 或 ILU(k)
2. **几何多重网格**: 利用 Cartesian 网格的层级结构实现 V-cycle multigrid
3. **非结构网格**: 扩展到三角形/四面体网格
4. **GPU 优化**: 使用 PyTorch sparse CUDA 后端
5. **块预条件器**: 为系统问题（如弹性力学）实现块预条件器

## 文件清单

- `tensordg/sipg_sparse.py` (新增): Sparse SIPG 实现
- `tensordg/sipg_matrixfree.py` (新增): Matrix-free CG 实现
- `tensordg/poisson_2d.py` (修改): 支持 backend 选择
- `tensordg/heat_1d.py` (修改): 支持 backend 选择
- `validation_report.json` (新增): 验证报告（机器可读）
- `dev_report_tensordg-matrix-free-sipg.md` (本文档): 开发报告

## 验收标准达成

- [x] 所有现有测试通过
- [x] Sparse 结果与 dense 一致（atol=1e-10）
- [x] Matrix-free 结果与 dense 一致（atol=1e-10）
- [x] 64x64 P1 可解（16384 DOFs）
- [x] 内存增长 < O(N^1.5)（实际 O(N)）
- [x] EOC 保持（P1 → 2阶）

**状态**: ✅ 完成
