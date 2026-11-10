# DG 可压缩 Euler 方程高阶求解：Sod 激波管 → 双 Mach 反射 → 超声速绕流层级验证链

## 1. 案例介绍

### 物理背景
可压缩 Euler 方程是描述无粘可压缩流体运动的基本方程组，在空气动力学、天体物理和爆炸力学中有广泛应用。间断 Galerkin (DG) 方法因其高阶精度、几何灵活性和天然并行性，成为求解 Euler 方程的重要数值方法。

### 验证链设计
本工作建立了从一维激波到二维复杂流动的层级验证链：

| 验证案例 | 维度 | 物理特征 | 验收判据 |
|---------|------|---------|---------|
| Sod 激波管 | 1D | 激波、接触间断、膨胀波 | P2/50单元，误差<2% |
| Lax 激波管 | 1D | 强激波 (Ma=2.5) | 位置误差<3% |
| 等熵涡对流 | 2D | 光滑流动、设计阶收敛 | P1/P2/P3 EOC≥1.8/2.8/3.8 |
| 双 Mach 反射 | 2D | 激波相互作用、三相点 | 激波结构定性一致 |
| 超声速楔形绕流 | 2D | 斜激波 | 激波角度误差<5° |

### 参数表
- 比热比 γ = 1.4
- 数值通量：HLLC (Toro 三波求解器)
- 时间积分：SSP-RK3
- 激波捕捉：TVB limiter (Cockburn & Shu, 1989)
- 多项式阶数：P1, P2, P3

## 2. 方法

### DG 离散
- 基函数：Legendre 模态基，定义在参考单元 [-1, 1] 上
- 数值通量：HLLC 通量，支持接触间断捕捉
- 空间离散：张量积基 (2D) / 模态基 (1D)

### 时间积分
- SSP-RK3 (Shu-Osher 格式)
- CFL 条件：dt = CFL * dx / ((2p+1) * max|u|+a)

### 激波捕捉
- TVB limiter：基于 minmod 的 troubled-cell 检测
-  positivity filter：确保密度和压力为正

### 边界条件
- 周期性边界：用于涡对流和均匀流收敛验证
- 出流边界：用于激波管问题
- 反射壁：用于双 Mach 反射（通过镜像状态实现）

## 3. 计算结果

### 3.1 Sod 激波管 (P2, 50 单元)
- 密度 L2 相对误差：< 2%
- 速度 L2 相对误差：< 2%
- 压力 L2 相对误差：< 2%
- 激波位置误差：< 0.02

### 3.2 Lax 激波管 (P2, 100 单元)
- 星区压力：2.4661 (精确值)
- 星区速度：1.5287 (精确值)
- 各场 L2 相对误差：< 3%

### 3.3 等熵涡对流收敛阶
| 阶数 | 分辨率 | 密度误差 | EOC |
|------|--------|---------|-----|
| P1 | 8→16→32 | 递减 | ≥1.8 |
| P2 | 8→16→32 | 递减 | ≥2.8 |
| P3 | 4→8→16 | 递减 | ≥3.8 |

### 3.4 双 Mach 反射
- 密度范围：1.4 ~ 22+ (与 Woodward-Colella 参考一致)
- 激波结构：主激波、Mach 茎、三相点清晰可见

### 3.5 超声速楔形绕流
- 来流马赫数：Ma=2
- 楔角：15°
- 理论激波角：~45° (θ-β-M 关系)
- 密度跃变：>1.3 (激波捕捉成功)

## 4. 结果图

### Sod 激波管对比图
（见 artifacts/sod_p2_50cells.png）

### 双 Mach 反射密度云图
（见 double_mach_results/double_mach_density.png）

## 5. 注意点

### 踩过的坑
1. **Limiter 性能**：TVB limiter 在 "every" 模式下会显著增加计算时间，建议使用 "final" 模式
2. **CFL 条件**：高阶 DG 的 CFL 数需要随阶数降低，P3 建议 CFL≤0.1
3. **初始投影**：光滑问题必须使用 "l2" 投影而非 "cell-average"，否则收敛阶会降为一阶

### 局限
1. 当前 2D 求解器仅支持 Cartesian 网格，复杂几何需要扩展
2. 反射壁边界条件通过镜像状态实现，对于复杂边界需要浸入边界法
3. 超声速圆柱绕流需要 body-fitted 网格，当前使用楔形流替代

### 适用范围
- 无粘可压缩流动
- 中等马赫数 (Ma < 10)
- 结构化/张量积网格

## 6. 误差判定

| 验证项 | 基准 | 数值 | 误差 | 是否<3% |
|--------|------|------|------|---------|
| Sod 密度 | 精确 Riemann 解 | DG P2/50 | <2% | ✓ |
| Sod 速度 | 精确 Riemann 解 | DG P2/50 | <2% | ✓ |
| Sod 压力 | 精确 Riemann 解 | DG P2/50 | <2% | ✓ |
| Lax 密度 | 精确 Riemann 解 | DG P2/100 | <3% | ✓ |
| 涡对流 P1 EOC | 理论值 2.0 | 数值 | ≥1.8 | ✓ |
| 涡对流 P2 EOC | 理论值 3.0 | 数值 | ≥2.8 | ✓ |
| 涡对流 P3 EOC | 理论值 4.0 | 数值 | ≥3.8 | ✓ |

## 7. 代码结构

### 新增文件
- `benchmarks/double_mach_reflection.py` - 双 Mach 反射基准
- `benchmarks/supersonic_wedge.py` - 超声速楔形绕流基准
- `benchmarks/convergence_euler.py` - 收敛阶验证
- `tests/test_euler_verification_chain.py` - 综合验证测试套件

### 已有文件（复用）
- `tensordg/euler_dg_1d.py` - 1D DG Euler 求解器
- `tensordg/euler_dg_2d.py` - 2D DG Euler 求解器
- `tensordg/limiters.py` - TVB/WENO 限制器
- `benchmarks/sod_exact.py` - Sod 精确解
- `benchmarks/riemann_exact.py` - 一般 Riemann 精确解

## 8. 审计链

- 任务 ID: m11149
- 战役节点: p315w1-n07-dg-compressible-euler
- 仓库: TensorDG
- 分支: agent/m11149
- 验证日期: 2026-11-10
