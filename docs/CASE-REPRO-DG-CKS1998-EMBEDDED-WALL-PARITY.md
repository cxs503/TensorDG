# CKS1998 · 域内固体壁（embedded wall）与外壳壁面的一致性：G3 归因

> 锚点 `REPRO-DG-CKS1998`（Cockburn & Shu, *The Runge–Kutta discontinuous Galerkin method
> for conservation laws V: multidimensional systems*, JCP **141** (1998) 199–224）。
> 本单是对 `docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md` §4(a) 遗留的 **G3 门**
> （「激波打到**域内**固体壁后，反射平台是否符合精确双激波解」）做的**归因**工作，不是新功能。

## 1. 待解问题（上一单留下的两个嫌疑）

上一单把 G3 记为 **WITHHELD/FAIL**，并明确写出两个尚未区分的嫌疑：

1. 域内固体壁的数值通量（`_solid_interface` 的本征滑移壁通量）本身写错了；
2. P1/P2 DG 在该网格上把两道激波抹得太宽，采样平台必然超差。

两者对结论的含义完全不同：若是 (1)，则前向台阶算例（依赖同一路径）的一切结果都不可信；
若是 (2)，则只是分辨率问题，能力本身可用。本单用**配对 A/B** 把两者分开。

## 2. 方法：等 dx 的配对实验（只差壁面处理）

`benchmarks/cks1998_embedded_wall_parity.py`（新增驱动）对同一个「Mach 2 激波打在
x=1.0 壁面上」的算例跑两条腿：

| 腿 | 计算域 | 壁面处理 |
|---|---|---|
| `solid` | [0, 1.2]，nx 个单元 | `i ≥ i_wall` 的单元被冻结为固体（`solid=` 掩膜） |
| `boundary` | 截断到壁面 [0, 1.0]，nx·5/6 个单元 | `bcs={"right": "wall"}`（既有外壳壁路径） |

`nx_boundary = nx_solid · x_wall/xmax` 使两条腿的 **dx 完全相同**，于是第 i 个单元在两条腿里
占据同一个物理区间，单元均值可以逐格相减。**两条腿的差异只能来自壁面处理**；两条腿**共同**
带的误差是激波捕捉误差。激波位置取格心面（`_snap_to_face`）、固体掩膜用 `wall_solid_mask`
（壁面落在单元面上），与 G3 门现有配置一致；唯一偏离是 `extra_time=0.15`（门默认 0.08 在
nx<200 时平台不足 4 个缓冲单元，会全部报 `not_run`）。

驱动同时给出：与精确双激波解（`WallReflection`，Rankine–Hugoniot 现算）的 L1/L∞ 相对误差、
平台带（离反射波前 3 格、离壁 1 格之外）的均值/最差误差、以及 3 % 门的判定。

## 3. 结果（本机实跑，数字由程序打印）

配对实验（dx = 0.1，P1，两条腿各 562 步，本机 51 s / 55 s）：

| 腿 | nx | P | dx | 步数 | L1(ρ) 相对误差 | 壁面处理差 |
|---|---|---|---|---|---|---|
| boundary | 10 | 1 | 0.1 | 562 | 10.732 % | — |
| solid | 12 | 1 | 0.1 | 562 | 10.732 % | **6.2e-15（机器精度）** |

逐格：`max|ρ_solid − ρ_boundary| = 1.87e-14`，量级 3.0（精确解均值），即
**1.9e-14 / 3.0 = 6.2e-15**，比双精度舍入堆积（562 步 × 3 级）还小。两条腿的
质量残差同为 ~2e-14（solid 腿额外给出 `obstacle_impulse`，boundary 腿无此项，故 NaN 语义不同）。

结论：**嫌疑 (1) 被否定**。在可以逐格比较的判据上，域内固体壁与外壳壁面给出**同一个解**到机器
精度；两条腿共同承担的 10.7 % L1 误差是 dx=0.1 上两道激波的捕捉误差（该网格上平台带连
1 个可取样单元都不足，`plateau_cells=0` → `not_run`，不是 FAIL）。

单元测试把这一条钉成回归判据（`tests/test_euler_dg_2d_solid.py::test_embedded_wall_matches_boundary_wall`，
`whole_domain_l1_relative < 1e-10`，实跑 96 s）。

## 4. 顺带查明的第二个缺陷：旧 G3 单测的初值方向是反的

上一单的 `tests/test_euler_dg_2d_solid.py` 用 `xfail(strict=False)` 记录「反射平台不符」，
理由是「物理归因未定」。读代码发现**测试自己写错了初值方向**：

```python
wr = WallReflection(x_shock=0.2, x_wall=0.5, ...)   # 激波向右跑：波后态在左侧
ahead = x < wr.x_zero                              # ← 判定「波前」为左半区
torch.where(ahead, wr.pre[...], wr.post[...])      # 于是把波前态放到左侧
```

`WallReflection` 的语义是「波后态（高密度、向右速度）在前锋**左侧**」，旧辅助函数把
`pre`（静止、低密度）塞到左侧，等于让气体**背离**壁面跑；参考解却仍按「激波撞壁反射」给平台。
初值方向反了，平台断言当然不成立，而这与壁面通量毫无关系。同源问题：旧掩膜用
`solid[xc > wr.x_wall]`（格心判据）把固体起点提前一格，离散壁面比解析壁面短一个 dx，
与 G3 门里 `wall_solid_mask` 的注释记录的是同一个坑。

已替换为两条新测试：

- `test_exact_two_shock_orientation`（无需求解器，4 s）：钉住「波后态在左、速度朝 +x」，
  并断言驱动的初值与之同向——防止方向写反复发；
- `test_embedded_wall_matches_boundary_wall`（96 s）：§3 的配对不变量。

`xfail` 已删除：它记录的是一条**测试侧缺陷**导致的失败，继续挂着会把错误归因固化进历史。

## 5. 尚未完成（honest，不粉饰）

- **分辨率阶梯未取回。** `scripts/run_cks1998_parity_ladder.sh`（nx = 60/120/240 × P1/P2，
  12 臂并行）已作为集群作业 **cj3657**（12 核，timeout 3600）递交，交接时排队 38 min 仍在
  **queued**。因此本文**没有** dx ≤ 1/120 的配对数字，也**没有**「G3 在哪个分辨率上进入 3 %」
  的结论——那一条仍是 **WITHHELD**。取回后应直接把 §3 表的 dx 列延伸下去，并在
  `parity.json` 的 `parity[]` 里读 `plateau_max_percent`。
- **3 % 平台门本身未重判。** 本单只证明「误差不是壁面造成的」，没有证明「误差足够小」。
  据 dt 与网格推算，nx ≥ 200 才有 ≥4 个可取样平台单元，故门级判定要等 cj3657 的 nx=240 臂。
- 单步成本实测 ~60–115 ms（40×1 的 P1 网格，python 层开销主导，与网格无关），因此阶梯
  必须并行递交；本机 128 核但计算门只放行 <8 核 <10 min，故走 `run_compute`。

## 6. 复现

```bash
# 配对不变量（单元测试，~2 min）
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/test_euler_dg_2d_solid.py -q \
    -k "orientation or matches_boundary"

# 单臂 / 聚合（本机做得完的规模，dx=0.1）
python -m benchmarks.cks1998_embedded_wall_parity --single solid:12:1 --outdir runs/parity
python -m benchmarks.cks1998_embedded_wall_parity --single boundary:10:1 --outdir runs/parity
python -m benchmarks.cks1998_embedded_wall_parity --aggregate --outdir runs/parity

# 完整阶梯（12 臂并行，需递交集群）
bash scripts/run_cks1998_parity_ladder.sh
```

产物：`docs/benchmarks/results/cks1998-embedded-wall/{parity.json,parity.png,arm_*.json}`。