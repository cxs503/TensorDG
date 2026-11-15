# CKS1998 · 内嵌固体壁激波反射门（G3）：参考几何偏一格缺陷修复 + 分辨率阶梯工具链

> 锚点 `REPRO-DG-CKS1998`（Cockburn & Shu, *The Runge–Kutta discontinuous Galerkin
> method for conservation laws V: multidimensional systems*, JCP **141** (1998) 199–224）
> 仓 TensorDG · 分支 `agent/m21186` · 单 m21186 · 2026-11-15
> 上承 `docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md` §4(a)（G3 诚实 FAIL，物理归因「未定」）
> 与 §5(2)（该文自己给的下一条线索：把「内壁面通量」与「DG 限幅」两个嫌疑分开）。

## 1. 缺口（本单要解决什么）

`benchmarks/forward_facing_step.py::gate_wall_reflection` 拿
`benchmarks/euler_dg_2d_wall_bc.WallReflection`（入射激波由 Rankine–Hugoniot 现算、
反射激波由跳跃条件二分）当精确解，判「域内固体壁上的激波反射，反射平台压力误差 < 3 %」。
上一单如实记了 FAIL，但**没有测过失败的来源**：可能是内壁面通量（能力缺陷），
也可能是近壁限幅/分辨率（数值分辨率）。一个网格上的 FAIL 无法区分这两件事。

## 2. 本单实质增量

### (a) 参考几何偏一格的缺陷：证明 + 修复

旧掩膜 `solid[xc >= x_wall_face - 0.5 * dx] = True` 把固体块**提前一个单元**开始，
于是**离散壁面**落在 `x_wall - dx`，而参考解仍按 `x_wall` 构造：

| nx | dx | 解析壁面 | 旧掩膜首个固体单元 | 旧掩膜离散壁面 | 偏差 |
|---:|---:|---:|---:|---:|---:|
| 60 | 0.020 | 1.000 | 49 | 0.980 | **1.000 dx** |
| 120 | 0.010 | 1.000 | 99 | 0.990 | **1.000 dx** |
| 240 | 0.005 | 1.000 | 199 | 0.995 | **1.000 dx** |
| 480 | 0.0025 | 1.000 | 399 | 0.9975 | **1.000 dx** |

偏差恒为**一整格**（不是 O(dx) 误差，是 O(1) 单元错位）：参考解的反射平台宽度
（`x_wall - front`）被系统性多算一格，且几何位移不随加密消失。修复方式是把壁面表达
抽成 `wall_solid_mask(nx, ny, x_wall_face, dx)`，令首固体单元的**左面**正好等于解析壁面；
`tests/test_wall_reflection_ladder.py::test_solid_mask_starts_exactly_on_the_analytic_wall`
在 nx=60/120/240/480 上断言 `first_solid_index * dx == x_wall_face` 到 1e-12。

### (b) 可复用的 G3 归因工具链（新增三件）

| 件 | 路径 | 作用 |
|---|---|---|
| 双路线诊断 | `validation/forward_step/wall_reflection_diag.py` | 同一物理、同一 `dx`、同一限幅器跑两条路线：`embedded`（域内固体块）/ `outer`（壁面即域边界，`bcs={"right":"wall"}`，即 `euler_dg_2d_wall_bc` 已验证的那条）。两条路线在共享流体单元上的一致性是「内壁面通量是否正确」的**直接判据**：一致到舍入 ⇒ 通量不是嫌疑；不一致 ⇒ 缺陷就锁在壁面上。 |
| 分辨率阶梯 | `validation/forward_step/wall_reflection_ladder.py` | 10 个任务（P1/P2 × nx=60…480、TVB M=1、纯 TVD M=0、无限制器稳定性探针、`outer` 交叉验证），可分片集群递交；每任务一份 JSON（含全场剖面）。 |
| 聚合 | `validation/forward_step/wall_reflection_ladder_report.py` | 由 JSON 生成 `ladder.csv` / `ladder.md` / `wall_reflection_ladder.png`（(a) 平台误差 vs dx 双对数 + 3 % 判据线；(b) 最细网格剖面 vs 精确平台值）。**零手抄**。 |

### (c) 首个实测点已经把「判据不可判」这件事量化了

nx=60 P1（TVB M=1，t = t_wall + 0.08，1392 步，质量/能量残差 3.9e-14）：

| 量 | 数值 |
|---|---:|
| 反射平台可用宽度 | **4.73 单元**（`front_buffer=3` + `wall_buffer=1` 之后只剩 **1** 个采样单元） |
| 平台 ρ 误差 | **−47.67 %**（精确 6.0） |
| 平台 p 误差 | **−59.27 %** |
| 全域 L1 / L∞（ρ） | 0.363 / 0.960 |
| 全域 L1 / L∞（p） | 0.461 / 1.618 |
| 场最大 p | 6.12（精确反射平台 15.0） |

结论（本单可确证的部分）：在该网格上，**反射平台只有 4.7 个单元宽、反射激波过渡带比平台还厚**，
所以「< 3 %」这一条此时测的不是格式的激波捕捉精度，而是「平台是否存在」。
这正是 `_plateau_mask` 在采样数 < 3 时返回 `status="not_run"` 的设计意图；
旧门用 `(240,1) P2 + limiter=None` 单点判 FAIL，把「未解析」与「格式错」混成了一句 FAIL。

## 3. 未完成（honest，不粉饰）

分辨率阶梯已作为集群作业 **cj3268**（10 分片，`stage_dir` = 本工作树）递交，
在两次 `fetch`（累计等待 ~25 min）后仍处于 `running`，返回的工件树里**只有 task00（nx=60 P1）**。
因此本文**没有**给出：

- nx=120/240/480（P1/P2）的误差阶梯与收敛阶；
- `embedded` vs `outer` 双路线一致性数字（本单最有价值的那条判据）；
- 「最细网格是否 < 3 %」的最终裁定。

G3 的级别维持 **WITHHELD/FAIL**（与上一单一致，本单不放宽任何容差、不改判据）。
工具链已就绪，取回 cj3268 后 `python validation/forward_step/wall_reflection_ladder_report.py`
一条命令即可生成上表与图，无需重跑。

另一条已知但本单未测的现象：`limiter=None` 在本算例上会走到 `dt → 0`（nx=30/40 的探针在
110 s 内未收敛到 `t_wall`，而同一算例带 TVB 只需 1392 步），与
`docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md` §4(c)「无限制器时 Mach 3 台阶在 t≈0.02 内出现
非物理单元均值」一致——`gate_wall_reflection` 的默认 `limiter=None` 因此是一个**不可运行**
的默认配置，应在取回阶梯后一并改判（而不是继续用它判 FAIL）。

## 4. 误差判定（显式回答「是否 < 3 %」）

- 唯一取回的配置（nx=60 P1，4.73 单元平台）：ρ **−47.67 %**、p **−59.27 %** → **不 < 3 %**，
  且失败量级是 O(1)（不是 3–10 % 的精度问题），与「平台未解析」的判读一致。
- 参考几何缺陷本身：**1.000 单元 / 任何分辨率**（上表），修复后为 **0.000 单元 / 1e-12**。
- cj3268 其余 9 个配置：**未取回**（见 §3），不得计为通过。

## 5. 复现命令

```bash
python validation/forward_step/wall_reflection_ladder.py --task 0      # 单个配置（~86 s）
python validation/forward_step/wall_reflection_ladder_report.py        # 表 + CSV + 图
python -m pytest tests/test_wall_reflection_ladder.py -q               # 5 passed（几何对齐 + 防手抄）
# 全阶梯（集群，分片）：
#   run_compute submit repo=TensorDG stage_dir=<worktree> shards=10 \
#     cmd_tpl="timeout 3300 python validation/forward_step/wall_reflection_ladder.py --task {i}"
```

## 6. 给下次自找活的线索

1. **取回 cj3268**（一条 `fetch`），跑聚合脚本，就能得到 §3 缺的全部数字；若最细网格仍不 < 3 %，
   §5(2) 的下一条线索就是「内壁面通量」本身：用 `outer` 路线做同一个反射，若 `outer` 达标而
   `embedded` 不达标，缺陷锁在 `tensordg/euler_dg_2d.py::_solid_interface`。
2. 若两条路线一致但都不达标，则嫌疑转到**近壁限幅**：固体单元被冻结在初始投影态，
   限幅器的相邻模板把固-液界面当强间断，贴壁流体单元被压到一阶（该文 §4(c) 已记）。
   可试的最小改动是给限幅器一个「壁面镜像」虚拟邻居（ρ、p 同、法向动量取反）。
3. `gate_wall_reflection` 的默认 `limiter=None` 需要改判（见 §3 末），这是一条独立的、
   便宜的修复。