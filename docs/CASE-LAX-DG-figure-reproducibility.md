# CASE-LAX-DG · 图与记录的绑定：把「图只是装饰」变成可判定命题

> 锚点 `CASE-LAX-DG`（Lax 激波管 / Toro test 2，ref = 精确解，容差 <3 %）。
> 本文**全部数字由 `scripts/render_lax_figure_doc.py` 从两份已提交记录及其**
> `figures_binding.json`** 程序化注入**，零手抄；`--check` 可判陈旧。
> 渲染器：`scripts/render_lax_figures.py`；闸门测试：
> `tests/test_lax_figure_reproducibility.py`。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]` 域，隔膜 `x0 = 0.5`，γ = 1.4，
快照 `t = 0.13`；左态 (ρ, u, p) = (0.445, 0.698, 3.528)，右态 (0.500, 0.000, 0.571)；参考值取闭式 Riemann 解。算例池九条 `(cells, degree)` 腿与限幅器三臂（`none` / `tvb` / `weno`）各有一份 `metrics.json` + `profiles.npz`，外加若干 PNG。

本文处理的是**证据等级**问题，不是物理问题：`metrics.json` 里的数可以逐位复核，但 PNG 是二进制，谁也不能从一张图看出它是哪次运行的产物。原状是「图由记录渲染」只写在 `figures_provenance` 的一句话里，没有任何检查能发现「图被替换」或「记录重跑了、图没重画」。

## ② 怎么做的

1. **一行侧车同时钉住两端**：每次渲染写 `figures_binding.json`，里面既有**输入**（`record_sha256` / `profiles_sha256` / `renderer_sha256`），也有**输出**（`figures_sha256` 一图一摘要）。摘要若只写进记录内部，就只能证明「记录里写的摘要 == 磁盘上的图」——记录整份被新运行重写、图没重画时它依然自洽；侧车记录 `record_sha256` 恰好补上这个方向。
2. **G1 绑定闸门**（不需要 matplotlib）：记录字节、`profiles.npz` 字节、渲染器字节、每张 PNG 字节四项全对，且记录 `figures`、磁盘 `*.png`、侧车摘要键三者集合相等；缺侧车、孤儿图、缺图、字节漂移一律 FAIL。
3. **G2 图篡改对照**：在**副本**里翻转某张图最后一字节，G1 必须 FAIL。
4. **G3 记录篡改对照**：在**副本**里改写记录里的一个字段（模拟「重跑未重画图」），G1 也必须 FAIL —— 这条正是「摘要写进记录内部」做不到的方向。
5. **G4 记录幂等**：渲染器写回记录时沿用记录自身的 JSON 缩进。原实现硬编码 `indent=1`，而算例池记录是 `indent=2`，于是**一次无变化的重新渲染也会重写 2518 行**数字——真的改了数和只是换了缩进，在 diff 里再也分不出来。（`PR#575` 的 `metrics.json` 差异 2292 行里既有真实重算，也有这类整份重写，这正是需要幂等写回的理由：让**没有变化的**那次渲染彻底不出现在 diff 里。）
6. **G5 逐字节重渲染**：从已提交记录 + `profiles.npz` 重新渲染到临时目录，要求每张 PNG 与已提交字节完全相同（由测试执行，需要 matplotlib）。

## ③ 计算结果

### 3.1 绑定输入（图从哪儿来）

| 记录 | 绑定量 | 侧车记录值（前 16） | 实测值（前 16） | 判定 |
|---|---|---|---|---|
| `pool` | `record_sha256` | `cab44001048395cc` | `cab44001048395cc` | PASS |
| `pool` | `profiles_sha256` | `7bdcc57cb9a0c212` | `7bdcc57cb9a0c212` | PASS |
| `pool` | `renderer_sha256` | `5aaa860b14941e86` | `5aaa860b14941e86` | PASS |
| `arms` | `record_sha256` | `42b449bab0098de7` | `42b449bab0098de7` | PASS |
| `arms` | `profiles_sha256` | `17e4dae7b725c15f` | `17e4dae7b725c15f` | PASS |
| `arms` | `renderer_sha256` | `5aaa860b14941e86` | `5aaa860b14941e86` | PASS |

### 3.2 绑定输出（图是什么）

| 记录 | 图 | 字节 | 侧车 sha256（前 16） | 实测 sha256（前 16） | 判定 |
|---|---|---|---|---|---|
| `pool` | `entropy_wave_order.png` | 59423 | `91bd74b53d9a4da5` | `91bd74b53d9a4da5` | PASS |
| `pool` | `lax_acceptance_a0.png` | 108114 | `b18efb3bbc640302` | `b18efb3bbc640302` | PASS |
| `pool` | `lax_convergence.png` | 91941 | `e0d2435c305669ea` | `e0d2435c305669ea` | PASS |
| `pool` | `lax_profiles.png` | 88993 | `f78845d66a09ee9a` | `f78845d66a09ee9a` | PASS |
| `pool` | `time_refinement.png` | 50076 | `972a3bc56946c56f` | `972a3bc56946c56f` | PASS |
| `arms` | `lax_limiter_arms_orders.png` | 146985 | `bdada8ebbbffbe71` | `bdada8ebbbffbe71` | PASS |
| `arms` | `lax_limiter_arms_profiles.png` | 80051 | `d85838b51a45a3f8` | `d85838b51a45a3f8` | PASS |

### 3.3 闸门总表

| 闸门 | 判定 | 说明 |
|---|---|---|
| G1 绑定一致（7 张图、6 项输入） | **PASS** | 记录 / profiles / 渲染器 / 图 四类字节全对，三方集合相等 |
| G2 图篡改可检出 | **PASS** | `pool`: `figures/entropy_wave_order.png` 翻转第 59422 字节 → FAIL；`arms`: `figures/lax_limiter_arms_orders.png` 翻转第 146984 字节 → FAIL |
| G3 记录篡改可检出 | **PASS** | `pool`: 改写记录 `generated_at_utc`（模拟重跑未重画图） → FAIL；`arms`: 改写记录 `generated_at_utc`（模拟重跑未重画图） → FAIL |
| G4 记录幂等 | **PASS** | `pool` 字节不变；`arms` 字节不变 |
| 溯源串指向渲染器 | **PASS** | `figures_provenance` 与 `PROVENANCE` 常量逐字相同 |
| G5 逐字节重渲染 | 由测试判定 | `tests/test_lax_figure_reproducibility.py`（需 matplotlib，用例内 `importorskip`） |

### 3.4 工件规模

| 记录 | 生成时刻 (UTC) | 图数 | 图总字节 |
|---|---|---|---|
| `pool` | 2026-11-14T08:05:22Z | 5 | 398547 |
| `arms` | 2026-11-14T08:28:09Z | 2 | 227036 |

## ④ 注意点

1. **绑定不等于可复现**：G1 只把「记录/图/渲染器」三者钉在一起，任何人只要同时改图、改记录、重写侧车就能一起骗过 G1。真正闭口的是 G5——从记录重渲染必须得到同样的字节；G1/G2/G3 是它的廉价前置门（无需 matplotlib，秒级）。
2. **G5 的适用范围**：PNG 字节依赖 matplotlib 版本与后端。本环境实测逐字节相同（matplotlib 3.10.8，Agg）；换绘图栈后 G5 可能 FAIL 而图语义未变——此时正确动作是重渲染并刷新侧车（`python scripts/render_lax_figures.py`），而不是放宽闸门。
3. **`renderer_sha256` 是有意的耦合**：改渲染器（哪怕只改一句注释）都会让 G1 变 FAIL，迫使重渲染。代价是两百毫秒，收益是「代码改了、图没跟上」这类漂移无法静默存活。
4. **G4 是格式守卫，不是语义守卫**：它只保证「无变化的记录重写后字节不变」，不保证记录内容正确；记录内容由算例池自身的闸门负责。
5. **本机渲染 vs 节点渲染**：loong64 计算镜像无 matplotlib，图一律在节点外渲染；「记录 → 图」这一步本来就跨机器，所以更必须有侧车把它钉住。

## ⑤ 误差判定

本文的关键量是**绑定关系**，不是物理量，故误差判据为字节相等（偏差 0 才算真）：

| 关键量 | 基准 | 偏差 | 是否 < 3 % |
|---|---|---|---|
| 图字节 sha256（7 张） | 侧车 `figures_sha256` | 0 | 是（严格 0） |
| 记录 / profiles / 渲染器字节 | 侧车三项输入摘要 | 0 | 是（严格 0） |
| 图篡改检出 | 必须 FAIL | 检出 | 是 |
| 记录篡改检出 | 必须 FAIL | 检出 | 是 |
| 记录重写字节 | 原记录 | 0 | 是（严格 0） |

**物理误差本文明不涉及**：本文不动求解器、不重算任何误差，`verdict = FAIL` 的 L2 判据与限幅器归因仍以 `docs/CASE-LAX-DG.md`、`docs/CASE-LAX-DG-limiter-arms.md`、`docs/CASE-LAX-DG-acceptance-A0.md` 为准；那三份文档的结论（整场相对 L2 超容差、C3/R3 阶数闸门 FAIL）**未被本文改变**。

