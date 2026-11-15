# REPRO-DG-CKS1998 · G3 收口进行中：域内固体壁的激波反射

> 锚点：`REPRO-DG-CKS1998`（Cockburn & Shu, *The Runge--Kutta discontinuous Galerkin
> method for conservation laws V: multidimensional systems*, JCP **141** (1998) 199--224）
> 承接：`docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md` §4(a)——该单留下的唯一
> **honest FAIL / WITHHELD** 项：「域内固体壁的激波反射定量正确」没有证据。
>
> **本单状态（逐字诚实）**：把「无证据」变成「可测量的闸」，并把闸里查到的两类量测缺陷修掉；
> **数值判决仍未取得**——两条集群作业在本单交接时仍在排队（`cj3576` 梯子 12 分片、
> `cj3578` 剖面 4 分片，排队位次 >30，队列在飞 48 作业），本机逐次 120 s 上限内
> 跑不完一次单臂（实测 ≈0.13 s/RK 步 × 700+ 步），故本文件**不宣称**任何 <3 % 结论。

## 1. 问题是什么

`tensordg.euler_dg_2d` 对「网格内部固体块」的处理是：固-液面用滑移壁通量 `(0,p,0,0)`
（`p` 取**流体侧**迹），固体单元冻结在初始投影态，固体单元的外壳面通量置零。
前向台阶单证明了这套机制在**精确可判的门**上是对的（自由流保持 1.9e-12、封闭方腔
质量/能量/含冲量动量守恒 ~5e-15、台阶上游不动 <1e-10），但唯一的**定量物理**门
——把壁面从域边界挪到域内部后与精确二激波解对比——当时**不过**，且残余偏差的归因未定
（「壁面通量实现」vs「P1 + 粗网格抹平激波」）。

## 2. 这一单做了什么（全部落盘，可复核）

### 2.1 把闸参数化（`benchmarks/forward_facing_step.py`）

`gate_wall_reflection` 原先硬编码 `x_shock=0.4 / x_wall=1.0 / xmax=1.2`，因此无法
缩短「反射前行进」（占总步数 ~3/4）来做分辨率梯子，也无法做臂间对照。现在新增四个
关键字参数且**默认值不变**（历史调用逐位不变）：`x_wall`、`x_shock`、`xmax`、
`wall_mode ∈ {"solid","boundary"}`。

`wall_mode="boundary"` 是**归因臂**：同一组黎曼数据，把域截到 `x=x_wall`、壁改成域边界
`bcs right="wall"`（本仓此前所有 `WallReflection` 验证走的就是这条路径），域内无固体单元。
两臂对**同一个**闭式解打分、几何/CFL 相同，**只换壁的实现位置**，所以两臂之差就是
「域内壁面实现」的净值——这条对照不需要任何文献数据。

### 2.2 修掉闸里两处会给出**无意义判决**的量测缺陷

| 缺陷 | 后果 | 现在 |
|---|---|---|
| `extra_time ≤ 0`（反射尚未发生）时闸照常取样 | 把入射波前气体与**反射平台**态比较，输出一个无意义的 FAIL | 直接 `ValueError` 拒绝（§「reflected plateau 不存在」） |
| 平台取样带与波形不挂钩 | 早期版本在 5 格宽的平台上取 0.12/0.45/0.7 三处，落在抹平的激波里，任何捕捉激波的格式都过不了 3 % | 取样带由 `_plateau_mask` 给出：距**精确**反射波前 ≥3 格、距壁面 ≥1 格；存活单元 <3 时闸报 `status="not_run"` 而不是假 FAIL/PASS |

### 2.3 两个新驱动（集群）

- `benchmarks/g3_internal_wall_ladder.py`：12 行梯子（`nx∈{60,120,240,480}` ×
  `P1/P2/P3`… 实际为 P1/P2 × 两条臂 × TVB(M=1) 工程配置），每行报
  平台逐单元最差误差、平台均值误差、全域 L1/L∞、步数、质量残差；JSON 落
  `validation/g3_internal_wall/ladder.json`（分片写 `ladder.shard{i}of{n}.json`）。
- `benchmarks/g3_wall_reflection_profiles.py`：末时刻 ρ、p 一维场 + 精确解，供报告出图
  （`profiles.json`）。几何/参考解复用同一套 helper，图不会与数字脱钩。
- `benchmarks/g3_local_pair.py`：短行进（`x_shock=0.9`）的两臂本地版，参数已按本机
  时间预算调小（`nx=32`），但**本机仍不够快**——见 §4 局限 1。

### 2.4 发布脚本

`publish_g3_internal_wall.py`：只读上述 JSON，程序化生成 markdown 表格 + 两张图
（误差 vs Δx 的 log-log；剖面 vs 精确解），并 POST `/tensor/reports`。**数字零手抄**，
artifact 到位后一条命令即可发布。

## 3. 已经证明的（不依赖集群）

`tests/test_g3_internal_wall_ladder.py`，本机实跑 **10 passed / 3 skipped**（skip 的
3 条是需要 `ladder.json` 的判决类断言，artifact 缺失时**明确 skip 而非放行**）：

| 断言 | 内容 |
|---|---|
| 壁面几何不变量（`nx∈{60,120,240,480}` 参数化） | 首个固体单元的**左面**与解析壁面重合到 1e-12；固体块是从该面起**连续**的右半板，液态里不会出现空洞 |
| 平台取样带 | 取样点严格落在 `front+3Δx` 与 `x_wall−Δx` 之间；平台过窄时返回空集 |
| 平台宽度可测性 | `WallReflection` 的 `extra_time` 配置在四种网格下都要留下 ≥3 个可测单元，否则闸会在 `not_run` 状态「静默通过」——这里把它变成构造期断言 |
| 参考解自洽 | 反射平台 `u=0`（气体停在壁上）、`p₁ < p₂ < p₅`、反射激波速度与入射激波反号 |
| 反射前取样被拒 | `extra_time=0` → `ValueError` |
| 边界臂几何 | `wall_mode="boundary"` 时域末端=`nx·Δx`=`x_wall`，且真实跑通了该分支（`steps>0`、`status="not_run"` 符合 12 格网格的预期） |

这些断言把 §1 里「几何错位」与「取样落在波里」两类**不可通过细化消除**的假失败排除掉了，
剩下的才是真问题。

## 4. 注意点 / 局限（honest）

1. **数值判决未取得**。本机单次 shell 上限 120 s，而本问题一次单臂实测 ≈0.13 s/RK 步
   （128 核机上小张量 op 开销主导，`nx` 几乎不影响单步成本）× 700–4000 步；两次本地
   尝试（`nx=48/32`）均在 105 s 内被杀，**未产生数字**。因此本文件不含任何平台误差数字，
   也不含判决。集群作业 `cj3576`（12 分片梯子）/`cj3578`（4 分片剖面）已递交，交接时
   排队位次 >30（在飞 48 作业）。
2. **本算例是 ny=1 的降维版**：精确检验 x 向固体面通量路径，**不**检验 y 向固体面、
   90° 拐角（两向都只有一侧是流体）与斜壁；前向台阶的拐角仍只由 G4 定性覆盖。
3. **平台宽度由 `extra_time` 控制**（宽 = `|S_r|·extra_time`），粗网格必须放大
   `extra_time`（60 格用 0.20、240 格用 0.08），因此各行平台**位置**不同，
   跨行比较的是同一物理量的误差，不是同一批单元。
4. **左端 `outflow`**：反射后平台与入射波后态分界以外，左端会流出少量波后气体，
   故以平台内部带为主判据、全域 L1 为辅，两者不可互相替代。
5. **原 `xfail` 未动**：`tests/test_euler_dg_2d_solid.py::test_shock_reflects_off_embedded_wall`
   仍在 `xfail(strict=False)` 下——它用的 `nx=30`、`extra_time=0.05` 配置只有 ~3 个平台
   单元，判据 6 % **无法由任何捕捉激波的格式满足**。要改它必须先有 §3 的梯子数字，
   否则就是「为了让测试变绿而放宽闸」。

## 5. 收口路径（下一步，一条命令量级）

```bash
# 1) 取回梯子与剖面（队列空的时候）
run_compute action=fetch job_id=cj3576 ; action=fetch job_id=cj3578
# 2) artifact 到位后，把判决类断言激活（3 条 skip 会自动转红/绿）
python -m pytest tests/test_g3_internal_wall_ladder.py -q
# 3) 生成带图报告（程序化注入，零手抄）
python publish_g3_internal_wall.py
# 4) 若梯子证明 solid 与 boundary 两臂在细化下收敛到同一误差水平且 <3 %：
#    把 test_shock_reflects_off_embedded_wall 的 xfail 摘掉，改判据为梯子实测值 + 余量；
#    若两臂系统性分离：域内壁面通量确有缺陷，需查 _solid_interface 的迹选择与
#    _freeze_solid 和限幅模板的相互作用（`docs/CASE-...-FORWARD-STEP.md` §4(a) 线索 2）。
```

## 6. 复现命令

```bash
python -m pytest tests/test_g3_internal_wall_ladder.py -q        # 不变量门（本机 ~50 s）
python -m benchmarks.g3_internal_wall_ladder --index i --shards 12   # 集群梯子
python -m benchmarks.g3_wall_reflection_profiles --index i --shards 4 # 集群剖面
python publish_g3_internal_wall.py                                # 报告（需 artifact）
```