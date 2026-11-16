# CASE-SHU-OSHER 认证参考：时间腿实测（CFL 探针）

锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE` · 任务 m22895 · 域：TensorDG 验证欠账

## 1. 案例介绍

Shu–Osher 激波-熵波相互作用，1-D Euler（γ=1.4），域 $x\in[-5,5]$，$t=1.8$：
左态 $(\rho,u,p)=(3.857143,\,2.629369,\,10.3333)$，右态 $(1+0.2\sin 5x,\,0,\,1)$，
透射边界，参考求解器 FV-WENO5-JS / HLLC / SSP-RK3。

**问题**：已发布的 round-2 参考梯次 `fv_cfl0p3_n800/1600/3200/6400` 全部在**单一** CFL=0.3
下生成，相邻差分被当作纯空间误差、Roache 带宽 $U=F_s d_{fine}/(r^p-1)$ 被当作参考自身
不确定度。时间离散到底贡献多少，从未被测量（审计 `benchmarks/xval_dg_round2_time_pollution.py`
自落地起真实分支即为 `WITHHELD`）。

## 2. 怎么做的

1. **固定网格只细化 CFL**（0.3 → 0.15 → 0.075，比例 2），差分即纯时间误差；
   差分用整条序列的**公共分母**（最细成员），逐对归一化会把误差自身混进分母。
2. **N=800**：参考梯次最粗一级，CFL 0.3 与 0.15 剖面**已提交**，直接量 $\delta_t$；
   与之比较的空间差分为 $d_{fine}(800)=\|\mathcal{C}(u_{1600})-u_{800}\|/\|u_{800}\|$
   （`certify` 的粗网格约定）。
3. **N=400 探针**：本地三级 CFL 序列（17.8 + 32.7 + 66.9 s），用来测**真实时间阶** $p_t$。
   探针级**故意不放进** `artifacts/raw`：认证参考的加载器按 `fv_cfl0p3_n*.npz` 通配，
   放进去会悄悄把认证梯次扩成 5 级、破坏逐位一致性测试；因此落在
   `validation/xval_dg_round2/artifacts/cfl_probe/`。
4. 污染比 $R=\delta_t/d_{fine}$ 与预声明门槛比较（CLEAN ≤ 0.1，
   MARGINAL ≤ 0.25）。
5. **未完成**的那一半：N=1600/3200/6400 的 CFL 细化级需 1.2 CPU·h，已递交集群
   `cj4464`（单核 6 分片，生成器 `validation/xval_dg_round2/gen_cfl_ladder.py`
   幂等、已落盘即跳过），本单未取回；没有它们做不了时间修正后重认证（T3）与逐腿翻转（T4）。

## 3. 计算结果（真实数据，程序化注入）

### 3.1 N=400 探针：真实时间阶

| CFL 对 | 比例 | 相对 L1 差分 | 步数 |
|---|---|---|---|
| 0.3 → 0.15 | 2 | 0.005275 % | 1148 / 2295 |
| 0.15 → 0.075 | 2 | 0.000519 % | 2295 / 4591 |

| 量 | 值 |
|---|---|
| 时间观测阶 $p_t$（SSP-RK3 理论 3） | **3.345479** |
| 相邻时间差分单调递减（T0 形状） | True |
| 最粗级时间误差 $\delta_t$（CFL 0.3） | 0.005275 % |
| 同网格空间差分 $d_{fine}(400)$ | 0.614577 % |
| 污染比 $R=\delta_t/d_{fine}$ | **0.0086** → CLEAN |

### 3.2 N=800（已提交数据）：污染比

| 量 | 值 |
|---|---|
| CFL 成员 | [0.3, 0.15] |
| $\delta_t$ | 0.010596 % |
| $d_{fine}(800)$ | 0.230839 % |
| $R$ | **0.0459** → CLEAN |
| $p_t$ | n/a（该网格只有两级 CFL，一个差分不含阶信息） |

### 3.3 认证梯次自身的空间差分（对照）

| 量 | 值 |
|---|---|
| 最细相邻空间差分 $d_{fine}(1600\to3200)$ | 0.100975 % |
| 参考网格 / 门槛 | N=3200 / 5 % |

## 4. 结果图

![fig_time_leg_n400_probe](validation/xval_dg_round2/artifacts/cfl_probe/fig_time_leg_n400_probe.png)

*图 1：N=400 探针上真实的时间差分随 CFL 的幂律（点=实测，虚线=观测阶拟合）。*

![fig_pollution_ratio_real](validation/xval_dg_round2/artifacts/cfl_probe/fig_pollution_ratio_real.png)

*图 2：两个已测网格的污染比 $R$ 与预声明门槛（绿=CLEAN 0.10，红=MARGINAL 0.25）。*

## 5. 注意点（坑 / 局限）

1. **通配污染坑**：把探针剖面写进 `artifacts/raw` 会让 `shu_osher_certified_ref` 的
   `fk_cfl0p3_n*.npz` 拾取到 N=400，认证梯次被悄悄扩级 —— 本单把探针隔离在
   `artifacts/cfl_probe/`，并保留该理由在生成器 docstring 里。
2. **两级 CFL 序列陷阱（本单修掉的引擎缺陷）**：审计文档与测试 `_REQUIRED` 都只要求
   N=6400 有 CFL 0.15（只算 $R$），但 `temporal_series` 硬要求全部三级，数据真到位时
   `audit()` 会以 `FileNotFoundError` 整单阵亡 —— 一个只会在数据出现后暴露的潜伏缺陷。
   现 `temporal_series(..., require=)` 允许"部分序列"：BASE 网格仍要求满三级
   （$p_t$ 需要两个相邻差分），额外网格只要最粗一对。
3. **$p_t$ 受激波限制**：SSP-RK3 光滑区三阶，间断处退化，探针值只用于判断"时间腿是否
   高阶"，不作为精度结论。
4. **探针网格不属认证梯次**：N=400 只为测时间阶；其 $R$ 与 N=800/3200 的 $R$ 不完全同
   口径（$d_{fine}$ 取自不同网格对），跨网格比较只看数量级。
5. **集群排队**：CFL 0.15/0.075 级在队列中排队（单核 6 分片），本单未取回；缺它们只能测 N=800 的 R，无法做时间修正重认证。

## 6. 误差判定（显式回答是否 < 3%）

| 关键量 | 基准 | 实测 | 误差 | < 3% ? |
|---|---|---|---|---|
| 探针时间阶 $p_t$ | 3（SSP-RK3 设计阶） | 3.345479 | 11.52 % | **否**（见注意点 3，激波限制） |
| N=400 污染比 $R$ | CLEAN 门槛 0.10 | 0.0086 | 裕度 91.4 % | **是** |
| N=800 污染比 $R$ | CLEAN 门槛 0.10 | 0.0459 | 裕度 54.1 % | **是** |
| 参考带宽位移（T3） | — | **未测**（需 1600/3200 的 CFL 级） | — | **无法回答** |

**结论**：已测网格上时间腿**远小于**空间腿（$R\approx$0.046），
即"带宽主要由空间差分设定"的既有读法在 N=800 与 N=400 上成立；但审计整体仍**不得**
判为 `TIME_CLEAN`——参考网格 N=3200 的 $R$、时间修正带宽位移、11 条 DG 腿的翻转都还缺数据。

## 7. 复现

```bash
# 探针（本地 ~2 min）
python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 400 --cfl 0.3 \
    --out validation/xval_dg_round2/artifacts/cfl_probe/fv_cfl0p3_n400.npz
# 完整梯次（集群，幂等）
python validation/xval_dg_round2/gen_cfl_ladder.py --shard 0 --nshards 6
# 审计（数据齐后自动走真实分支）
python -m benchmarks.xval_dg_round2_time_pollution --figures --require-accepted
# 测试
python -m pytest -q tests/test_xval_dg_round2_time_pollution.py \
    tests/test_xval_dg_round2_cfl_ladder_driver.py
```

## 8. 产物

| 文件 | 说明 |
|---|---|
| `validation/xval_dg_round2/gen_cfl_ladder.py` | CFL 细化级生成器（分片、幂等、命名与审计一致） |
| `tests/test_xval_dg_round2_cfl_ladder_driver.py` | 18 项不变式：级表覆盖、命名一致、可用性门、分片划分、幂等 |
| `benchmarks/xval_dg_round2_time_pollution.py` | T0–T4 审计引擎（本单修部分序列支持） |
| `tests/test_xval_dg_round2_time_pollution.py` | 25 项（+4 项部分序列契约） |
| `validation/xval_dg_round2/artifacts/cfl_probe/*.npz` | N=400 三级 CFL 探针剖面 |
| `validation/xval_dg_round2/artifacts/cfl_probe/time_leg_probe.json` | 本页全部数字的源 |
| `validation/xval_dg_round2/artifacts/cfl_probe/fig_*.png` | 两张结果图 |
| 本文件 | 方法 / 实测 / 局限（数字程序化注入） |
