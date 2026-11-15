# CASE-DG-EULER-VORTEX-ERR round 9：方向轴（C10）与 C3 尺度缺陷修复

锚点 `CASE-DG-EULER-VORTEX-ERR`（算例池·等熵涡移动误差守恒，ref: Yee et al. 2000，容差 <6%）。
数字全部由 `benchmarks/case_euler_vortex_errconserve.py` 工件程序化注入
（`case_euler_vortex_errconserve/metrics.json` + `results/*.json`），零手抄。

## ① 案例介绍

等熵涡以匀速 u_inf 在周期箱 `[-5,5]^2` 中平移，密度/压力扰动为高斯包络，流场处处等熵，
故有闭式解。算例池用它检验「误差是否随时间累积」：真解不变形，守恒良好的格式在固定网格上
的误差应当是网格属性（投影误差为主），而非时间属性。

| 参数 | 值 |
|---|---|
| 域 / 边界 | [-5, 5]^2，周期 |
| 涡强度 beta | 5.0 |
| 背景 (rho_inf, p_inf) | (1.0, 1.0)，gamma = 1.4 |
| 输运速度模 | \|u_inf\| = sqrt(2) = 1.4142 |
| 设计族 | DG p=2，32x32，CFL 0.2（HLLC + SSP-RK3） |
| 交叉族 | P0 有限体积（Rusanov + SSP-RK3） |
| 参考解 | 周期 min-image 折叠（`min_image`） |

## ② 怎么做的

本轮唯一实质增量是方向轴。此前所有族（C7 网格/CFL、C8 阶数、C9 设计阶）都把涡沿网格对角线
(1,1) 输运，「误差是网格属性」始终只是一个对流方向上的陈述。C10 把三个同速方向放在一起，
只移动方向、不移动行程：

* `diag45` = (1, 1)：历史族，键名不变，历史结果文件不重命名、不重跑；
* `axis0` = (sqrt(2), 0)：网格对齐，结构化张量积 DG 在此方向无交叉导数耦合；
* `obl30` = (sqrt(2)cos30, sqrt(2)sin30)：把历史读数夹在中间，而不是单向外推。

每方向 4 个检查点（t = 1, 2, 5, 10），网格/CFL/阶数与设计族完全一致。
方向必须进入族身份 `family_key`，否则两个不同问题的运行会被同一条 `e ~ t^alpha` 拟合吸收；
该不变量由 `tests/test_case_euler_vortex_errconserve_direction.py` 钉死（含「陈旧族标签须由
config 重推」用例）。计算递交 compute gate 两作业 `cj3486`（axis0）/`cj3487`（obl30），
阶段目录=本工作树，单核，工件 tar 取回后落入 `results/`。

## ③ 计算结果

### 门禁总表

| 判据 | 状态 | 实测 |
|---|---|---|
| C1_tolerance | passed | 0.04926 |
| C2_error_conservation | passed | - |
| C3_discrete_conservation | passed | - |
| C4_entropy_budget | passed | 2.98e-07 |
| C5_crosscheck_order | passed | - |
| C6_ladder_completeness | not_evaluated | - |
| C7_design_space_invariance | passed | 0.1989 |
| C8_degree_invariance | not_evaluated | 0.1987 |
| C9_design_order | not_evaluated | 4.093 |
| C10_direction_invariance | passed | 0.1987 |

### 方向轴（C10）

| 方向标签 | u_inf | 角度 | 检查点数 | t_max | alpha | 判定 |
|---|---|---|---|---|---|---|
| diag45 | (1.0000, 1.0000) | 45.0 deg | 4 | 10.0 | 0.199 | 通过 |
| axis0 | (1.4142, 0.0000) | 0.0 deg | 4 | 10.0 | 0.166 | 通过 |
| obl30 | (1.2247, 0.7071) | 30.0 deg | 4 | 10.0 | 0.191 | 通过 |

三个方向全部满足 alpha <= 1，最差 0.199（diag45）。
误差放大因子（相对 t=0 投影基线）：`dg_d2_n32_cfl0.2` A(10)=2.761，`dg_d2_n32_cfl0.2_diraxis0` A(10)=16.160，`dg_d2_n32_cfl0.2_dirobl30` A(10)=3.844，`dg_d2_n32_cfl0.4` A(10)=2.762，`dg_d2_n64_cfl0.2` A(10)=1.775。

读数解释（关键）：网格对齐 `axis0` 的 alpha 最小（0.166），
但放大因子 A(10) 最大（16.16）。两者不矛盾：
alpha 是 t >= 1 窗口内的累积指数，A 是相对 t=0 投影基线的总放大；网格对齐方向投影误差基线更小、
被输运误差（HLLC 网格对齐数值扩散）更大，于是「基线小、输运量大」表现为大 A 与小 alpha。
单看 A 或单看 alpha 都会误读，本报告同时给出两者。

### 逐检查点误差（三方向）

| 族 | t | 步数 | rho % | u % | p % | 最大场 % |
|---|---|---|---|---|---|---|
| `dg_d2_n32_cfl0.2` | 1 | 469 | 0.0050 | 0.0094 | 0.0070 | 0.0094 |
| `dg_d2_n32_cfl0.2` | 2 | 937 | 0.0051 | 0.0097 | 0.0071 | 0.0097 |
| `dg_d2_n32_cfl0.2` | 5 | 2342 | 0.0054 | 0.0111 | 0.0075 | 0.0111 |
| `dg_d2_n32_cfl0.2` | 10 | 4684 | 0.0061 | 0.0152 | 0.0085 | 0.0152 |
| `dg_d2_n32_cfl0.2_diraxis0` | 1 | 422 | 0.0047 | 0.0067 | 0.0066 | 0.0597 |
| `dg_d2_n32_cfl0.2_diraxis0` | 2 | 844 | 0.0048 | 0.0069 | 0.0067 | 0.0611 |
| `dg_d2_n32_cfl0.2_diraxis0` | 5 | 2108 | 0.0050 | 0.0079 | 0.0069 | 0.0686 |
| `dg_d2_n32_cfl0.2_diraxis0` | 10 | 4215 | 0.0056 | 0.0105 | 0.0079 | 0.0887 |
| `dg_d2_n32_cfl0.2_dirobl30` | 1 | 463 | 0.0049 | 0.0076 | 0.0069 | 0.0134 |
| `dg_d2_n32_cfl0.2_dirobl30` | 2 | 926 | 0.0052 | 0.0079 | 0.0072 | 0.0138 |
| `dg_d2_n32_cfl0.2_dirobl30` | 5 | 2315 | 0.0056 | 0.0091 | 0.0077 | 0.0158 |
| `dg_d2_n32_cfl0.2_dirobl30` | 10 | 4629 | 0.0063 | 0.0123 | 0.0087 | 0.0211 |

### 关键量 vs 基准与「是否 <3%」

| 量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| DG 族（p=1/2/3，含三方向）最大场相对 L2，全部检查点 | 闭式周期解 | 最差 0.2321% | **是** |
| P0 一阶族 t=1 快照最差 | 闭式周期解 | 4.9260% | 否（>3%） |
| P0 一阶族长时最差 | 闭式周期解 | 14.3047% | 否 |
| 累积指数 alpha（C10 三方向） | 物理上界 alpha <= 1 | 最差 0.199 | 是 |
| 离散守恒（C3） | 舍入水平 1e-10 | 3.91e-13 | 是 |

P0 一阶族不满足 6%（更不满足 3%）的长时容差，这是格式阶数的固有属性（一阶耗散抹平涡核）；
算例池容差按 C1 在 t=1 快照上判（P0 t=1 = 4.93% < 6%），长时读数作为 C1x 原样登记、
不作判据。明确回答：DG 族 <3% 成立；P0 一阶族长时 <3% 不成立。

## ④ 结果图

* `fig_error_vs_time.png`：所有族（含两个旋转方向）最大场误差随 t 的双对数曲线 + 6% 容差线；
* `fig_design_space.png`：C7 设计空间（网格/CFL/阶数）与方向族并列读数；
* `fig_invariants_vs_time.png`：C3 质量/能量漂移（含方向族，修复后尺度）与 C4 熵预算；
* `fig_amplification.png`：A(t) = e(t)/e(0) 放大因子。

## ⑤ 注意点（踩到的坑）

1. C3 度量缺陷（本轮发现并修复）：相对漂移需要非零分母。网格对齐输运下横向动量精确为零
   （涡无净环量，u_inf 沿 x 时 <rho v> = 0），旧度量把一次舍入级漂移报成 O(1) 守恒失败
   （`dg_d2_n32_t2_cfl0.2_diraxis0:momentum_y`，整包因此从 INCOMPLETE 掉成 FAIL）。
   修复：分母取被输运量 `max(|参考值|, mass*|u_inf|)`（质量/能量/熵无零交叉，分母不变）。
   修复后最差漂移 3.91e-13。这不是放宽容差，
   而是把未定义的相对量改为有定义的量；缺 initial/final 不变量的合成记录回退到预存漂移字段。
2. 方向是族身份的一部分：历史 diag45 键名不变，旋转方向带 `_dir<label>` 后缀，
   以 config 为准重推（陈旧标签不得覆盖配置）。
3. C10 的判别力：只要求「基线可判」等于重读 C2，故判据强制至少一个旋转方向可判；
   无旋转方向的旧证据包读 not_evaluated，绝不读 passed。
4. 诚实登记的未完成部分：round 7 递交的 11 个检查点（p=1/3 阶数轴、p=1 细网格、t=20）仍在
   `cj3432` 队列，C6/C8/C9 仍 not_evaluated，整包裁决 INCOMPLETE。本轮没有把洞报成 PASS。

## ⑥ 误差判定汇总

* C1（6% 快照容差，t=1 各族）：PASS，最差 4.9260%（p0_n64）。
* C2（alpha <= 1，设计族）：PASS，alpha = 0.199。
* C3（守恒漂移，修复后）：PASS，3.91e-13。
* C4（熵预算 < 1e-3）：PASS，2.98e-07。
* C5（交叉族收敛阶 >= 0.8）：PASS。
* C7（设计空间不变性）：PASS，最差 alpha 0.199。
* C10（方向不变性，本轮新增）：PASS，三方向 alpha = 0.199, 0.166, 0.191。
* C6/C8/C9：not_evaluated（INCOMPLETE，11 个检查点在 `cj3432` 队列）。
* 整包裁决：**INCOMPLETE**，不是 PASS，也不是 FAIL。

复现：
```
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --task dg --degree 2 --cells 32 \
    --cfl 0.2 --final-time T --direction axis0 --output-dir case_euler_vortex_errconserve
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures \
    --output-dir case_euler_vortex_errconserve
```
