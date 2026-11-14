# Sod 1978 冲击管 — 认证参考引擎的第二个物理算例

锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`。审计 `docs/AUDIT-m17539-XVAL-DG-ROUND2.md` 的 L1 发现
（"通用性只有一个物理算例支撑、带的覆盖率只由合成阶梯验证"）在本文件关闭。

## ① 案例
`(rho,u,p)_L=(1,0,1)` / `_R=(0.125,0,0.1)`，`x_0=0.5`，`gamma=1.4`，域 `[0,1]`，`t=0.2`。
关键差别：**有闭式 Riemann 解**，因此认证参考解的"真误差"可测，而不仅是自洽。

## ② 怎么做的
* 认证者阶梯：FV-WENO5-JS/HLLC(Davis)/SSP-RK3，N=800/1600/3200，CFL=0.3（与 Shu–Osher 同族同 CFL）；
* 候选腿：DG P1/P2/P3 × N=200/400/800，WENO 限制器，CFL=0.4，共 9 条；
* 引擎不打补丁：`certify/judge/classify/rel_l1` 原样 import（若需算例专属修改，"通用"主张即为假）；
* 外部神谕：`benchmarks.riemann_exact`（Toro 精确 Riemann 解，8 点单元平均）+ 闭式星区/激波速度；
* 集群作业 `cj2724`（n315）产出全部数字：`python -m benchmarks.certified_reference_sod --json-stdout --dump-profiles`；
  装置输出 `artifacts/sod_certified_reference/sod_certified_reference.json`（本次 stdout 与工件文档内容一致，stdout 文件未落盘，
  ingest 以该工件作为 workload 文档，见 ⑤.4 如实登记）。

## ③ 结果（程序化注入，零手抄）
| 量 | 值 |
|---|---|
| 观测阶 p（FV 阶梯） | 1.0163 |
| 认证带 U | 0.037263 %（占 5 % 门 0.745 %） |
| 判决 | **CERTIFIED** |
| 认证参考解真误差（vs 精确解） | **0.022741 %** |
| 真误差 / U | 0.6103 |
| 带是否包住真误差 | **True** |
| 原始最粗级（N=3200）真误差 | 0.031960 % |
| DG 腿 @3 % 门 | {'PASS': 9} |
| DG 腿 @5 % 门 | {'PASS': 9} |

### 解析不变量（在认证参考解上测量）
| 不变量 | 精确值 | 实测 | 相对误差 % |
|---|---|---|---|
| conserved_mass | 0.5625 | 0.5625 | 0.000000 |
| plateau_density | 0.26557371 | 0.26557254 | 0.000442 |
| star_left_density | 0.42631943 | 0.42631803 | 0.000327 |
| shock_position | 0.85043115 | 0.850625 | 0.022795 |

### 误差按解析窗口分解（认证参考解 vs 精确解）
| 窗口 | rel L1 % |
|---|---|
| rarefaction | 0.003456 |
| star_left | 0.002308 |
| star_right | 0.005682 |
| untouched_right | 0.000000 |

### DG 腿逐条
| 配置 | rel L1 % | @5 % | @3 % |
|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 0.5244 | PASS | PASS |
| DG P1 N=400 CFL=0.4 | 0.2793 | PASS | PASS |
| DG P1 N=800 CFL=0.4 | 0.1504 | PASS | PASS |
| DG P2 N=200 CFL=0.4 | 0.5777 | PASS | PASS |
| DG P2 N=400 CFL=0.4 | 0.3041 | PASS | PASS |
| DG P2 N=800 CFL=0.4 | 0.1630 | PASS | PASS |
| DG P3 N=200 CFL=0.4 | 0.5184 | PASS | PASS |
| DG P3 N=400 CFL=0.4 | 0.2738 | PASS | PASS |
| DG P3 N=800 CFL=0.4 | 0.1474 | PASS | PASS |

## ④ 图
`fig1_profiles.png`（认证阶梯与 DG 腿 vs 精确解）、`fig2_window_errors.png`（窗口误差 vs 带 U vs 真误差）。

## ⑤ 注意点（踩坑与局限）
1. **引擎无法识别"自证"**（实测）：把 DG 阶梯本身递给 `certify`，三族都被判 CERTIFIED（p≈0.92、带占门 <4 %）：

| DG 阶梯当认证者 | 判决 | p | U/门 |
|---|---|---|---|
| dg_p1 | CERTIFIED | 0.9180 | 0.0375 |
| dg_p2 | CERTIFIED | 0.9372 | 0.0398 |
| dg_p3 | CERTIFIED | 0.9225 | 0.0366 |

   引擎的拒证规则是纯数值的（阶/停滞/舍入地板/带吃门），无从得知"这条阶梯与被判的腿同源"。选择异族认证者是
   **调用方义务**；机器可检的唯一防线是外部神谕（判据 C3）。这一点在 artifact 与测试里显式登记，不粉饰。
2. `_shock_position` 需要搜索下界：Sod 的**接触间断**密度跳变（0.4263→0.2656）大于激波（0.2656→0.1250），
   全域 argmax 会定位到接触面；代码以解析接触位置 `x0+u* t` 作下界，位置本身仍是测量值。
3. 精确解单元平均在跨间断的两格上有 O(dx) 残差（质量积分相对偏差 ~3e-5），故测试按 `rel<1e-4` 断言。
4. 集群节点无 matplotlib，第一次作业（cj2705）在画图处崩溃而丢失 stdout；已改为 try/except 降级 +
   `--dump-profiles` 落 `profiles.npz`，图可离线重建。cj2724 的 stdout 亦未落盘，故 ingest 直接以
   同一次作业写出的 `sod_certified_reference.json` 作为 workload 文档（同一文档，非再造数据）。
5. 只覆盖 1D Euler 标量密度场的认证；不涉及多维/湍流/粘性算例。

## ⑥ 误差判定（是否 < 3 %）
| 量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| 认证参考解 | 精确 Riemann 解 | 0.022741 % | **是** |
| 后激波平台密度 | 闭式 rho*_R=0.2655737 | 0.000442 % | **是** |
| 星区左密度 | 闭式 rho*_L=0.4263194 | 0.000327 % | **是** |
| 激波位置 | x0+Wt=0.850431 | 0.022795 % | **是** |
| 质量积分 | 0.5625 | 1.236e-11 % | **是** |
| 9 条 DG 腿 | 认证参考解 | 最差 0.5777 % | **是（9/9）** |

判据 C1–C5 全 PASS，`accepted = True`。Sod 1978 原文只给图不给数表 → 逐点文献对比 **WITHHELD**。
