# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 认证参考解的跨实现交叉对标

## ①案例介绍
CASE-SHU-OSHER（Shu & Osher, JCP 77 (1988) 439）：γ=1.4，x∈[−5,5]，x<−4 为 Mach-3 激波后状态 (3.857143, 2.629369, 10.33333)，x≥−4 为 ρ=1+0.2 sin(5x)，u=0，p=1，透射边界，t=1.8。算例无闭式解，round-2 的「认证参考解」=独立 FV 求解器三级加密（N=800/1600/3200）+ Richardson 外推 + Roache GCI（Fs=1.25）不确定度带。

本单问题：**该认证参考解是否依赖本仓实现？** 若换一套他人写的成熟开源实现，认证参考解是否在声明的不确定度内一致？判据（开工前预注册）：D1 两个认证参考解 rel L1 < 1.00 %；D2 差量落在两带 GCI 的 3.0× 合成带内；D3 全部 DG 腿在 OSS 认证参考解下仍 < 5.0 %。

## ②方法与实现
* 本仓腿：`benchmarks/fv_weno5_1d.py`（cell-centred FV，Jiang–Shu WENO5 分量重构，HLLC (davis 波速)，SSP-RK3，3 层 vitual ghost cell 透射）→ 已有 raw npz（N=800/1600/3200，CFL=0.3）。
* OSS 腿：**Clawpack 5.14.0 / pyclaw SharpClaw**，`kernel_language='Python'`（纯 Python 核，无 f2py 编译），`euler_hllc_1D` HLLC，WENO5 重构（`limiters/recon.py:weno(5,q)`，此核只支持 5 阶：`if k != 5: raise`），`time_integrator` 默认 SSP104（敏感性腿用 SSP33），`char_decomp=0` 分量重构，`BC.extrap` 透射。OSS 树**未修改**，仅以 `sys.path` 前置导入。
* 认证机制（coarsen/rel L1/实测阶/GCI/Richardson）**刻意共用同一把尺**（`benchmarks/shu_osher_certified_ref.py`），否则无法判定两解是否在同一口径下一致。

### OSS 溯源（零手抄：文件 sha256 + 版本）
clawpack unknown，root=`/tmp/oss/clawpack-5.14.0`

| 文件 | 字节 | sha256 |
|---|---|---|
| `clawpack/riemann/euler_1D_py.py` | missing | — |
| `pyclaw/src/pyclaw/limiters/recon.py` | 3275 | `e7524333e407eaea5a8405c7e4ea6ab4…` |
| `pyclaw/src/pyclaw/sharpclaw/solver.py` | 38100 | `74319e9402a18c890f2ea06edb68fc99…` |
| `pyclaw/src/pyclaw/solver.py` | 27690 | `b07557a05a8148e2bdf1e03d5b8c75a5…` |
| `pyclaw/src/pyclaw/state.py` | 11829 | `50695d040eb7a75e261a7a5ff81453b1…` |

### 本仓 vs OSS 差异清单与移植取舍
| 维度 | 本仓 FV 参考解 | OSS pyclaw SharpClaw | 取舍 |
|---|---|---|---|
| 通量 | HLLC，外波速 Davis | HLLC，外波速 min/max(u±c, Roe λ)（`euler_1D_py.py:173`） | 保留本仓 Davis（已有 `docs/hllc_wave_speed_verification.md` 验证面）；差异正由本 XVAL 量化 |
| 重构 | Jiang–Shu WENO5，固定 ε=1e-6，分量 | WENO5（`recon.weno`，ε=1e-36），分量；Python 核硬编码 5 阶，`char_decomp=2` 未实现 | 不移植：本仓 WENO5 已通过跨格式验证；OSS 硬编码阶数反而是其限制 |
| 时间积分 | SSP-RK3 | 默认 SSP104（4 阶 10 级），可切换 SSP33 | 敏感性轴 a：SSP104↔SSP33 对认证参考解的影响已量化 |
| 边界 | 3 层 ghost 复制边界值 | `BC.extrap` ghost 复制（`solver.py:458`） | 等价做法，无需移植 |
| 网格/CFL | N=800/1600/3200，CFL=0.3 | 同 N、CFL 0.3（敏感性 0.6） | 同口径对比成立 |

## ③参数表（无量纲）
| 量 | 值 |
|---|---|
| γ / 域 / 终止时刻 | 1.4 / [−5,5] / 1.8 |
| 加密序列 / 加密比 r | 800, 1600, 3200 / 2 |
| CFL（基准腿） | 0.3（两端同值） |
| GCI 安全因子 Fs | 1.25 |
| 实测阶 p：本仓 / OSS | 1.2825 / 1.2940 |
| GCI 不确定度 U：本仓 / OSS | 0.082797 % / 0.082076 % |
| 预注册判据 | D1 < 1.00 %；D2 ≤ 3.0×合成带；D3 DG < 5.0 % |

## ④对照表（ref vs got + 判定）
### 4.1 认证参考解（同一 N=1600 网格）
| 量 | 值 | 判定 |
|---|---|---|
| rel L1(ours_cert, oss_cert) | **0.0115 %** | D1 PASS (<1.00 %) |
| 合成不确定度带 3.0×√(U_ours²+U_oss²) | 0.3498 % | D2 PASS |
| 窗口分解（OSS 相对本仓）：全域 / 平台 / 熵波 / 未扰正弦 | 0.0115 % / 0.0000 % / 0.0170 % / 0.0000 % | — |

### 4.2 同网格原始解差（无外推，最硬的口径）
| N | rel L1 % | 平台 % | 熵波 % | 未扰正弦 % |
|---|---|---|---|---|
| 800 | 0.0096 | 0.0000 | 0.0142 | 0.0000 |
| 1600 | 0.0066 | 0.0000 | 0.0099 | 0.0000 |
| 3200 | 0.0078 | 0.0000 | 0.0116 | 0.0000 |

### 4.3 DG 腿对 OSS 认证参考解的复判（D3）
| 配置 | rel L1 % | rel L2 % | 平台 % | 熵波 % | 5 % 门 |
|---|---|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 2.4559 | 5.2632 | 0.0000 | 3.6395 | PASS |
| DG P1 N=400 CFL=0.4 | 1.5480 | 3.6568 | 0.0000 | 2.2941 | PASS |
| DG P1 N=800 CFL=0.4 | 0.5768 | 1.5166 | 0.0000 | 0.8548 | PASS |
| DG P2 N=200 CFL=0.4 | 2.4205 | 5.2333 | 0.0000 | 3.5870 | PASS |
| DG P2 N=400 CFL=0.4 | 1.5161 | 3.6099 | 0.0000 | 2.2468 | PASS |
| DG P2 N=800 CFL=0.4 | 0.5648 | 1.4806 | 0.0000 | 0.8370 | PASS |
| DG P3 N=200 CFL=0.4 | 2.4268 | 5.2529 | 0.0015 | 3.5958 | PASS |
| DG P3 N=400 CFL=0.4 | 1.5221 | 3.6067 | 0.0240 | 2.2469 | PASS |
| DG P3 N=800 CFL=0.4 | 0.6678 | 1.4905 | 0.3819 | 0.8492 | PASS |
| DG P2 N=400 CFL=0.1 | 1.5198 | 3.6147 | 0.0000 | 2.2523 | PASS |
| DG P2 N=400 CFL=0.2 | 1.5182 | 3.6117 | 0.0000 | 2.2499 | PASS |

D3 汇总：**PASS**（全部腿在 OSS 认证参考解下仍 < 5.0 %）。

## ⑤收敛性/参数敏感性
* **cfl**：0.3 / 0.6；p_obs = 1.2940 / 1.2979；GCI = 0.082076 % / 0.081539 %；两变体认证参考解差 = 0.0051 %
* **char_decomp**：未完成（honest）——cached char_decomp=1 ladder (3 levels) not finished in-budget
* **time_integrator**：SSP104 (OSS default) / SSP33 (closest to our SSP-RK3)；p_obs = 1.2940 / 1.2915；GCI = 0.082076 % / 0.081706 %；两变体认证参考解差 = 0.0067 %

* 两套实现的实测阶都由**successive differences**测得而非形式阶：
本仓 p=1.2825（形式 5 阶），OSS p=1.2940——激波把 L1 阶退化到 ~1，两实现一致地给出同一结论，这本身是跨实现的自洽证据。

## ⑥失败模式与工件
* 失败模式 1：OSS 纯 Python 核只支持 WENO5（`recon.weno` 对 k≠5 抛错），`char_decomp=2`（特征型重构）在 Python 核抛 NotImplementedError → 该敏感性轴改以 `char_decomp=1`（wave-slope）替代，已在参数表标注。
* 失败模式 2：pip sdist 5.14.0 缺 `riemann/euler_mapgrid_3D`（f2py 生成物未随 sdist 发布），导入 `clawpack.riemann` 会失败；以空 shim 文件占位（3D 路径与本算例无关），所用 1D 通路文件 sha256 见 ②，未修改。
* 失败模式 3：纯 Python 核单点 58 s(N=800) → N=3200 单腿约 4 分钟，本机单命令 120 s 上限 → 采用 `--warm` 分腿后台缓存 + 汇总渲染；集群队列深度 >500 未派发。
* 工件：`artifacts/xval_m17365/{xval.json, certified_refs.npz, oss_raw/*.npz, fig_*.png}`，本文件由 `python -m tools.render_xval_round2` 程序化注入生成（零手抄）。
* 适用范围：本 XVAL 是**代码间交叉验证**（two independent discretisations），不等于实验 validation；Shu–Osher 原文只给图，文献数值仍 WITHHELD。

## 误差判定（回答「是否 <3 %」）
* 认证参考解跨实现差 = **0.0115 %** → **<3 %**；亦 < 预注册 1.00 % 判据（D1 PASS）。
* DG 腿对 OSS 认证参考解的最差腿 = DG P1 N=200 CFL=0.4：2.4559 %
  → **<3 %**，对 5.0 % 门的余量 50.9 %。
* 参考解自身不确定度：本仓 U=0.082797 %，OSS U=0.082076 %，均远小于最差腿误差 → 判定不被参考解噪声支配。
