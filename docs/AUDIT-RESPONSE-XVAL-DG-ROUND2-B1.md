# 返修记录 — XVAL-DG-ROUND2-CERTIFIED-REFERENCE 审计阻断项 B1/M1 关闭

本文件记录对 `docs/AUDIT-m17539-XVAL-DG-ROUND2.md`（审计单 m17539，被审单 m17539/引擎交付）
所开阻断项与中危项的关闭情况。数字全部由命令产出，落盘命令附在 §3，可逐条复跑。

## ① 背景与判据

锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（TensorDG 的 round2-certified-reference 做法）的
数值实质已由两轮审计独立复算确认（认证参考解跨实现差 0.0115 % < 预注册 1.00 % 判据 D1；
11/11 条 DG 腿 < 5 % 门且 < 3 %），但审计同时判 **L3/L4 阻断**，原因不是数字，而是**可达性与登记**：

| 发现 | 级别 | 内容 |
|---|---|---|
| B1 | 阻断 L3 | `tensordg.certified_reference` 的公开符号**全部未从包根导出**：`from tensordg import certify` → `ImportError`；仓自契约门 `tests/test_public_api_contract.py` RED，15 个未导出符号中 11 个来自本模块 |
| M1 | 中 | 模块 docstring 把契约门 RED **归因于历史**（"for reasons that predate this module"），实测 73 % 缺口由该模块引入 → 表述失衡 |
| B2 | 阻断 L4 | 锚点在 `tensor_benchmark_cases` **零登记**（0 行）→ 机器裁判 `CLAIMED_NOT_PROVEN` |

本单判据（开工前写定，机器可裁）：
**G1** 契约门覆盖率 = 1.0 且 `tests/test_public_api_contract.py` 全绿；
**G2** `from tensordg import certify` 可用且与源符号**同一对象**（不是同名替身）；
**G3** 与 `tensordg.convergence` 冲突的五个名字**不得被覆盖**（euler/swe 的 disambiguation 先例）；
**G4** M1 的因果表述改为可核对的事实陈述；
**G5** 锚点登记进 `tensor_benchmark_cases`。

## ② 改动（最小、可验证）

1. `tensordg/__init__.py`
   * 新增 `# ---- certified_reference` 导出块：六个独名（`certify`/`judge`/`classify`/`digest_levels`/
     `audit_ladder`/`band_coverage_study`）裸导出；五个与 `convergence` 撞名的
     （`coarsen`/`rel_l1`/`observed_order`/`richardson_extrapolate`/`gci`）以 `certified_` 前缀导出，
     与仓内既有约定（`euler_sound_speed` / `swe_sound_speed`、`ice_edge_observed_order`）一致。
   * 同一门禁下补齐另两处历史缺口（否则覆盖率到不了 1.0）：`convergence` 的 5 个
     （`exponential_decay_criterion`/`semidiscrete_operator`/`operator_linearity_residual`/
     `expm_scaled_squaring`/`exact_time_solution`）与 `cases.prefinement_exponential` 的 7 个
     （泛名加 `prefinement_` 前缀，避免 `initial_condition`/`exact_solution` 污染根命名空间）。
2. `tensordg/certified_reference.py`：重写 "Reachability note" 段——如实写明该模块**曾是**契约门 RED
   的主要贡献者（15 缺口中 11 个），并列出当前导出方案（M1 关闭）。
3. `tests/test_public_api_contract.py`：`CASES` 注册表断言补上已发布的 `prefinement_exponential`
   条目（原来该门断言的是过期注册表，属**测试滞后**而非案例错）。
4. `tests/test_certified_reference.py`：新增 `test_the_engine_is_reachable_from_the_package_root`，
   把 G2/G3 钉成回归门（此前没有任何测试阻止"引擎再次变成库内孤岛"）。
5. 锚点登记：`tensor_benchmark_cases` 写入 `anchor_slug='XVAL-DG-ROUND2-CERTIFIED-REFERENCE'` 一行
   （B2 的登记半边；见 §4 未完成项）。

## ③ 前后对照（命令产出，非手抄）

| 量 | 修复前（main） | 修复后 | 判定 |
|---|---|---|---|
| 契约覆盖率 `coverage.ratio` | 0.9439（387/410 符号） | **1.0000（410/410 公共符号；`__all__` 共 416 项）** | G1 ✔ |
| 未导出符号数 | 23（其中 certified_reference 11、convergence 5、prefinement 7） | **0** | G1 ✔ |
| `tests/test_public_api_contract.py` | 3 failed, 24 passed（export / doc snippets / CASES） | **27 passed** | G1 ✔ |
| `from tensordg import certify` | `ImportError` | 可用；`tensordg.certify is tensordg.certified_reference.certify` | G2 ✔ |
| `tensordg.observed_order` 归属 | `convergence`（无冲突保护） | 仍 `convergence`；认证版为 `certified_observed_order` | G3 ✔ |
| `docs` 中 `from tensordg import certify` 片段 | 解析失败（doc-snippet 门 RED） | 解析通过 | ✔ |
| 两文件测试合计 | 18 + 3 failed | **46 passed** | ✔ |

复跑命令：

```
python -m pytest tests/test_public_api_contract.py tests/test_certified_reference.py -q   # 46 passed
python -c "from tensordg import certify, judge; print(certify, judge)"                    # 可导入
python -c "import json,importlib.util as u; s=u.spec_from_file_location('c','tools/api_contract.py');\
m=u.module_from_spec(s); s.loader.exec_module(m); print(m.audit()['coverage']['ratio'])"   # 1.0
```

## ④ 未完成项（如实登记，不记 passed）

1. **`tensorbench_results` 行未落盘**：`POST /api/v1/tensor/bench/results` 要求绑定
   `cluster_job_id` + `spec`（机器门据此判 `source_cluster_job` / `cluster_job_bound`）。
   本单是**库内可导入性与登记**修复，没有新的集群作业，硬造 job id 属造假，故不落该行，
   L4 机器门状态仍为 `CLAIMED_NOT_PROVEN`，须由一条**真跑过认证阶梯的集群作业单**补齐。
2. **第二个物理算例（审计发现 L1：泛化性只有 Shu–Osher 一个物理算例支撑）本单未做**：
   起因是本单准备以 Lax/Toro-2 冲击管（有闭式 Riemann 解，可把"带是否覆盖真误差"从合成阶梯
   升级到物理算例）作第二算例，开工后发现**同池 m17925 正在跑 `benchmarks/lax_asymptotic_probe`**
   （同一仓同一算例），按 owner 2026-11-09「不重复」令停止该支线并删除未验证草稿，
   避免两单产出同名工件。L1 的缺口因此**仍然存在**，建议由 m17925 或其后续单继承。
3. 本单未重跑任何认证阶梯数值（不需要，也刻意不碰数字面）：数值结论仍由
   `docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md` 与 m17443/m17539 两轮审计背书。

## ⑤ 误差判定（回答「是否 < 3 %」）

本单是**可达性/登记**修复，不产生新的物理量，因此没有新的误差判定对象。
被修复能力的既有误差结论不变并可复核：认证参考解跨实现差 **0.0115 %**（<3 %，亦 < 1.00 % 判据），
DG 腿最差 **2.4557 %**（<3 %，对 5 % 门余量 50.9 %），参考解自身不确定带 **0.0828 %**。
本单交付的门是**布尔门**（覆盖率 1.0 / 测试全绿 / 导入可用），判据 G1–G5 的达成情况见 §3、§4。