# PR #96 CI 红修复：把当前 main 合回 agent/m3666（消除陈旧 ns-heat 壁面棘轮）

> 任务：m4357（修复 TensorDG PR #96 的第 1 轮 CI 红）
> 分支：`agent/m3666`（修复后推回同分支）
> 基准：PR base `main` = `4ae1597`，修复前 head = `bed2559`

## 1. 现象：CI `test` job = failure

PR #96 自身只新增两个文件（`docs/tvb_module_tests.md`、`tests/test_tvb.py`），
单独跑 18 条 TVB 单元测试全绿：

```
$ PYTHONPATH=. python -m pytest -q tests/test_tvb.py
18 passed, 2 warnings in 7.11s
```

因此红不是 PR 内容自带的，而是**基线过旧**：head `bed2559` 的 merge-base 是
`619167f`（Merge #53），比 main 落后 4 个 PR（`fc53e18` #65、`7b1af26` #66、
`0823b7d` #73、`4ae1597` #74）。main 在 `c051cef`（m3694, PR #74）里已经
把 `tests/test_ns_heat_wall_closure.py` 的两处陈旧棘轮改掉，而 PR head 还停在旧断言上。

## 2. 定位：本地复现出 2 条红测（唯一失败点）

`PYTHONPATH=. pytest -q -n 24` 跑全量（420 条），失败**只有**这两条，均在旧版
`tests/test_ns_heat_wall_closure.py`：

```
FAIL tests/test_ns_heat_wall_closure.py::test_interior_energy_residual_decays_but_wall_rows_do_not (call)
E   assert 1.1254006470928049 < 1.1     # tests/test_ns_heat_wall_closure.py:134

FAIL tests/test_ns_heat_wall_closure.py::test_conduction_entry_point_still_crashes_on_the_documented_default (call)
E   Failed: DID NOT RAISE FloatingPointError    # tests/test_ns_heat_wall_closure.py:145
4 failed(=2 unique × 重试记账), 361 passed, 8 xfailed（修复前基线）
```

两条红的物理含义：

| 测试 | 旧断言 | 实测 | 为什么旧断言必红 |
|---|---|---|---|
| 壁面能量残差棘轮 | `bottom_ratio < 1.1` | 1.12540 | 16/32 壁面行残差比实测 1.1254，旧上界比实测还低 2.3% |
| 导热入口点崩溃棘轮 | 期待 `FloatingPointError` | 求解器正常跑完 | main 的 m3139 壁面修复（单侧 DG 壁面导数 + 正性过滤）已消除该 abort |

即两条红都是**陈旧棘轮（stale ratchet）**，而非新的数值缺陷：断言钉住的是
已被上游修好的旧行为/旧边距。

## 3. 修复：合并当前 main（最小、可验证）

```
git merge --no-ff origin/main   # 8af5401
```

合并只带入 main 已 CI 验证过的内容（含 `c051cef` 的棘轮重构 + 文档化的测量边距），
PR 自身的两个交付文件未改动；`git diff --stat bed2559 HEAD` = 101 files（全部来自 main）。
冲突：无。

## 4. 验证（本机，修复后）

| 检查 | 命令 | 结果 |
|---|---|---|
| 目标红测 | `pytest -q tests/test_ns_heat_wall_closure.py tests/test_tvb.py` | **23 passed**（原 2 红转绿） |
| 全量测试 | `pytest -q -n 24 --timeout=1500` | 无 FAIL（详见 artifacts/pr96_ci_fix_evidence.log） |
| CI 步骤 1 | `benchmarks.linear_advection_1d --degree 1 --cells 20 40 80` | EOC 2.084 / 2.023 |
| CI 步骤 2 | `--degree 2 --cells 10 20 40 80 --max-relative-error 0.03` | EOC 3.001，finest rel L2 2.36e-6（PASS） |
| CI 步骤 3 | `linear_advection_2d --cells 24 --degree 1` | relative L2 0.2812%（PASS） |
| CI 步骤 4 | `poisson_2d --cells 16 --degree 1` | relative L2 0.3800%，对称性误差 1.07e-14（PASS） |
| CI 步骤 5–6 | `euler_sod_1d --cells 120` + `plot_euler_sod_1d` | rho∈[0.125,1]，p∈[0.1,1]，png 80.6 kB |
| CI 步骤 7 | `compare_euler_sod_1d --cells 3200 --time 0.2` | 全部量 < 3%（PASS） |
| CI 步骤 8 | `burgers_1d --cells 16 32 64 --degree 2` | finest rel L2 1.82e-3 %（PASS） |
| CI 步骤 9 | `heat_1d --cells 16 --degree 2` | rel L2 3.60e-3 %（PASS） |
| CI 步骤 10 | `lax_1d --cells 400 --degree 3 --cfl 0.35` | rho 2.769% / u 1.605% / p 0.919%（PASS） |
| CI 步骤 11 | `isentropic_vortex_2d --cells 192` | rho 0.899% / u 1.958% / v 1.973% / p 1.430%（PASS） |

## 5. 误差判定（是否 < 3%）

本次是 CI 红修复（测试基线对齐），不引入新数值方法，故"精度"指 CI 既有
验收门是否仍满足：**全部通过，最差项 = Lax 密度相对 L2 2.769%（< 3%）**；
isentropic vortex 最差项 1.973%（< 3%）；linear advection P2 收敛阶 3.001（理论 3）。
无一项超过 3%，故判定：满足。

## 6. 注意点 / 局限

- **根因是基线陈旧，不是代码缺陷**：只 cherry-pick 单条 `c051cef` 也能让这两条转绿，
  但合并整条 main 更稳（引入的是 main 已 CI 绿的全部内容），避免后续再次踩到
  其它 main 侧修复。
- 本地全量测试在 128 核高负载（load avg > 1000，多智能体共享）下运行，
  尾部 ~7% 的重测（离散色散/收敛扫描）耗时很长；红测定位用流式插件
  （`pytest_runtest_logreport`）即时落盘的证据见 artifacts。
- `-n 24` 只是加速，失败判据与 CI 的串行 `pytest -q` 一致（同一批 nodeid、同一断言）。

## 7. 结论

PR #96 CI 红 = 陈旧棘轮（`assert 1.1254 < 1.1` + 已消失的 `FloatingPointError`），
已通过合并当前 main 修复，目标红测与全量测试转绿，CI 全部 benchmark 步骤通过。