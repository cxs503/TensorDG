# PR #67 修复报告：`limiter_stage` 贯通 2-D 涡算例 + 测试判据纠正

任务：TensorDG#67（分支 `agent/m3627`）CI 红（test=failure）。本单（m4076）第 2 轮修复。

## 1. 红的根因（两处，第二处才是真凶）

| 层 | 症状 | 证据 |
|---|---|---|
| ① 接口未贯通 | `TypeError: solve_euler_dg_2d_vortex() got an unexpected keyword argument 'limiter_stage'` → 3 failed / 1 passed | `artifacts/pr67_ci_fix/red_before_fix.log` |
| ② 测试判据本身站不住 | 打通接口后仍红：`AssertionError: 'every' EOC (3.042) should be < 'final' EOC (3.042)` | 见 §3 实测表 |

① 是显式缺陷：`solve_euler_dg_2d` 已支持 `limiter_stage`（"every"/"final"），但
`solve_euler_dg_2d_vortex` 没有转发该参数。修复 = 加一个默认 `"every"` 的形参并透传
（8 行，向后兼容）。

② 是实验设计错误：原测试断言「`limiter_stage="every"` 会毁掉设计阶，`"final"` 能恢复」，
用 TVB c = 10 的二维等熵涡。但 c = 10 是**尺度一致**的 TVB 常数，该光滑流动中
无胞被判定为 troubled，限制器完全不激活 → 两种策略**逐位相同**（3.042 = 3.042），
断言必然失败。若把常数放到 c = 0（严格 TVD，处处限制），限制器本身就主导误差，
两种策略的观测阶都塌到 ≈1.07，仍然分不开。因此「every 毁阶」在**这个算例 + 这个限制器**
上根本不成立，判据必须重写。

## 2. 改了什么

- `tensordg/euler_dg_2d.py`：`solve_euler_dg_2d_vortex(..., limiter_stage="every", ...)`
  转发给 `solve_euler_dg_2d`，docstring 说明两种语义。
- `tests/test_euler_dg_2d_limiter_stage.py`：
  - 保留：参数校验 / 无限制器时两策略等价 / metadata 回显；
  - 保留并明确：c = 10 时 `"final"` 的 P2 等熵涡保持 EOC > 2.7（实测 3.042）——这是
    「新参数不得伤害精度」的真正护栏；
  - 新增 `test_limiter_stage_reaches_the_rk_substeps`：c = 0（严格 TVD，限制器一定激活）
    下两策略必须给出**不同**状态，证明标志真的传到了 SSP-RK3 子步（若标志被忽略两者逐位相同）。
  - 删除原「every 必须比 final 差」的伪判据（实测不支持）。
- 新增证据/脚本：`artifacts/pr67_ci_fix/{red_before_fix.log, green_after_fix.log,
  tvb_sweep.json, measure.py, probe_tvb_constant.py, run_evidence.py, figs/}`。

## 3. 实测（P2 等熵涡，Hu & Shu，域 [-5,5]²，T = 1，密度相对 L2 误差，N = 8 → 16）

| TVB 常数 / 策略 | ρ 误差 N=8 | ρ 误差 N=16 | EOC |
|---|---|---|---|
| c = 10，`"every"` | 3.275075e-03 | 3.975616e-04 | **3.042** |
| c = 10，`"final"` | 3.275075e-03 | 3.975616e-04 | **3.042** |
| c = 0，`"every"` | 4.267281e-02 | 2.027499e-02 | **1.074** |
| c = 0，`"final"` | 4.266176e-02 | 2.039001e-02 | **1.065** |

（`artifacts/pr67_ci_fix/tvb_sweep.json`，由 `measure.py` / `probe_tvb_constant.py`
程序化写出，图中含 h⁻³ 与 h⁻¹ 参考斜率。）

结论：设计阶 p+1 = 3 由**限制器的尺度一致性**决定，而非 RK 子步限制策略；c = 0 时
两策略误差差 2.6e-4（相对），说明策略确实生效但被限制器主导。

## 4. 误差判定

- EOC(c=10，final) = 3.042 vs 设计阶 3：相对偏差 **1.4% < 3%**（守恒量级：密度相对 L2）。
- 「两种策略在 c = 10 下等价」：误差差 = 0（逐位相同），**0% < 3%**。
- c = 0 下两策略差 2.6e-4（**< 3%**）——说明该常数下策略分离度极小，故不能作为判据。
- 局限：**没有**在带激波/欠分辨流动上给出「every 毁阶」的定量证据（该环境下每个
  c = 0/8×8 求解在 1000+ 负载机器上需 30–240 s，未做更贵的扫描）。因此本单不宣称
  任何「policy 提高激波分辨率」的结果。

## 5. 验证

```
python -m pytest -q tests/test_euler_dg_2d_limiter_stage.py
# 修复前（2a801dc）：3 failed, 1 passed  -> artifacts/pr67_ci_fix/red_before_fix.log
# 修复后（本提交）  ：5 passed in 112.28s -> artifacts/pr67_ci_fix/green_after_fix.log
```

## 6. 注意点

- 该测试文件的耗时主要在两处等熵涡求解（N = 8 → 16，T = 1，约 117/234 步）；修正后
  比原版少两次求解（原版 4 次，现版 2 次），CI 时间反而下降。
- 复现时须给 `PYTHONPATH=.`（CI workflow 已设置），否则 `benchmarks` 不可导入。
- 机器负载 >1300 时单次 N=16 求解可长达 240 s，local 复跑请留足预算。