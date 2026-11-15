# AUDIT · `lax_scheme_variants` 工件的「陈旧未知」与闸门空断言

> 锚点 `CASE-LAX-DG`（ref: 精确解｜容差 < 3 %，status 仍 `open`）。
> 对象：`docs/benchmarks/results/lax_scheme_variants/`、`benchmarks/lax_scheme_variants.py`、`tests/test_lax_scheme_variants.py`。
> 本文所有数字取自磁盘上的腿记录（`legs/<arm>.json`）与工件（`metrics.json`），未手抄任何测量值。

## ① 发现的两条缺陷

### D1（严重）工件把已存在的测量报成 unknown —— 陈旧未知

`legs/cfl020.json` 在磁盘上**存在且是一次完整测量**（8746 步、`dt_mean = 1.4864e-05`、最差场量 L2 ρ = `3.2644e-02`），
但 `metrics.json` 里 `cfl020` 这一行仍是 `not_run: true`、`l2` 空缺、`margin: null`，
生成的 `docs/CASE-LAX-DG-scheme-variants.md` 也照抄成「未跑出」。

后果是**证据被低估且不可见**：集群作业回填了腿记录，但工件从未重新聚合，
于是讨论「时间离散是不是缺口」的那一节完全没有用上已经付过算力的数据。
这是方向安全（低报）但事实错误的记账——同一套机制在相反方向就会变成高报，
所以不能只当作笔误。

### D2（严重）闸门对「工件↔腿记录」的绑定是空断言

原 `test_arm_set_is_closed_and_pre_registered` 里的守卫是

```python
assert (LEGS / f"{row['variant']}.json").exists() or row.get("not_run")
```

对 `not_run` 行，右项恒真，所以左项写什么都没关系：**腿记录存在而工件说没跑**这一情形被闸门直接放过（即 D1）。
反向更糟：一个带有限 `l2` 的已测行根本不进入这个分支，**没有任何检查证明它有一个腿记录做来源**——
手工编造一行数字、或腿记录被删掉，闸门都不响。

换言之：闸门只断言「行数没变」，没有断言「每个数字都有出处」。

## ② 修复

1. **新增双向绑定闸门** `test_artifact_rows_are_bound_to_their_leg_records`：
   * 腿记录存在 ⇒ 该行不得是 `not_run`，且除派生字段 `decision` 外必须与腿记录**逐键相等**（禁止手改）；
   * 腿记录不存在 ⇒ 该行必须是 `not_run`（禁止无出处的测量）。
2. **删掉空断言**，把该处职责交给上面这条双向闸门。
3. **重新聚合**：`--aggregate` 目前把 `cfl020` 的真实数字并入工件（`ao` 仍无腿记录，如实记 `not_run`）。

## ③ 重新聚合后的实质结论（时间误差 vs 空间误差）

`cfl020` 的数字一进表，原本悬空的「时间探针」问题就有了答案。三条臂只动 `dt`：

| 臂 | 角色 | CFL | dt_mean | 最差场量 L2 |
|---|---|---|---|---|
| `cfl020` | temporal probe | 0.20 | `1.4864e-05` | 3.2644 % |
| `base` | control | 0.35 | `2.5445e-05` | 3.1156 % |
| `cfl050` | temporal probe | 0.50 | `3.6333e-05` | 3.0992 % |

* 覆盖 `dt` 的 **2.44×** 行程，最差场量 L2 只扫过 **0.1651 个百分点**（= 容差的 5.50 %）；
* **误差随 `dt` 变小而单调变大**，最小 `dt` 的臂仍未过 3 % 线 ⇒ **只减小 `dt` 不能达标，缺口不在时间离散**；
* 这个方向本身就是信息：整场 L2 由间断处的振荡幅度主导，时间耗散在压制它，
  所以「把 CFL 拧小」只会让振荡更完整地暴露出来。这与 §可行性界给出的「solver 腿是达标上界的 1.0133 倍」互相印证。

于是配置空间的裁定从「已测臂无一可用」升级为：**`dt` 这根旋钮的可达范围被实测扫完，整段都在门上方**。

## ④ 仍未闭合的部分（honest）

* `ao`（高阶 WENO 限幅臂）**至今没有腿记录**：作业 `cj3724` 递交后长时间排队（>24 min，集群在飞 89 个作业），
  本单结束时仍未起跑。它在工件里如实记为 **unknown，不是 pass**；`verdict.scope_complete = false` 也说清了这一点。
* `pos_none` 是**有腿记录的崩溃**（`FloatingPointError: Nonphysical high-order Euler state; reduce CFL`），
  属「正性兜底不是可选项」的实测结论，不是未测。
* 台账判定**没有变**：可用臂仍为空，`verdict = FAIL`；本单只是把证据补齐、把闸门修硬，不改口径、不改容差。

## ⑤ 复现

```
PYTHONPATH=. python -m benchmarks.lax_scheme_variants --aggregate --allow-missing
PYTHONPATH=. python -m benchmarks.lax_scheme_variants --figures --report --check
python -m pytest tests/test_lax_scheme_variants.py -q     # 15 passed
```

D1/D2 的现场证据（修复前）：
`AssertionError: cfl020: leg record exists on disk but the artifact reports it as 'not run' -- stale artifact, re-run --aggregate`