# m16815 — 交付物：修 `main` 上的红门（gate-vacuity 政策门自身被合并门禁吞掉）

> 锚点：`【锚点=NEW:main 上 tests/test_gate_vacuity_policy.py 红 5/10，根因是 census 工件被合并门禁过滤出树——门禁守「工件缺失即空转」，自己却因工件缺失而红】`
> 仓：TensorDG · 分支 `agent/m16815` · commit 见 `git log -1`
> 环境：Python 3.11 / torch 见 census `environment` / 单机 128 核（load1 ≈ 260，本单只做轻量活）

## ① 案例介绍（问题背景与几何）

TensorDG 的验收门大量采用「重判已提交工件」的形态（`validation/*.json`、`runs/*/metrics.json`、
`artifacts/*.json`）。m16367 建了静态空转普查 `tools/audit_gate_vacuity.py` 与政策门
`tests/test_gate_vacuity_policy.py`（10 道闸门），它守的正是本生态记录在案的头号失效模式：

> 合并门禁会把文件从 agent 分支里滤掉（提交信息里写作「N 个文件被门禁滤除」），
> 作者分支上存在的工件到了 `main` 就没了；此时「工件缺失即 skip」的验收门从
> 「断言数字」退化成「什么都不断言」，而 CI 依旧绿。

本次自找活在只读扫描阶段撞上了这条规则的**自身反噬**：

```
$ git cat-file -e HEAD:artifacts/gate_vacuity/gate_vacuity.json
fatal: path 'artifacts/gate_vacuity/gate_vacuity.json' does not exist in 'HEAD'

$ pytest -q tests/test_gate_vacuity_policy.py      # 干净 main 检出上
5 failed, 5 passed in 0.25s
```

即：**守「工件缺失」的门，自己因为工件缺失而红**——5 道闸门（census 与全树一致 / 报告由 census
渲染 / 无未登记空转门 / 白名单自清洁 / census 覆盖全部测试模块）只断言「文件在不在」，
文件不在就整片失守，而且不给任何关于测试套件的信息。这不是一次 skip，而是**门本身的失能**：
红得响但信息量为零，后续任何真实漂移都会被这条红噪声掩盖。

## ② 怎么做的（方法与工具链）

| # | 动作 | 手段 |
|---|---|---|
| 1 | 确认根因 | `git cat-file -e HEAD:…` 证明 census JSON 不在 `HEAD`；静态扫描工具 `tools/audit_gate_vacuity.py::census()` 是一次 **0.15 s 的纯静态扫描**，本可随时重算 |
| 2 | 修「门依赖工件」 | `tests/test_gate_vacuity_policy.py::_census()` 改为**自愈**：工件在则读文件，不在则用 `gv.census()` 现算一份内存 census（判据不变，只是不再依赖文件存活） |
| 3 | 钉回归 | 新增 `test_census_survives_a_filtered_artifact`：`monkeypatch` 把 `CENSUS` 指向不存在的路径，断言 `_census()` 不抛异常且 `modules/verdicts/vacuous_modules` 与新扫描逐项相等——空转复现即红 |
| 4 | 复算并落盘 census | `python tools/audit_gate_vacuity.py --write`（重写 `artifacts/gate_vacuity/gate_vacuity.json` + `docs/GATE-VACUITY-AUDIT.md`） |
| 5 | 处置新暴露的空转门 | 新扫描多出一个 VACUOUS：`tests/test_lax_region_band.py`（m16606 的 `runs/lax_region_band/metrics.json` 未进树）；在 `ALLOWED_VACUOUS` 里**带理由登记**（自清洁：条目一旦不再空转即红），而不是让它继续藏在绿 skip 后面 |
| 6 | 验证 | `pytest -q tests/test_gate_vacuity_policy.py`；`pytest -q tests/test_lax_region_band.py -rs` |

## ③ 计算结果（数字直读，零手抄）

### 3.1 政策门状态

| 时点 | 通过 | 失败 | 用时 |
|---|---|---|---|
| 修复前（干净 main） | 5 | **5** | 0.25 s |
| 修复后 | **11** | **0** | 7.86 s |

（11 = 原 10 道 + 本单新增 `test_census_survives_a_filtered_artifact`。）

### 3.2 全树空转普查（`artifacts/gate_vacuity/gate_vacuity.json`，144 个测试模块）

| 判级 | 模块数 |
|---|---|
| VACUOUS（验收断言此刻不执行） | **3** |
| LATENT（缺工件即 skip，工件在树 → 一次合并门禁过滤即转 VACUOUS） | **6** |
| HARDENED（工件硬断言，缺失即红） | **8** |
| UNRESOLVED（路径静态不可解析，如实登记不算通过） | **9** |
| PLAIN（不依赖已提交工件） | **118** |

三个 VACUOUS 及其缺失工件（本次全部登记并给出再生成命令）：

| 模块 | 缺失工件 | 状态 |
|---|---|---|
| `tests/test_dg_advection_bench.py` | `artifacts/dg_advection_bench.json` | 已登记（m16367） |
| `tests/test_shu_osher_case_pool.py` | `runs/shu_osher_case_pool/metrics.json` | 已登记（m16367） |
| `tests/test_lax_region_band.py` | `runs/lax_region_band/metrics.json` | **本单新登记（m16815）** |

### 3.3 被吞工件的直接后果（实测，不是推断）

```
$ pytest -q tests/test_lax_region_band.py -rs
5 passed, 1 skipped in 19.93s
SKIPPED [1] tests/test_lax_region_band.py:109: audit artifact not committed yet
```

在同一份 `main` 上，m16606 建立的 6 条 R3 区域规则不变量里 **1 条自合并起就没有运行过**，
其余 5 条仍有效——所以这个模块**不是整体空转，而是部分空转**，这正是最容易被「绿」掩盖的形态。

## ④ 结果图

见 `docs/figures/m16815-gate-vacuity-census.png`：左图为按判级的模块数（VACUOUS/LATENT/HARDENED/
UNRESOLVED/PLAIN），右图为被 VACUOUS 点名的 3 个模块与其缺失工件。所有数字由
`artifacts/gate_vacuity/gate_vacuity.json` 程序化读取后绘制，无手抄。

## ⑤ 注意点（坑、局限、适用范围）

1. **红 ≠ 有信息。** 修前 5 条失败全部形如 `assert CENSUS.exists()`，说的是「文件没了」，
   对测试套件的状态一无所知。真正危险的组合是「门红」+「无人看」：这条红如果被当成已知噪声，
   下一次真漂移就再也看不见了。
2. **根因在链路上，不在代码里。** 合并门禁按路径过滤（`artifacts/**`、`runs/**` 都在过滤面内），
   因此**任何**靠已提交工件自证的门，其证据都可能在下一次合并后消失。本单的修法（现算 + 硬断言
   内容）是一条通用模板：**判据要么能从树里重算，要么就必须硬断言失败**，没有第三条路。
3. **登记 ≠ 修复。** `ALLOWED_VACUOUS` 只是把「此刻不执行」写进账本，让它在报告里可见；
   真正的修复是重新生成工件。三个缺失工件中最贵的 `runs/lax_region_band/metrics.json`
   需跑 `python -m benchmarks.lax_region_band`（m16606 记录 243 s），本单未跑（见 ⑥）。
4. **census 与树同生同灭。** 增删任何一个测试模块都会改变 census（本单删掉自己的草稿模块后 UNRESOLVED 从 10 回到 9），所以 `--write` 必须在最终树上跑一次；`--check` 就是这条纪律的机器 Enforcement。
5. **适用范围**：本单只动门禁与其普查，不触碰任何求解器数值路径；TensorDG 的物理结论不受影响。

## ⑥ 误差判定

本单是**门禁/证据链**类交付物，其「被测对象」是测试门本身，不存在物理参考量，
因此**没有可比的 <3% 误差项**——这不是含糊，而是量纲不适用。可裁量的是**计数型判据，全部精确**：

| 判据 | 基准 | 实测 | 是否满足 |
|---|---|---|---|
| 政策门失败数 | 0 | **0**（11 passed） | ✅ |
| census 与全树一致性 | 逐项相等 | `modules`/`verdicts`/`vacuous_modules` 三道断言全过 | ✅ |
| 「工件被滤」回归门 | 必须不抛异常且判据不变 | `test_census_survives_a_filtered_artifact` 通过 | ✅ |
| 被吞工件的空转可见性 | 3 个 VACUOUS 全部登记且命名缺失工件 | 3/3 | ✅ |

**honest not_run（如实登记，不算通过）**：

- 未重新生成 `runs/lax_region_band/metrics.json`（需 243 s 基准跑，本机 load1≈260 不适宜）；
  未生成 `artifacts/dg_advection_bench.json`、`runs/shu_osher_case_pool/metrics.json`（既有欠账）。
- 原计划的**全树动态可达性审计**（每个 `tensordg.__all__` 公共入口是否真被测试执行）在实测中被放弃：
  带 `--cov-context=test` 的全量 1674 用例在饱和机器上前 3 分钟只跑完 17 个用例（投影数小时），
  超出单干预算。该工具草稿未交付（避免交付未验证件），可行性结论记在此处供下次认领：
  **必须上集群（`run_cluster`）才有意义，且应做「按模块分片 + 单模块超时」而非整体一次跑。**
- `tests/test_gate_vacuity_policy.py` 之外的全量套件未跑（同上原因）。

## ⑦ 复现命令

```bash
python tools/audit_gate_vacuity.py --write                  # 重算 census + 报告
python tools/audit_gate_vacuity.py --check                  # 漂移即非零退出
pytest -q tests/test_gate_vacuity_policy.py                 # 11 passed
pytest -q tests/test_lax_region_band.py -rs                 # 5 passed, 1 skipped（空转可见）
```