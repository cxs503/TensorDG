# m18125 — CASE-SHU-OSHER-DG 复现链缺陷定位与修复（重开 #377「管到底」）

## 结论先说

#377 的**数值件**已在前序合并中进入 `main`（`tensordg/cases/shock_entropy.py`、
`tensordg/euler_dg_1d.py::verify_shock_entropy`、`benchmarks/shu_osher_{1d,case_pool}.py`、
`validation/repro_shu_osher/`、`tests/test_shock_entropy_case_api.py`；该测试在本树 7 passed / 100 s）。
因此本轮不重复造件，而是修掉「交付件在**干净检出**上跑不起来」的三个结构性缺陷——即 #377 的
「功能+测试齐备」在交付链意义上并未真正完成。

| # | 缺陷 | 证据（修复前实测） | 修复 |
|---|---|---|---|
| 1 | 三个复现脚本把运行目录写死成 `runs/shu_osher_case_pool`，该目录被 `.gitignore` 忽略、永不进 `main` | `python validation/repro_shu_osher/smoke_repro.py` → `FileNotFoundError: runs/shu_osher_case_pool/metrics.json`；`make_evidence.py`、`platform_bench.py` 同 | 新增 `validation/repro_shu_osher/bundle.py::resolve_run_dir()`：搜索序 `$TENSORDG_RUN_DIR` → `runs/`（刷新研究优先）→ `artifacts/`（被跟踪副本）；三个脚本改吃它，输出 JSON 记录 `artifact_source` |
| 2 | `artifacts.sha256.json` 用裸文件名登记 `docs/figures/` 下 3 张 PNG，验证方在 bundle 目录里找不到（m15550 独立审计只能命中 16/19） | `bundle.resolve_manifest_entry("shu_osher_case_pool_cfl.png")` 旧键在 `artifacts/` 下不存在 | manifest 键改仓内相对路径（19 条全部可解）；`make_evidence.py` 只对图取哈希、不再把 PNG 复制进 `artifacts/`；新增 `bundle.py --verify` 一键复核（漂移即非零退出） |
| 3 | 上述链路无任何回归测试，缺陷可静默复发 | `grep -rn "runs/" tests/` 无覆盖 | 新增 `tests/test_repro_shu_osher_bundle.py`（12 例） |

## 修复后的验证（可复跑）

```
python validation/repro_shu_osher/bundle.py --verify        # 19/19 sha256 命中
python validation/repro_shu_osher/smoke_repro.py --dry-run  # 解析结论：source=curated
python validation/repro_shu_osher/smoke_repro.py            # 20.2 s 单线程，rel L1=2.461510% vs 表值 2.461510%（漂移 0.00e+00 pp）
python validation/repro_shu_osher/make_evidence.py          # 由被跟踪副本重生成：9 配置全 PASS，与仓内版本数值逐字节等价
pytest -q tests/test_repro_shu_osher_bundle.py \
         tests/test_shock_entropy_case.py \
         tests/test_shu_osher_case_pool.py                  # 25 passed / 10.8 s
```

数值本身未变：9 条 DG 腿相对 L1 0.5561%–2.4615%（全部 <5%，且全部 <3%），参考解自收敛 0.101%，
激波后平台密度 vs Rankine–Hugoniot 左态 2.4e-6%，全局质量闭合到 0.0e+00 % of M0，
未扰动段正弦 rel L1 0.0347%，透射激波位置 vs RH 0.714%。外部文献表值仍 **WITHHELD**
（论文只给图；执行环境无 PDF 可达）。

## 已知与本轮无关的既有失败

`tests/test_xval_oss_shu_osher.py::test_oss_leg_is_independent_and_converges` 在本树与
**HEAD 原样树**（`git archive HEAD` 解包）上表现一致地失败：沙箱内 `clawpack` 安装残缺
（`ImportError: cannot import name 'acoustics_1D' from ... clawpack.riemann`）。非本轮引入。

## 边界

- 只改 `validation/repro_shu_osher/` 内的交付/复现工具与新增测试；**未触碰** TensorDG 核心求解器模块
  （`tensordg/euler_dg_1d.py`、`tensordg/cases/*`），因此与并行 PR 的热文件不重叠。
- 新增二进制为零：图表仍只存 `docs/figures/`（原已跟踪），manifest 记录其哈希。
