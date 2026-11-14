# audit/ — m21040 交差核验（对象：m21003 · TensorDG/TensorLBM）

| 路径 | 内容 |
|---|---|
| `refs/TensorLBM_m21003/` | 被审交付树的未改动副本（sha256 见 `artifacts/sha256_subject.txt`；原树 `/root/tp-agent/work/m21003`） |
| `probe_poiseuille_convergence.py` | 收敛探针：交付 BGK 配置续算到 16000 步（不改交付代码，只改步数） |
| `make_figures.py` | 剖面 + 收敛曲线图 |
| `insert_verdict.py` | 写入 `tensor_maturity_verdicts`（幂等，按 repo/scope/verdict_by 冲突更新） |
| `artifacts/poiseuille_convergence.json` | 探针原始数据（4 个 checkpoint × 2 种解析参考 × 速度剖面） |
| `artifacts/pytest_rerun.log` | 独立重跑单元测试日志（8 passed） |
| `artifacts/repro_failure.log` | 交付入口 `benchmarks/poiseuille.py` 的崩溃栈（exit=1，MRT dtype） |
| `artifacts/scheme_parity.txt` | BGK/MRT 在 float64 场下的解析解对照（方案真伪检验） |
| `artifacts/sha256_subject.txt` | 被审文件哈希（副本与原树一致） |
| `artifacts/verdict_evidence.json` | 落表 evidence 全文 |
| `artifacts/report_m21040.md` | 已发布报告正文（tensor/reports id 19060） |
| `artifacts/fig_poiseuille_convergence.png` | 报告图 |

结论：L2（独立功能实现）；L3 拒绝（零调用方）；L4 拒绝（无注册基准 + 工件缺失 + 入口崩溃 + 交付配置 11% > 3%）。
详见 `docs/AUDIT-m21003-XVAL-LBM-MASS-EXCHANGE-SELECTOR.md`。
