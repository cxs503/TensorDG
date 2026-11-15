# CASE-DG-EULER-VORTEX-ERR round 16：崩溃隔离与证据入库（C11 未验证 5→3）

> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜算例池·等熵涡移动误差守恒（ref: Yee, Sandham & Djomehri, JCP 162, 2000）｜容差 6%（t=1 快照，最大场相对 L2）
> mission 22520｜TensorDG `agent/m22520`
> 本轮裁决（写作时）：**INCOMPLETE**（C1–C10 全 passed；C11 由 5 条 `unverified` 降为 3 条，3 个重放作业在飞）

## ① 本轮做了什么（两个实质增量）

### 增量 A：把已经算出来的证据真正入库（5 → 3）

round 15 递交的成本均衡分片 `cj4121`–`cj4124` 里，`cj4124`（分片 4/4）**确实**产出了两条证书：
`dg_d2_n32_t10_cfl0.4`、`dg_d2_n32_t5_cfl0.2`，`replay_delta` 均为 `0.000e+00`（位相同复现）。
但这两条证据从未被搬进本地 bundle——「算完了」和「入库了」是两件事。本轮用严格搬运器入库：

```
PYTHONPATH=. python3 tools/land_provenance_certificates.py \
    /root/tp-agent/cluster/fetch/cj4124_n315.tar.gz
# landed 2, rejected 0, kept 14; certificates on disk 16
```

搬运器逐条核对：检查点必须本地存在（拒孤儿）、配置必须与本地记录在物理字段上一致
（task/degree/cells/cfl/final_time/strength，拒「不同网格的证书」）、必须指名当前聚合的参考模型、
`replay_delta` 必须有限且 ≤1e-9、重放百分比要与存储百分比同容差一致、已存在的证书只能被更好的一条替换。
再用聚合器重算判据：C11 的 `unverified` 由 5 条降为 3 条，`certificates_on_disk` 14→16，
其余 C1–C10 读数不变。

### 增量 B：重放崩溃的隔离（缺陷修复）

`cj4121`/`cj4122`/`cj4123` 三个分片的状态是 `succeeded`、`rc=0`，但它们的日志尾部都是

```
bash: line 1: 2578062 Segmentation fault (core dumped) \
  PYTHONPATH=. ... python3 tools/provenance_certify.py --remaining --shard 1/4 --threads 8 ...
```

也就是说：**作业包装层报成功，实际重放进程段错误**，分片里所有还没写盘的证书全部丢失。
`tools/provenance_certify.py` 原来是「一个进程顺序重放本分片的全部检查点，每算完一条写一次盘」，
所以一次崩溃会带走该分片剩余的全部工作量——这正是 3 条 `unverified` 存活的机制。

本轮加 `--isolate`：每条检查点在自己的子进程里重放（`python -m tools.provenance_certify --keys K`），
父进程只负责派发与记账；子进程被信号杀死（`returncode < 0`）或非零退出时，
该 key 记为 `crashed (signal N)` / `failed (rc N)` 并**继续尝试后面的 key**，不会静默当作已认证。
父进程结束时从磁盘重读 bundle 报告进度（绝不使用派发前读到的陈旧副本）。

## ② 怎么验的

| 测试 | 断言什么 | 结果 |
|---|---|---|
| `tests/test_provenance_certify_isolation.py::test_isolated_status_survives_a_segfaulting_child` | 真的让子进程 `os.kill(SIGSEGV)`，父进程必须继续并把它记为 `crashed (signal 11)` | 通过 |
| `...::test_isolated_status_reports_nonzero_exit_as_failure` | 非零退出记为 `failed (rc 3)`，不记为成功 | 通过 |
| `...::test_key_command_targets_exactly_one_checkpoint` | 子进程命令行只含一个 `--keys` | 通过 |
| `...::test_isolate_batch_writes_nothing_itself_and_reloads_from_disk` | 一条成功+一条崩溃时，只有成功的那条进入 bundle；崩溃那条**不入库** | 通过 |
| `tests/test_provenance_certify_shard.py`（原 9 条） | 分片仍是「好集」划分、不重叠不丢点（回归） | 通过 |
| `tests/test_case_euler_vortex_errconserve.py` | C1–C10 与 C11 判据 | 仅 `test_c11_all_legacy_records_have_certificates` 失败，如实反映 3 条未验证 |

## ③ 在飞作业（本轮递交，单 key 作业，互相不牵连）

| 作业 | 检查点 | steps | 线程 | 说明 |
|---|---|---|---|---|
| `cj4184` | `dg_d2_n64_t2_cfl0.2` | 1874 | 1（`OMP_NUM_THREADS=1`，`PYTHONFAULTHANDLER=1`） | 探「多线程/嵌套 OMP 是否崩溃诱因」，崩溃时 faulthandler 打印 Python 栈 |
| `cj4185` | `dg_d2_n32_t10_cfl0.2` | 4684 | 1 | 同上，成本最低 |
| `cj4186` | `dg_d2_n64_t10_cfl0.2` | 9369 | 4 | 最贵的一条，先排队 |

三条各自独立成败，任何一条崩溃只损失它自己；取回后再过严格搬运器入库、重聚合，C11 才有机会 `passed`。

## ④ 注意点 / 局限

1. **递交号 ≠ 证据**（round 15 事故的延续）：`cj4121`–`cj4123` 状态 `succeeded` 而证据为零；
   本轮把「只在写盘成功后才有证据」做成代码语义（父进程重读、崩溃记 named 状态），而不是靠人记。
2. **段错误根因仍未定**：目前只知道它发生在较大检查点的 `replay()` 内部（DG 路径走 torch float64 CPU），
   P0/numpy 路径与较小的 DG 检查点（2342 steps）从未崩过；线程数是被怀疑但未证实的变量。
   本轮不宣称修好了崩溃，只修好了「崩溃不该抹掉同批其余证据」。
3. **C11 未闭合**：3 条 `unverified` 仍然存在，裁决仍是 INCOMPLETE，不得写成通过。

## ⑤ 复现

```bash
PYTHONPATH=. python3 tools/land_provenance_certificates.py <artifact.tar.gz>   # 严格入库
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate \
    --output-dir case_euler_vortex_errconserve                                  # 重算判据
PYTHONPATH=. python3 tools/provenance_certify.py --remaining --isolate --threads 4
PYTHONPATH=. python3 -m pytest tests/test_provenance_certify_isolation.py -q
```