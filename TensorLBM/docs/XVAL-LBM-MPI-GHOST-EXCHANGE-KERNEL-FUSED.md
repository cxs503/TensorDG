# XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED · 融合式（方向选择性）周期 ghost 交换对标

- **锚点**：`XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED`（feature_ledger，验收：对标误差 <15%）
- **上游做法**：`lssfau/lbmpy` 的 `mpi-ghost-exchange-kernel-fused`
  （`src/lbmpy/advanced_streaming/communication.py:20` `LBMPeriodicityHandling`、
  `:121` `get_communication_slices`、`:185` `periodic_pdf_gpu_copy_kernel`）
- **交付**：`src/tensorlbm/fused_periodicity.py`（实现）、
  `tests/test_fused_periodicity.py`（24 项回归，含 2 项故障注入）、
  `tests/test_fused_periodicity_lbmpy_conformance.py`（与上游 `comm_slices` 逐种群对账）、
  `benchmarks/fused_periodicity_xval.py`（对标工装）、
  `outputs/fused_periodicity/xval_summary.json`（数字工件）、`outputs/fused_periodicity/fig_*.png`（3 图）
- **环境**：3.11.6 / numpy 2.4.6 / loongarch64 / CPU / float64
- **数字口径**：本文件所有数字由 `xval_summary.json` 程序化注入（`scripts/render_fused_periodicity.py`），零手抄。

## 1. 上游到底做了什么（先读代码，再判定）

`LBMPeriodicityHandling` **不做域分解、不做异步通信**。它把周期 ghost 填充编译进通信/streaming
kernel：对每个通信方向 `comm_dir`，只拷贝 **ghost 面（厚度 1）** 上的 **单个布居**（
`_compile_copy_kernels` 里 `pdf_idx` 是常数），CPU 侧就是 `arr[dst] = arr[src]`，GPU 侧生成 copy kernel。
代码注释称生成的 copy kernel 比 cupy 数组拷贝「快近 2 倍」。

因此本单要验证的是两条，而不是直接采信注释：

1. **选择性是否真的是「逐方向 × 逐布居」**——把选择集合与上游 `comm_slices` 逐种群对账（Arm A）；
2. **融合是否引入正确性损失**——融合交换必须与整场拷贝逐比特一致，且物理仍对得上解析解（Arm B/D）。

## 2. 怎么做（方法与判据先落盘）

1. 交换计划**由 stencil 自动派生**（`wrap_plan`），不含任何按格点手写的表：
   对每个布居 q 的每个非零分量方向，把目标区域划分为「内部副本」与「跨缝副本」两类互不重叠的盒，
   跨缝盒就是需要从周期另一侧取值的 ghost 值。选择掩膜 `selection_matrix` 即该集合的布尔表示。
2. 三个可互换后端，结果必须逐比特相同、只有数据搬运不同：
   - `stream_fused`：内部与跨缝值在**同一遍**写出，直接读碰撞后的数组（对应上游的融合 copy）；
   - `stream_fullcopy`：逐布居 `np.roll` 整场拷贝（对应上游注释里的「数组拷贝」基线）；
   - `stream_ghost`：先 `np.pad(mode="wrap")` 造 ghost 数组再取内区（未融合的「先交换后流动」）。
3. 数值口径统一计数：整遍写入元素数、跨缝元素数、整场临时拷贝次数、ghost 缓冲个数。
4. 物理臂用两个**精确**解：2-D Taylor-Green（速度模态按 `2νk²` 衰减）与 3-D Kolmogorov 剪切波
   （`u=(A cos ky,0,0)`，对流项恒为零，精确解 `νk²`）。衰减率用最小二乘拟合 `ln(amplitude)-t` 斜率。
5. 上游对账：直接构造 lbmpy `LBMPeriodicityHandling` 对象读取其 `comm_slices`，
   把 ghost 坐标系换算到内部坐标，再按 pull 语义做 `+c mod N` 平移后与 `selection_matrix` 比较。

## 3. 结果

### Arm A：选择集合 vs 上游 `comm_slices`（全对账）

| 格点 | 域形状 | 上游拷贝条目 | 上游跨缝单元 | 本实现跨缝单元 | 不匹配布居数 |
|---|---|---|---|---|---|
| D2Q9 | [3, 4] | 16 | 38 | 38 | 0 |
| D2Q9 | [7, 5] | 16 | 68 | 68 | 0 |
| D3Q19 | [4, 5, 3] | 42 | 422 | 422 | 0 |
| D3Q27 | [3, 4, 2] | 98 | 368 | 368 | 0 |

`all_conform = True`。上游条目的最内层索引是**单布居**（D2Q9 16 条 / D3Q19 42 条 / D3Q27 98 条），
确认其选择性等价于本实现从 stencil 派生的跨缝集合。

### Arm B：三后端逐比特一致性

| 格点 | 域形状 | max&#124;fused−fullcopy&#124; | max&#124;fused−ghost&#124; | 逐比特相同 | 融合/整场元素比 |
|---|---|---|---|---|---|
| D2Q9 | [32, 40] | 0.0e+00 | 0.0e+00 | True | 0.50 |
| D3Q19 | [12, 14, 10] | 0.0e+00 | 0.0e+00 | True | 0.50 |
| D3Q27 | [10, 12, 8] | 0.0e+00 | 0.0e+00 | True | 0.50 |

`all_bitwise_equal = True`：融合与「先交换后流动」在数值上完全等价，差异只在搬运量。

### Arm C：成本（元素搬运与单步墙钟，单线程 NumPy）

| 格点 | N | fused ms | fullcopy ms | ghost ms | 对整场拷贝加速 | 对 ghost 数组加速 | 跨缝元素占比 | fused 跨缝字节 | fullcopy 写入字节 |
|---|---|---|---|---|---|---|---|---|---|
| D2Q9 | 32 | 0.04 | 0.27 | 0.63 | 6.12× | 14.20× | 4.1233% | 3040 | 147456 |
| D2Q9 | 64 | 0.06 | 0.31 | 0.69 | 5.06× | 11.37× | 2.0725% | 6112 | 589824 |
| D2Q9 | 128 | 0.11 | 0.43 | 0.83 | 3.86× | 7.43× | 1.0390% | 12256 | 2359296 |
| D2Q9 | 256 | 0.34 | 0.85 | 1.24 | 2.51× | 3.68× | 0.5202% | 24544 | 9437184 |
| D3Q19 | 16 | 0.15 | 0.68 | 1.86 | 4.60× | 12.52× | 9.6217% | 59904 | 1245184 |
| D3Q19 | 24 | 0.26 | 0.92 | 2.14 | 3.52× | 8.20× | 6.4693% | 135936 | 4202496 |
| D3Q19 | 32 | 0.44 | 1.34 | 2.63 | 3.05× | 5.98× | 4.8725% | 242688 | 9961472 |
| D3Q19 | 48 | 2.21 | 3.67 | 5.57 | 1.66× | 2.52× | 3.2621% | 548352 | 33619968 |
| D3Q27 | 16 | 0.26 | 1.05 | 2.50 | 4.05× | 9.66× | 11.9864% | 106048 | 1769472 |
| D3Q27 | 24 | 0.43 | 1.39 | 3.01 | 3.23× | 6.99× | 8.1040% | 241984 | 5971968 |
| D3Q27 | 32 | 0.78 | 1.96 | 3.93 | 2.52× | 5.06× | 6.1207% | 433216 | 14155776 |
| D3Q27 | 48 | 3.53 | 5.22 | 7.53 | 1.48× | 2.13× | 4.1091% | 981568 | 47775744 |

跨缝占比随 N 单调下降（表面积/体积），与「选择性收益随问题规模增大」的上游动机一致；
加速比在本机 NumPy 上为 1.48×–6.12×（含 Python 循环开销，见局限）。

### Arm D：物理对标（解析基准）

| 算例 | 网格 | 步数 | 实测衰减率 | 解析率 | 相对误差 |
|---|---|---|---|---|---|
| 2-D Taylor-Green | 48×48 | 936 | -1.714629e-03 | -1.713473e-03 | 0.0675% |
| 2-D Taylor-Green | 96×96 | 3783 | -4.284211e-04 | -4.283682e-04 | 0.0123% |
| 3-D D3Q19 shear wave | 24³ | 350 | -3.448704e-03 | -3.426946e-03 | 0.6349% |
| 3-D D3Q19 shear wave | 32³ | 682 | -1.934344e-03 | -1.927657e-03 | 0.3469% |

500 步质量漂移 4.29e-14（相对）。

## 4. 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 2-D Taylor-Green 衰减率（48²，936 步） | `2νk²` | 0.0675% | **是** |
| 2-D Taylor-Green 衰减率（96²，3783 步） | `2νk²` | 0.0123% | **是** |
| 3-D D3Q19 剪切波衰减率（24³） | `νk²` | 0.6349% | **是** |
| 3-D D3Q19 剪切波衰减率（32³） | `νk²` | 0.3469% | **是** |
| 融合 vs 整场拷贝 | 逐比特 | 0（bitwise） | **是**（精确） |
| 选择集合 vs 上游 `comm_slices` | 上游代码 | 0 个不匹配布居 | **是**（精确） |
| 质量守恒（500 步） | 初值 | 4.3e-14 | **是** |

锚点验收（<15%）→ **通过**；实测最差 0.6349%（3-D 剪切波 24³）。
物理臂的误差极限由 BGK 压缩性误差与拟合窗口决定，随网格加密下降（96² 比 48² 低 5.5×）。

## 5. 注意点与局限（诚实清单）

- **上游是「周期 ghost」不是 MPI 域分解**：`LBMPeriodicityHandling` 明确只支持
  `SerialDataHandling`，代码里 `raise ValueError("Only serial data handling is supported!")`。
  本单因此**不主张域分解正确性**；跨进程/跨 rank 的 ghost 交换属于另一个锚点
  （`XVAL-LBM-R2-SELECTIVE-FUSED-GHOST-EXCHANGE-PLUS-3`）。
- **无 GPU/cupy**：容器内无 CUDA、无 cupy，上游「生成 copy kernel 比 cupy 数组拷贝快近 2 倍」的
  原始声明**未复现**；本单给的是 NumPy CPU 上的同构对照（fused vs 整场 `np.roll` vs ghost 数组），
  数值与搬运量可信，倍数不可外推到 GPU。
- **加速比含 Python 开销**：fused 每步有 Q 次切片写入的 Python 循环，规模越大越被内存带宽摊薄
  （D3Q27 N=48 只剩 1.48×）。若要有上游那种量级收益，需要把计划编译成 kernel（本单未做）。
- **故障注入是刻意的**：`test_fault_injection_*` 会删掉一个跨缝盒、交换两个布居的计划，
  并断言比对必须失败——保证「三后端一致」这条测试有牙齿。
- **两个「跨缝」定义等价但坐标不同**：上游把内部值写进 ghost（push 帧），本实现直接从对侧读取（pull 帧），
  两者差一个 `+c mod N` 平移；测试按该关系断言，而非直接集合相等（否则会误判为不一致）。

## 6. 复现

```
cd TensorLBM
python3 -m pytest tests/test_fused_periodicity.py -q                        # 24 passed
python3 -m pytest tests/test_fused_periodicity_lbmpy_conformance.py -q      # 上游对账（需 lbmpy）
PYTHONPATH=src python3 benchmarks/fused_periodicity_xval.py                # 生成 outputs/fused_periodicity/
```
