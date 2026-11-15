# XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM · TensorLBM 稀疏 list-based AOS LBM 复现与解析对标

- **锚点**：`XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM`（feature_ledger，验收：对标误差 <15%）
- **上游做法**：lssfau/lbmpy 的 `sparse-list-based-aos-lbm`
- **交付**：`TensorLBM/src/tensorlbm/sparse_aos.py`（实现）、`TensorLBM/tests/test_sparse_aos.py`（10 项回归）、
  `TensorLBM/benchmarks/sparse_aos_xval.py`（对标工装）、`TensorLBM/outputs/sparse_aos/xval_summary.json`（数字工件）、
  `TensorLBM/outputs/sparse_aos/fig_*.png`（4 图）
- **环境**：torch 2.7.1 / CPU / 2 线程 / float64
- **数字口径**：本文件所有数字由 `xval_summary.json` 程序化注入（`python3` 渲染脚本见提交记录），零手抄。

## 1. 做法（两要素）

1. **稀疏 list-based 域**：由掩膜得到流体单元列表 `index`（行主序稠密下标）与源表
   `src[q, c] = list_index(x_c - e_q)`；`src = -1` 表示上游为固体/域外，按半程反弹
   `f_new[c,q] = f_post[c, opp(q)]` 处理。周期轴先在稠密格上取模再查表；非周期轴越界即壁。
2. **AOS 布居布局**：`f.shape == (n_fluid, Q)`，方向轴单位步长；同一物理另有 SoA `(Q, n_fluid)`
   实现（`layout="soa"`）作为对照，两者共用同一套数学。

pull 型稀疏流动分支自由：每步只有 Q 次 source-indexed gather，无逐步格点索引运算。

## 2. 结果（数字直读 `xval_summary.json`）

**① 稀疏 vs 稠密等价性**（对 `stream_periodic + bgk_collision` 的稠密解，相对最大偏差）

| 网格 | 步数 | AOS | SoA |
|---|---|---|---|
| 16×16 | 80 | 2.12e-15 | 2.24e-15 |
| 32×32 | 80 | 3.00e-15 | 3.24e-15 |

**② Poiseuille 定常剖面**（τ=0.8，ν=0.1000，g=1e-5，x 周期，上下各 1 层壁）

| Ny | n_fluid | 步数 | 定常步 | rel-L2（半程壁面） | rel-L2（节点中心） | u_max/u_ref | 质量漂移 |
|---|---|---|---|---|---|---|---|
| 18 | 128 | 4600 | 4600 | 0.492% | 15.592% | 0.9964 | 3.1e-13 |
| 34 | 256 | 17800 | 17800 | 0.123% | 8.162% | 0.9991 | 1.1e-12 |
| 66 | 512 | 45000 | None | 0.033% | 4.180% | 0.9998 | 2.8e-12 |

**③ Taylor-Green 粘性衰减**（128×128，12000 步，ω=1.9）：数值 λ=4.230302e-05
vs 解析 νk²=4.227318e-05 → 相对误差 **0.071%**。

**④ 成本**（256×256，流体占比 0.444，n_fluid=29076，60 步平均）

| 项 | dense (Q,Ny,Nx) | sparse AOS (n,Q) | sparse SoA (Q,n) |
|---|---|---|---|
| 布居状态 | 4.719 MB | 2.093 MB | 同 AOS |
| 源表 int64 | — | 2.093 MB（int32 可减半） | 同 AOS |
| ms/step | 29.11 | 21.88 | 14.40 |

## 3. 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| Poiseuille rel-L2（最差分辨率） | 半程反弹解析解 | **0.492%** | 是 |
| Poiseuille rel-L2（节点中心口径，最差） | 节点中心解析解 | 15.592% | 否（口径差异，非格式误差） |
| Taylor-Green 衰减率 | νk² | 0.071% | 是 |
| 稀疏↔稠密等价性 | 稠密 BGK 解 | 3.0e-15（相对） | 是（舍入级） |

锚点验收（<15%）→ **通过**；实测最差 0.492%。

## 4. 踩坑与局限（诚实清单）

- **张量列序**：`D2Q9.e` 列是 `(e_x, e_y)`，而 `streaming.stream_periodic` 用 `roll(shifts=(e_y,e_x))`。
  首版源表把 `e[:,0]` 当 `e_y`，x/y 方向互换——被「源表逐项 vs 暴力稠密邻居枚举」测试当场抓住。
- **解析式口误**：Taylor-Green 该模态的衰减率是 **νk²**（涡量扩散：ω_z=2Ak sin sin e^{-νk²t}），
  不是常见的 2νk²；首版按 2νk² 对标得出 49.9% 误差，是**基准写错**而非求解器错，已由本文件的 0.07% 纠正。
- **AOS 在 PyTorch/CPU 上不占优**：SoA 每步 14.40 ms vs AOS 21.88 ms
  （SoA 快 0.66×）。AOS 的优势在代码生成/GPU 上把 Q 个方向放进寄存器向量，
  该优势**本单未测**，属未验证项；本单只证明 AOS 布局与物理正确。
- **稀疏收益有代价**：源表额外 Q·n·8 字节（int64）。本算例流体占比 0.44 时
  仅状态即 44.4% dense，加上源表后总占用约为 dense 的 89%。
- **未覆盖**：3D（D3Q19/27）、多相/MCMP 稀疏化、GPU/编译后端、开口/曲线边界、ny=66 在 45000 步内未达 1e-10 稳态判据
  （但剖面误差已 0.03%，说明判据过严而精度已饱和——如实记录）。

## 5. 复现

```
cd TensorLBM
python3 -m pytest tests/test_sparse_aos.py -q     # 10 passed
python3 benchmarks/sparse_aos_xval.py            # 生成 outputs/sparse_aos/xval_summary.json + 4 图
```
