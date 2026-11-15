# GAP-CAD-T1-SOURCE-HUNT — 六标模几何源头追查（DTC / ONRT / JBC / KCS）

锚点：`GAP-CAD-T1-SOURCE-HUNT`。产物：`artifacts/baseline_fleet_source_hunt/`
（manifest.json / geometry.sha256.json / pages.sha256.json / bbox_measurements.json /
simman2020_ship_data_summary.json / fleet_specs_patch.json / figures/）。
工具：`scripts/hull_source_hunt.py`（可重跑，`--offline` 只读缓存）。

## ① 案例介绍（追查对象与来源）

| 船模 | 用途 | 权威源（页面 sha256 见 pages.sha256.json） | 几何包 |
|---|---|---|---|
| DTC（Duisburg Test Case，355 m 级 post-Panamax 集装箱船） | 总账缺额，本次新增 | https://www.uni-due.de/inam/（DTC 原始作者单位：University of Duisburg-Essen, INAM，原 IST） | `igs_ordner.zip`（5 个 IGES：hull / bilge keel / propeller / propeller-hull connection / rudder-skeg） |
| ONRT（ONR Tumblehome，154 m） | 复算已合并条目 + 解决排水量争议 | https://t2015.nmri.go.jp/onrt_gc.html | `onrt_w_gap.zip`, `onrt_wo_gap.zip`, `propeller_IIHR2.zip` |
| JBC（Japan Bulk Carrier，280 m） | 补 scale 因子（发布页只印实船列） | https://t2015.nmri.go.jp/jbc_gc.html | `JBC_IGES.zip`, `MP687_particulars.zip`, `jbc-rudder.igs` |
| KCS（KRISO 集装箱船，230 m） | 方法对照：发布几何即实船尺度 | https://t2015.nmri.go.jp/kcs_gc.html | `KCS_hull_SVA.zip`, `KCS_hull_Case2-11.zip`, `KCS_Rudder.zip`, `KP505.zip` |

**关键发现（源头）**：`www.duisburg-test-case.com` 在本沙箱内 DNS 直接 NXDOMAIN，
`web.archive.org` 不可达，通用搜索引擎返回被污染的结果——但 DTC 几何仍有活源：
DTC 原始作者单位 INAM 官网仍在 `/imperia/md/content/inam/igs_ordner.zip` 挂出
"Duisburg Test Case (DTC)" 链接。这是本次追查的实质增量（此前总账记为"源头不可得"）。

## ② 怎么做的（方法 / 工具链）

1. `scripts/hull_source_hunt.py` 抓取并缓存源页面（HTML）与几何包（ZIP/IGS），逐字节 sha256；
   页面与几何均落 `$HULL_SOURCE_CACHE`（默认 `~/.cache/hull_sources`），**几何字节不进仓**
   （发布方未声明再分发许可）。
2. 主尺度不再手抄：页面 `<table>` → 单元格文本 → `parse_number()`（兼容发布方千分位空格
   `7.27 86` 与单位后缀 `8507 ton`）；T2015 页面"量名 + 单位"双列布局用
   `t2015_layout()` 定位 **Full scale** 列的 `+1` 数据列。
3. 几何侧：IGES 参数段（P 节 1–64 列）逐记录扫描，读取 **CARTESIAN POINT(116)** 与
   **NURBS 曲面(128)/曲线(126)** 的控制点，得点集包围盒；单位取 Global 段 `nHM/ nHMM`
   Hollerith 串（本次文件为 `1HM`=m 与 `1HMM`=mm 两种混用，必须按文件各自判读）。
4. 尺度推定：`λ = 发布实船长度 / 实测 IGES 点集跨度`，并用 **KCS 对照**（发布即实船尺度，
   λ 应为 1.0）验证方法本身。
5. 一致性检查：`Cb = ∇/(L·B·T)` 用页面自身数字复算，与页面 Cb 比较；
   ONRT 排水量争议用"隐含密度"与"模型质量 × λ³"两条独立路径交叉。

## ③ 计算结果

### 3.1 主尺度与体积（全部由源页面程序化解析，未手抄）

| 量 | ONRT | JBC | KCS |
|---|---|---|---|
| Lpp / Lwl (m) | — / 154 | 280 / 285 | 230 / 232.5 |
| Bwl (m) | 18.78 | 45 | 32.2 |
| T (m) | 5.494 | 16.5 | 10.8 |
| ∇ (m³) | 8500.7794（= Cb·L·B·T） | 178369.9 | 52030 |
| 页面 Cb | 0.535 | 0.858 | 0.6505 |
| 复算 Cb | 0.535 | 0.85796 | 0.650499 |

### 3.2 Cb 一致性（页面数字自洽性）

| 检查 | 复算 Cb | 页面 Cb | 相对误差 | 判定 |
|---|---|---|---|---|
| JBC | 0.85796 | 0.858 | 4.653e-05 | PASS |
| KCS | 0.650499 | 0.6505 | 2.160e-06 | PASS |

### 3.3 ONRT 排水量争议（总账 `published_disputed: displacement_volume`）

- 页面实船排水量 8507 ton；Cb·Lwl·Bwl·T = 8500.7794 m³；
- **隐含密度 1000.73 kg/m³**（与淡水 1000 kg/m³ 相差
  0.073%）；
- 结论：发布方的 "ton" 是 **公吨（1000 kg/m³）**，不是海水 1025 kg/m³；争议因此消解为
  "单位口径问题"，`8507 ton ↔ 8500.8 m³` 自洽。模型质量 ×λ³ 的独立复核本次未能完成
  （ONRT IGES 未解析出控制点，λ 缺失，见 ⑤）。

### 3.4 尺度因子（数据驱动，非手抄）

| 船 | 实测点集跨度 (m) | 发布实船长 (m) | λ = L/跨度 | 几何成员 |
|---|---|---|---|---|
| ONRT | — | — | — | 未解析出控制点 |
| JBC | — | — | — | 未解析出控制点 |
| KCS | 243.844 | 230 | 0.943225 | KCS_hull_SVA/KCS_hull_SVA.igs |

KCS 对照：λ = 0.943225，相对 1.0 偏差
0.05677。

### 3.5 几何归档指纹（节选，完整见 geometry.sha256.json）

| 归档 | bytes | sha256 |
|---|---|---|
| `igs_ordner.zip` | 2343841 | `7b18e40d783b670f226a55a7d3f0ccf13c0978e3d80a13a78eced65a7a852541` |
| `onrt_w_gap.zip` | 759503 | `6134d9c76ed2ac232f7d8eb82ad0aabce65ef45fe9798255584f34e0a88073a3` |
| `onrt_wo_gap.zip` | 752918 | `bd747eff8f531ba4c698c4ad87fa52e7a8fa631594e49eb3eccbd2ec5c695685` |
| `propeller_IIHR2.zip` | 934348 | `b4c846e02f7b37aa793ad94ded8776b3fe1dab5cc353586d573ad6c10295cc13` |
| `JBC_IGES.zip` | 2508534 | `e19800ee954f4960ef0ccb5bd764b45774b3200cb5c783fdaf17f6b1661f3f9c` |
| `MP687_particulars.zip` | 248313 | `feb298a22caef7efb9bfb17b55409f70c97547e0d57ea0ca55d132a61ae4342e` |
| `KCS_Rudder.zip` | 215590 | `6431c20ca2be5ca6abb05c97e8a0579b172e58016524ffacc74b3d3a7edf9d79` |
| `KCS_hull_Case2-11.zip` | 558122 | `8cb1e08e4400e307d0f87631f99ea74fca23903a370974d3a458323957a49a3c` |
| `KCS_hull_SVA.zip` | 74086 | `b9e334853857b819bbddfd549f4599dde7cba57a9540893b873aaae56dc99a20` |
| `KCS_prop_MOERI.zip` | 42568 | `54503eb99e946f12c14e47d71af48005665e027026952aab1db37510b3cf26f0` |
| `KP505.zip` | 2948295 | `cb3bc964bf8dd483f96d3268eb599d5f77525b2c03de83d9bffe03eec9fff152` |

DTC 归档（新增，源头=INAM）：`igs_ordner.zip` bytes=2343841，
sha256=`7b18e40d783b670f226a55a7d3f0ccf13c0978e3d80a13a78eced65a7a852541`，内含 5 个 IGES（hull/bilge keel/propeller/propeller-hull connection/rudder-skeg）。

### 3.6 SIMMAN 2020 ship_data_summary.php

页面已抓取并缓存，sha256=`c1a668f6d2440c5d83a83c3616285e9288ef618dbc9a2feebd26627be3f4ebb3`，
程序化解析出 2 条船（KCS, ONRT），明细见
`artifacts/baseline_fleet_source_hunt/simman2020_ship_data_summary.json`（逐表原样保存，
可复核到单元格级）。

## ④ 结果图

![geometry scale](figures/figure1_geometry_scale_inference.png)
*图 1 由 IGES 点集跨度反推的尺度因子，KCS 为方法对照（发布即实船尺度）。*

![Cb cross-check](figures/figure2_cb_consistency.png)
*图 2 用页面自身数字复算 Cb 与页面 Cb 的对比（两条独立路径同源同页）。*

![ONRT dispute](figures/figure3_onrt_displacement_dispute.png)
*图 3 ONRT 排水量争议：Cb 折算体积与公吨口径的一致性。*

（图由 `scripts/hull_source_hunt.py` 内 matplotlib 程序化生成，无手填数字。）

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **DTC 官网已死但源头未死**：`duisburg-test-case.com` NXDOMAIN、`web.archive.org` 不可达；
   活源是 DTC 原作者单位 INAM 的 `/imperia/md/content/inam/igs_ordner.zip`（本次取得）。
   该页未给主尺度表 → **DTC 的 Lpp/B/T/∇/Cb 本次 WITHHELD，不做任何断言**（宁缺勿造）。
2. **IGES 写法不统一**：T2015/DTC 的 IGES 是 free-format（`128,9,9,3,3,...;`，
   实体类型后直接跟参数），不是规范的 1–8 列右对齐；P 节 65–72 列是 DE 指针，
   误当数据处理会使"实体类型"错位（本项目首版即因此解析出 0 个点）。
3. **控制点包围盒 ≠ 曲面外形**：NURBS 控制网包住曲面，KCS 实测 x 跨度
   243.844 m vs Lpp 230 m（偏大
   6.02%）。
   因此 λ 由该口径反推**只能作为外边界估计**，误差带约 ±5%，不足以支撑 3% 判据。
4. **ONRT / JBC 主船体 IGES 未解析出控制点**（0 点）：这两个包的主船体以 trimmed
   surface（144/143）或其它实体组织，需补 144/143→表面、再取 surface 的控制点的解析链；
   本次未完成 → 对应 λ **WITHHELD**，不冒充已得。
5. **单位混用**：同一发布方不同文件分别为 `1HM`（m）与 `1HMM`（mm），必须逐文件读 Global 段，
   否则 λ 会差 1000 倍。
6. 几何字节未入仓（许可未声明），只入 sha256/bytes/URL；复现需重跑脚本联网取回。

## ⑥ 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| JBC Cb 复算 | 页面 Cb 0.8580 | 4.65e-05 | **是** |
| KCS Cb 复算 | 页面 Cb 0.6505 | 2.16e-06 | **是** |
| ONRT 隐含密度 | 淡水 1000 kg/m³ | 0.073% | **是** |
| KCS 尺度因子（方法对照） | λ=1.0 | 0.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.056770.05677% | **否（≈6%，超 3% 判据）** |
| JBC / ONRT 尺度因子 | 无发布基准（发布页只印实船） | 未能测得 | **不适用（阈值无法评估）** —— honest FAIL，见 ⑤.4 |
| DTC 主尺度 / Cb | 无可达一手基准 | 未测 | **不适用** —— WITHHELD |

**净结论**：几何源头追查的"取得字节 + 指纹入库"完成（DTC 新增 5 个 IGES、ONRT/JBC/KCS 全归档）；
主尺度自洽性检查全部 <3%；但**几何反推尺度因子未达 3% 判据（KCS 对照 +6%），
故 JBC/ONRT 的 scale 因子本次记 WITHHELD，不覆盖总账既有值**。
