# TensorDG 物性数据库 v1（FEAT-ASSET-PROPS-DB）

> 一次把「水/空气（黏度-温度-密度）+ 冰力学 + 泥沙 + 结构材料」做成**带出处、带适用域、SI 基单位**的整库，并提供查询 API / CLI，
> 供求解器与船舶平台按引用取物性，替代散落在各仓的魔法数字。

- 包路径：`tensordg/props/`（模块 `python -m tensordg.props` 即 CLI）
- 数据：`tensordg/props/data/{water,air,ice,sediment,structural}.json`
- 测试：`tests/test_props_db.py`（31 项，含文献锚点复核）
- 求解器接入样例：`examples/props_solver_integration.py`
- 船舶平台存量盘点：`docs/props-migration-map.md`（脚本 `scripts/props_migration_inventory.py`）

## 1. 统一 schema（验收①）

一条记录 = `物质 × 条件(T,P) × 物性 × 出处 × 适用域`：

```json
"density": {
  "dimension": "density", "unit": "kg/m^3",
  "mode": "table", "interp": "linear",
  "x": {"name": "T", "unit": "degC"},
  "domain": {"T": [0.0, 30.0], "P": [101325.0, 101325.0]},
  "points": [[0.0, 1028.1], ..., [30.0, 1021.7]],
  "source": {"ref": "ITTC 7.5-02-01-03 (2011) Table 3", "type": "table", "note": "S=35 g/kg"},
  "accuracy": "table values", "applicability": "0-30 degC, 1 atm, S=35 g/kg"
}
```

四种求值模式，全部强制带 `source` 与 `domain`：

| mode | 含义 | 例 |
|---|---|---|
| `table` | 表值 + 分段线性插值（节点值精确复现） | ITTC 水密度/运动黏度 |
| `correlation` | 解析相关式（参数显式存储） | Sutherland 空气黏度、理想气体密度/声速 |
| `derived` | 由同物质其它物性按恒等式导出 | `mu = rho * nu`、`nu = mu / rho` |
| `constant` | 规范/手册单值 + 文献区间写在 `note` | 冰强度、钢材 E/fy、Manning n |

`schema.py` 的校验项（`validate` 子命令与测试均跑）：`source.ref` 非空、`source.type` 合法、
单位量纲与 `dimension` 一致、表轴严格递增、表点落在声明域内、`correlation` 模型名可解析、
`derived` 表达式引用的物性存在。**查询结果里不可能拿不到出处**——`QueryResult.as_dict()`
的 `provenance` 永远非空。

## 2. 单位约定（验收③）

SI 基单位是唯一权威（`units.py` 的 `SI_BASE`）：

| 维度 | SI 基 | 允许的输出单位 |
|---|---|---|
| density | `kg/m^3` | `g/cm^3`, `kg/L`, `t/m^3` |
| viscosity_kinematic | `m^2/s` | `cSt`, `mm^2/s` |
| viscosity_dynamic | `Pa*s` | `cP`, `mPa*s`, `P` |
| stress | `Pa` | `kPa`, `MPa`, `GPa`, `N/mm^2` |
| length / velocity / angle | `m` / `m/s` / `rad` | `mm`, `cm`, `deg` |
| 温度轴 | 内部统一 °C | 输入可用 `degC`/`K`/`degF` |

跨维度请求直接报错（**不会**把 GPa 当 cSt 返回）：`db.query("water","nu",T=20,unit="GPa")` → `UnitError`。

## 3. 覆盖（验收②）

| 物质 | 类 | 状态 | 行数 | 物性 |
|---|---|---|---|---|
| `air` | gas | gas | 5 | `density`, `specific_heat_capacity`, `speed_of_sound`, `viscosity_dynamic`, `viscosity_kinematic` |
| `fresh_ice` | ice | solid | 6 | `compressive_strength`, `density`, `flexural_strength`, `poisson_ratio`, `tensile_strength`, `youngs_modulus` |
| `sea_ice` | ice | solid | 7 | `compressive_strength`, `density`, `flexural_strength`, `ice_steel_friction`, `poisson_ratio`, `tensile_strength`, `youngs_modulus` |
| `gravel` | sediment | granular | 9 | `cohesion`, `critical_shields`, `dry_bulk_density`, `friction_angle`, `grain_density`, `manning_n`, `median_grain_size`, `porosity`, `settling_velocity` |
| `medium_sand` | sediment | granular | 9 | `cohesion`, `critical_shields`, `dry_bulk_density`, `friction_angle`, `grain_density`, `manning_n`, `median_grain_size`, `porosity`, `settling_velocity` |
| `silt` | sediment | granular | 9 | `cohesion`, `critical_shields`, `dry_bulk_density`, `friction_angle`, `grain_density`, `manning_n`, `median_grain_size`, `porosity`, `settling_velocity` |
| `soft_clay` | sediment | cohesive | 8 | `cohesion`, `dry_bulk_density`, `friction_angle`, `grain_density`, `manning_n`, `median_grain_size`, `porosity`, `settling_velocity` |
| `AH36` | structural | solid | 6 | `density`, `poisson_ratio`, `thermal_expansion`, `ultimate_strength`, `yield_strength`, `youngs_modulus` |
| `EH40` | structural | solid | 6 | `density`, `poisson_ratio`, `thermal_expansion`, `ultimate_strength`, `yield_strength`, `youngs_modulus` |
| `alu_5083_H116` | structural | solid | 6 | `density`, `poisson_ratio`, `thermal_expansion`, `ultimate_strength`, `yield_strength`, `youngs_modulus` |
| `concrete_C40` | structural | solid | 6 | `compressive_strength`, `density`, `poisson_ratio`, `tensile_strength`, `thermal_expansion`, `youngs_modulus` |
| `frp_glass_polyester` | structural | solid | 6 | `compressive_strength`, `density`, `poisson_ratio`, `tensile_strength`, `thermal_expansion`, `youngs_modulus` |
| `stainless_316L` | structural | solid | 6 | `density`, `poisson_ratio`, `thermal_expansion`, `ultimate_strength`, `yield_strength`, `youngs_modulus` |
| `titanium_grade5` | structural | solid | 6 | `density`, `poisson_ratio`, `thermal_expansion`, `ultimate_strength`, `yield_strength`, `youngs_modulus` |
| `fresh_water` | water | liquid | 15 | `density`, `viscosity_dynamic`, `viscosity_kinematic` |
| `sea_water` | water | liquid | 15 | `density`, `viscosity_dynamic`, `viscosity_kinematic` |
| **合计** | 5 类 | | **125** | |

## 4. 查询 API 与 CLI（验收③）

```python
from tensordg.props import load_db
db = load_db()

res = db.query("sea_water", "nu", T=15.0, unit="cSt")
res.value        # 1.1883          (cSt)
res.value_si     # 1.1883e-06      (m^2/s, SI 基)
res.source["ref"]                # ITTC 7.5-02-01-03 (2011) Table 3
res.as_dict()                    # 可直接嵌入平台 API 响应（JSON 可序列化）

db.get("air", "mu", T=20.0)      # 只取 SI 值：Sutherland
db.query("water", "nu", T=60.0)  # OutOfDomain：60 °C 超出 ITTC 表域，拒绝外推
db.query("water", "nu", T=60.0, clamp=True)   # 显式钳到边界，clamped=True
db.query("AH36", "E", T=20.0, unit="GPa")     # 206 GPa
```

CLI：

```bash
python -m tensordg.props list                     # 物质/类/行数一览
python -m tensordg.props info sea_water           # 每条物性的出处、域、精度
python -m tensordg.props query water nu --T 15 --unit cSt --json
python -m tensordg.props provenance sea_water density
python -m tensordg.props validate                 # schema+出处体检，退出码 0/1
```

设计约束（刻意的）：**默认拒绝外推**。`T` 超域给 `OutOfDomain` 并打印出处，
而不是悄悄线性延伸——这是物性库最容易产生「看起来合理的错值」的地方。

别名：`water`→`fresh_water`、`seawater`→`sea_water`、`ice`→`sea_ice`、`rho`/`nu`/`mu`/`E`/`fy`/`c`。

## 5. 求解器接入样例（验收⑤，部分完成）

`examples/props_solver_integration.py` 给出 4 个真实消费路径（可直接跑）：

1. **阻力/尺度效应**：海水 ν(15 °C) → Re_FN → ITTC-1957 摩擦系数，模型(1:40) 与实船两点；
2. **可压 DG 初场**：空气 ρ/μ/a → `(ρ, u=M a, p, E)` 守恒量初值 + 显式 DG 的 CFL 步长；
3. **浅水源项**：泥沙 Manning n → 宽渠流速与等效二次阻力系数 `C_d = g n² / h^{1/3}`；
4. **结构/冰区筛分**：`E, ν` → DNV-RP-C201 板屈曲应力 `σ_cr = π²E/(12(1-ν²))(t/b)²`；
   海冰 `σ_f, E, ν, ρ_i` + 海水 ρ → ISO 19906 特征长度 `L_c` 与弯曲破坏载荷量级。

样例故意保留了一次**真实的域拒绝**：用 `T=-2 °C` 查海水密度会抛 `OutOfDomain`
（v1 表从 0 °C 起），样例改为 0 °C 并在注释里点出这是 gap——不外推、不粉饰。

**缺口（诚实）**：验收⑤要求「solver 接入样例 ≥2 仓」。本单只完成 TensorDG 一仓的
可执行样例；ship 仓侧只做到**盘点 + 迁移映射表**（`docs/props-migration-map.md`），
因为在 ship 仓工作树内落代码不属于本单 scope。第二仓接入需另派单（见 §7）。

## 6. 精度判定（对照锚点，逐项）

基准不是「和库自己比」，而是和文献/规范独立值比（`tests/test_props_db.py` 断言，
数值由 pytest 现场计算，未手抄）：

| 关键量 | 库值实现 | 独立基准 | 容差 | 结果 |
|---|---|---|---|---|
| 淡水 ρ(20 °C) | ITTC 表插值（节点精确） | 998.2 kg/m³（IAPWS/ITTC） | 0.05 kg/m³ | PASS |
| 淡水 μ(20 °C) | 派生 ρ·ν | 1.002e-3 Pa·s（Kestin/IAPWS） | 0.5% | PASS |
| 淡水 μ(15 °C) | 派生 ρ·ν | 1.138e-3 Pa·s | 0.5% | PASS |
| 海水 ν(15 °C) | ITTC 表 | 1.1883e-6 m²/s（ITTC-1957 线定义值） | 0.1% | PASS |
| 表值 vs Vogel 相关式 | 表 ρ·ν | `μ=A·10^{B/(T_K−C)}` | 10–30 °C ≤1%，0–30 °C <2.5% | PASS（0 °C 处 2.2%，已在容差内并记录） |
| 空气 μ(15 °C) | Sutherland | 1.789e-5 Pa·s（White） | 0.5% | PASS |
| 空气 ρ(15 °C,101325 Pa) | 理想气体 | 1.225 kg/m³（ISA） | 0.2% | PASS |
| 空气声速(15 °C) | √(γRT) | 340.3 m/s | 0.2% | PASS |
| 空气 ν(15 °C) | μ/ρ | 1.46e-5 m²/s | 0.5% | PASS |
| 海冰 ρ/E/σ_f | Timco & Weeks 2010 | 720–940 kg/m³ / 1–10 GPa / 0.5–1.5 MPa | 区间内 | PASS |
| AH36 fy | IACS UR W11 | 355 MPa | 精确 | PASS |
| AH36 E | DNV-RP-C201 | 206 GPa | 3% | PASS |
| 5083-H116 E | EN 1999-1-1 | 70 GPa | 3% | PASS |
| 泥沙 Stokes 沉速 | 表值 | `w=(ρ_s−ρ)gd²/18μ` 现场计算 | 15% | PASS |
| 干容重 = ρ_s(1−n) | 表值 | 恒等式 | 6% | PASS |

**是否 <3%**：水/空气的全部热物性锚点均 <1%（最优 0.0%，最差 2.2% 出现在
0 °C 的 Vogel 交叉校验，且 Vogel 本身在该温度段精度较低）；结构材料规范值 0%；
冰力学量按文献区间判定（冰强度本身散度 ±40%，不允许声称 <3%，见 §7）。
泥沙量为工程经验值，量级一致即通过，不声称 <3%。

## 7. 注意点 / 局限 / 适用范围

1. **表域窄**：水表 0–30 °C、1 atm（ITTC 工作段）。低温（海水凝固点 ~−1.9 °C、
   冰下）与高压需 UNESCO/Tanaka EOS 扩展——**v1 明确不支持，直接报错**。
2. **冰强度是区间量**：σ_f ±40%、E 1–10 GPa 强依赖应变率与孔隙率。库给名义值
   并强制在 `note` 写区间；任何「冰载荷误差 <3%」的说法都是错的，本库不背这个锅。
3. **泥沙是工程经验值**：Manning n、临界 Shields、沉速都是量级值，粘性泥沙需
   站点专属侵蚀律，库里只给无量纲近似。
4. **单位陷阱**：船舶平台里 `rho_water=1.0`、`nu_water=0.03` 是 LBM/相场**格子单位**，
   不是 SI。盘点表把这类明确标 `review` 不迁移（`docs/props-migration-map.md`）。
5. **气体域上限 1 MPa**：理想气体 + Sutherland 在更高压下需真实气体修正。
6. **不做的**：材料温度依赖（钢 E 随温度软化、冰 E 随温度）、海水盐度插值、
   混合物/乳化液——留给 v2。

## 8. 复现

```bash
python -m pytest tests/test_props_db.py -q          # 31 passed
python -m tensordg.props validate                    # OK: 16 substances, 5 classes, 125 rows
PYTHONPATH=. python examples/props_solver_integration.py
PYTHONPATH=. python scripts/props_migration_inventory.py <ship-repo> > docs/props-migration-map.md
```

## 9. 后续（v2 backlog）

| # | 事项 | 量化验收 |
|---|---|---|
| 1 | ship 仓为第二接入仓（≥3 个模块改走库查询） | 迁移映射表 `migrate` 行中 ≥3 处落地，测试对比迁移前后数值差 <1e-9 |
| 2 | UNESCO/Tanaka EOS 扩展：水 0–100 °C、0–10 MPa、盐度 0–40 g/kg | 与 ITTC 表重叠段误差 <0.2% |
| 3 | 冰力学温度依赖表（σ_f(T)、E(T)） | 与 Timco & Weeks 图 4/7 数字化点误差 <15% |
| 4 | HTTP 查询面（平台 `/api/v1/props`） | 与 CLI 同值（同一 `QueryResult.as_dict()`），契约测试 ≥10 例 |