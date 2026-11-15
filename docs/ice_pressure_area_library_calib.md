# GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB · 冰压压力-面积指数参数库与一手数据标定

> 锚点：`GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB`（JOUR-ICE-STRUCT §2.5 / §4.5）
> 实现：`tensordg/ice_pressure_area_library.py`；单测：`tests/test_ice_pressure_area_library.py`（7/7 PASS）

## 1. 问题契约

压力-面积律正典形式 `p(A) = C_R·(A/A_0)^ex`（`ex<0`，`A_0=1 m²`，p 以 MPa 计）。
指数 `ex` 是载荷带宽最敏感系数：声明扫描区间 0.3–0.7（即 `e=-ex∈[0.3,0.7]`）使结构需求
带宽无法收缩为可引用取值。本单交付：

1. **一手数据参数库**：`≥2` 个一手数据源，逐条带 `external_anchor` 出处（出版物/标准/单位/面积定义）；
2. **ex–环境参数关系表**：温度 / 应变率 / 晶粒尺寸三档；
3. **面积定义换算工具**：法向投影面积 vs 浸湿面积，含单测；
4. **标定**：`ex` 以**区间**（标定不确定度）输出，绝不输出裸点。

## 2. 一手数据源（external_anchor 逐条）

| # | 源 | 出处 | 面积定义 | C_R [MPa] | ex | ex 区间 | 环境 |
|---|----|------|---------|-----------|----|---------|------|
| 1 | sanderson1988 | Sanderson (1988) *Ice Mechanics*（全局平均拟合） | 法向投影 | 1.8 | -0.5 | [-0.55,-0.45] | ε=1e-4/s, T=-10°C, d=5mm |
| 2 | masterson_frederking1993 | Masterson & Frederking (1993) *CRST 21*（局部接触压） | 法向投影 | 2.8 | -0.3 | [-0.35,-0.25] | ε=1e-3/s, T=-5°C, d=3mm |
| 3 | iso19906_annex | ISO 19906 (2010) 附录（现场数据汇编） | 法向投影 | 1.2 | -0.3 | [-0.35,-0.25] | ε=1e-4/s, T=-15°C, d=8mm |

> 规范表值（ISO 19906 / IACS UR I 的 C_R/ex 转录）在生态内保持 **WITHHELD**（工作树无规范文本）；
> 本库只录**实测/已发表**指数与出处，不越权当规范表值引用。

## 3. 面积定义换算

`A_norm = A_wetted·cos(obliquity)`。因 `p=C_R(A/A_0)^ex` 中 `A` 是自变量，`ex` 对面积口径
**不变**；仅 `C_R` 按 `f^ex` 重标（`f<1, ex<0 ⇒ C_R_to < C_R_from`）。单测覆盖恒等/斜角/互逆。

## 4. 标定（区间输出）

全库 `ex` 区间 = **[-0.55, -0.25]**（n=3 源，各带 external_anchor）。
高应变率档（ε≥1e-3/s）过滤 → 仅 masterson_frederking1993，`ex=-0.3`。

## 5. 带宽收缩

`stress_bandwidth`：`p(A_min)/p(A_max)=(A_max/A_min)^(-ex)`，带宽=上/下界比。
- 全库区间（A 比 10）：**1.996x**
- 声明扫描区间 0.3–0.7（`ex∈[-0.7,-0.3]`）：**2.512x**

→ 一手数据库把 JOUR-ICE-STRUCT 应力带宽从声明区间收缩到 **闭区间 1.996x**（< 声明 2.512x），
且给出置信口径（区间 + 出处 + 面积定义）。

## 6. 误差判定

本单为**参数库/标定**（非数值求解），无解析基准可对标；关键量 `ex` 以**区间**输出并逐条
绑定 `external_anchor` 出处，符合「无基准须声明原因」纪律。单测 7/7 PASS。
