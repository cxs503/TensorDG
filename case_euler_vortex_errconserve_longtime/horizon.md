# 误差守恒的时间地平线：t = 40 延伸读数

预测冻结档案 digest `3e02a7a95a7c37a7`；预测时刻 t = 40。

| 族 | 实测 % | 最佳模型 | 该模型偏差 | 区间内 | 扩展窗口 α (robust) | α ≤ 1 | 容差内 |
|---|---|---|---|---|---|---|---|
| `dg_d2_n32_cfl0.2` | 未落盘 | — | — | — | — | — | — |
| `dg_d1_n32_cfl0.2` | 未落盘 | — | — | — | — | — | — |
| `dg_d3_n32_cfl0.2` | 未落盘 | — | — | — | — | — | — |
| `p0_n64` | 16.0445 | tail_local | -26.20% | no | 0.479 | yes | NO |

## 参考解截断缺陷：min_image 折叠 vs 格点求和（最坏场，随位移增长）

| t | 缺陷 % |
|---|---|
| 1 | 0.000148617 |
| 10 | 0.000191022 |
| 20 | 0.000191022 |
| 40 | 0.000191022 |

## 裁决：**INCOMPLETE**

```json
{
  "status": "INCOMPLETE",
  "checks": {
    "design_measured": false
  },
  "reason": "design family t = 40 checkpoint has not landed"
}
```
