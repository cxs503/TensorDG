# CASE-LAX-DG · 波结构/容许性审计（程序化渲染，零手抄）

生成：2026-11-15T12:52:48Z　锚点：`CASE-LAX-DG`

## 波位置误差 W1（`|dx|/|x_exact-x0|`）

| leg | head | tail | contact | shock |
|---|---|---|---|---|
| p3.N400 | 0.105 % | 0.254 % | 0.035 % | 0.049 % |
| p3.N800 | 0.079 % | 0.117 % | 0.030 % | 0.029 % |

## 间断厚度 W2（密度 10%–90%，格数 = w/dx）

| leg | contact w | contact w/dx | shock w | shock w/dx |
|---|---|---|---|---|
| p3.N400 | 0.00228 | 0.91300 | 0.00041 | 0.16208 |
| p3.N800 | 0.00127 | 1.01953 | 0.00029 | 0.23101 |

## 越界极值 W3 / 总变差 W4

| leg | rel L2 ρ | ρ 超调 | ρ 欠调 | ρ TV 比 | p 欠调 | p TV 比 | 越界节点 |
|---|---|---|---|---|---|---|---|
| p3.N400 | 3.116 % | 7.34 % | 4.94 % | 5.393 | 4.79 % | 11.099 | 1042 / 4000 |
| p3.N800 | 2.315 % | 7.37 % | 14.24 % | 7.764 | 11.06 % | 15.562 | 2113 / 8000 |

## 判据结论

```
{
 "acceptance_metric_untouched": "the ledger arm (relative L2, P3/N=400) is NOT re-graded by this audit",
 "admissibility_gate_proposed": "nodes_outside_exact_range == 0 and TV ratio <= 1.02 at the certified resolution",
 "admissibility_pass": false,
 "l2_pass_at_3pct": {
  "p3.N400": false,
  "p3.N800": true
 }
}
```
