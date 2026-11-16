# CASE-LAX-DG · A0 生产腿闭合（production-leg closure）

> 容差从注册算例导入 = `0.0300`；度量 = whole-field relative L2 vs exact Riemann solution, worst field, 10-node Legendre-Gauss (imported from benchmarks.lax_case_pool.run_ladder)。

## 旧生产腿（case-pool 阶梯最细腿）

| 腿 | DOF | 最差 L2 | 场 | 判定 | 裕度 |
|---|---|---|---|---|---|
| p3.N400 | 1600 | 3.116 % | density | FAIL | x0.963 |

## 新生产腿（认证配置）

| 腿 | DOF | 最差 L2 | 场 | 判定 | 裕度 |
|---|---|---|---|---|---|
| p2.N800 | 2400 | 2.570 % | density | PASS | x1.167 |

## 最小 DOF 通过腿（枚举 + 认证并集）

| p2.N800 | DOF 2400 | 最差 L2 2.570 % |

## honest-FAIL 保留

- p3.N400 全场 L2 = 3.116 %（间断主导，超 3 % 门槛），原样保留，不被 p2.N800 的 PASS 改写。

## A0 闭合判定：**PASS**

- 推荐：re-point the production leg at the certified configuration p2.N800 (minimal DOF that passes the < 3 % bar)
- 已实现：True
