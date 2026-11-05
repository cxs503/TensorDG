"""Publish the m3714 progress report (paper-grade, figures embedded)."""
import base64
import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def _request(path, payload=None, token=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def login():
    body = {"username": USER, "password": PASS}
    for path in ("/tensor/auth/login", "/auth/login", "/login"):
        try:
            data = _request(path, body)
        except urllib.error.HTTPError as exc:
            print("login", path, "HTTP", exc.code, exc.read().decode()[:160])
            continue
        token = data.get("access_token") or data.get("token")
        if token:
            print("login ok via", path)
            return token
    raise SystemExit("could not obtain a token")


def figure(name, caption):
    with open(f"/tmp/rep_{name}.png", "rb") as handle:
        return {"name": name, "caption": caption,
                "png_base64": base64.b64encode(handle.read()).decode()}


MARKDOWN = r"""
# 二维限制器光滑阶数验证 —— 关闭 m2549 遗留的「二维未验证」欠账（TensorDG m3714）

## ① 案例介绍
**物理**：二维可压缩 Euler 方程的**等熵涡**——Euler 方程的**精确解**，周期方盒内纯对流输运不变形，
是检验高阶 DG「光滑流是否被限制器破坏」的标准算例（Cockburn & Shu 1989 的经典配置）。
本单靶心是 `docs/limiter_policy_verification.md` §5 第 2 条原文欠账：
「二维限制器（`tensordg/euler_dg_2d.py` 的可选 TVB）沿用同一个 `_troubled_cell_indicator`，
本次修复对它**连带生效但未单独验证**（未做二维光滑阶数测试）」。

| 项 | 值 |
|---|---|
| 域 / 边界 | $[-5,5]^2$，双周期 |
| 涡强度 $\beta$ | 5.0（背景流 $(u,v)=(1,1)$） |
| 终了时刻 | $T=1.0$（涡心移至 $(1,1)$） |
| 阶数 | P1（设计阶 2）：$8^2,16^2,32^2$；P2（设计阶 3）：$8^2,12^2,24^2$ |
| 通量 / 时间推进 | HLLC + SSP-RK3，CFL 0.2，初值模态 $L^2$ 投影 |
| 限制器设定 | none / TVB($M$), $M\in\{0,1,4,10,100\}$ / WENO($M=0,10$) |
| dtype | float64，CPU 单线程（本机 load15=122，遵守容量纪律：单核小算例、未上多核） |

## ② 怎么做的
1. **判据先于跑数**。m2549 修的是「判据量纲不一致」：旧二维判据拿 $O(dx\,u')$ 的一阶差商比
   $M\,dx^2$ 容差，故 $dx>1/M$ 后**每个**光滑格心都被判 troubled、网格越细限制越多。修好后判据量为
   $|d_L-\mathrm{minmod}(d_L,d_R)|=O(dx^2u'')$，由此**先验**预测惰性阈值 $M_{\rm th}\approx\max|\Delta^2u|/dx^2$，
   在**精确解**格心上直接算（不拟合）。
2. **阴性对照**：$M=0$（严格 TVD）必须咬光滑极值并掉阶，否则「大 $M$ 不掉阶」无判别力。
3. **惰性用逐位相等判定**（`torch.equal`）：判据若不触发，限制器不得改动任何模态系数，
   两条轨迹必须逐位相同——最强、最难作弊的证据。
4. 工具链：`benchmarks/limiter_2d_order_study.py`（可分片扫描）→ `validation/limiter_2d_order/*.json`
   →`benchmarks/report_limiter_2d_order.py`（合表+出图，matplotlib）；本报告数字全部程序化直读 JSON。
5. 回归固化：`tests/test_limiter_2d_order.py`（7 例：惰性 / 阴性对照 / `degree≥2` 掩码分支 /
   判据量纲语义 / 阶数保持）。

## ③ 计算结果（数字直读 `validation/limiter_2d_order/*.json`）

**P2（设计阶 3），$8^2\to12^2\to24^2$**

| 限制器 | EOC($\rho$) | EOC($p$) | 最差相对 $L^2$ | 与无限制器逐位相同 |
|---|---|---|---|---|
| 无 | **2.925** | **2.873** | 5.620e-03 | （基准） |
| TVB $M=0$ | 1.395 | 1.301 | 9.974e-02 | 否 |
| TVB $M=1$ | 1.135 | 1.177 | 3.698e-02 | 否 |
| TVB $M=4$ | **2.925** | **2.873** | 5.620e-03 | **是** |
| TVB $M=10$ | **2.925** | **2.873** | 5.620e-03 | **是** |
| TVB $M=100$ | **2.925** | **2.873** | 5.620e-03 | **是** |
| WENO $M=0$ | 1.341 | 1.562 | 8.265e-02 | 否 |
| WENO $M=10$ | **2.925** | **2.873** | 5.620e-03 | **是** |

**P1（设计阶 2），$8^2\to16^2\to32^2$**

| 限制器 | EOC($\rho$) | EOC($p$) | 最差相对 $L^2$ | 逐位相同 |
|---|---|---|---|---|
| 无 | **2.038** | **2.003** | 3.116e-02 | （基准） |
| TVB $M=0$ | 1.386 | 1.431 | 9.982e-02 | 否 |
| TVB $M=1$ | 1.040 | 1.332 | 3.289e-02 | 否 |
| TVB $M=10$ | **2.038** | **2.003** | 3.116e-02 | **是** |
| WENO $M=10$ | **2.038** | **2.003** | 3.116e-02 | **是** |

**惰性阈值：先验预测 vs 实测**

| 阶 | 预测 $M_{\rm th}=\max|\Delta^2|/dx^2$ | 实测不再触发区间 | 一致 |
|---|---|---|---|
| P2（$24^2$） | **3.42**（x 3.42 / y 3.05） | $M\in(1,4]$ | ✅ |
| P1（$32^2$） | **3.52**（x 3.49 / y 3.52） | $M\in(1,10]$ | ✅ |

**独立交叉核对**：P1、$16^2$、无限制器、$\rho$ 相对 $L^2$ = **3.8091e-03**，与仓库既有回归钉值
`tests/test_euler_dg_2d_design_order.py::3.8091e-03` 一致到 5 位有效数字 ⇒ 新数字与仓库既有口径同源。

## ④ 结果图
- 阶数梯子（双对数误差-网格 + $h^{p+1}$/$h^1$ 参考斜率）：`none/tvb_M4/M10/M100/weno_M10` 五条曲线
  **完全重合**在 $h^3$ 上，`tvb_M0/M1/weno_M0` 明显跌向 $h^1$。
- 场量云图（P2 $16^2$：$\rho$、$|\mathbf u|$、$\rho-\rho_{\rm exact}$ × {无限制器, TVB $M=0$}）：
  可见 $M=0$ 把涡心光滑极值**削平**，误差场在涡核成亮环——掉阶的物理来源。
- EOC 柱状 vs 设计阶 3。

## ⑤ 注意点 / 局限
1. **主要结论**：m2549 对二维的修复**确实连带生效**，修好后 `tvb_constant` 在二维也成为**网格无关参数**
   （$M\ge4$ 惰性、$M\le1$ 必咬），与精确解二阶差商的量级预测吻合。欠账关闭。
2. **$M$ 非万能（如实）**：$M=0$/$M=1$ 仍限光滑极值、全局阶压到 $\approx1.0$–$1.4$，是 minmod 类检测器
   的经典保守性（Cockburn & Shu 1989 引入 $M dx^2$ 正为此）。本单是**测出**阈值并证明可预估，非消除该现象。
3. **WENO 与 TVB 共享检测器**：WENO($M=10$) 与无限制器**逐位相同**；一旦触发（$M=0$）同样掉阶
   （1.34–1.56）。即「WENO 重建本身不是光滑流保阶机制」，与 Zhang & Shu (2010) 综述一致。
4. **未做**：非结构网格、三维、$p>3$、float32/GPU、NS 粘性限制器、二维**间断**算例（见
   `docs/limiter_2d_verification.md`）；**性能开销（76–123%）仍未复核**，如实留档。
5. **踩坑**：`_vortex_error` 返回扁平字典（含 `pressure_max`）而非嵌套字典，初版 harness 直接 `TypeError`；
   `vortex_primitive` 在格心上须传一维 `x[:,None]`，传成四维求积坐标会广播失配。

## ⑥ 误差判定（vs 解析解，逐项显式回答「是否 <3%」）
| 关键量 | 值 | 基准 | 相对误差 | <3% |
|---|---|---|---|---|
| P2 + TVB($M\ge4$) 最细网格最差场误差 | 5.620e-03 | 解析解 | **0.562%** | **是** |
| P2 无限制器（对照） | 5.620e-03 | 解析解 | 0.562% | 是 |
| P2 观测阶 vs 设计阶 3（$M\ge4$） | 2.925 / 2.873 | 3 | −2.5% / −4.2% | $\rho$ 场是（2.5%），$p$ 场 **否**（4.2%，如实） |
| P1 无限制器 $16^2$ $\rho$ | 3.8091e-03 | 解析解 | 0.381% | 是 |
| P1 + TVB($M=10$) $32^2$ 最差场（$u,v$） | 3.116e-02 | 解析解 | **3.116%** | **否**（P1 自身离散误差，限制器未加剧：与无限制器逐位相同；同网格 $\rho$ 为 0.093%） |
| TVB $M=0$ 最细网格最差场（阴性对照） | 9.974e-02 | 解析解 | 9.97% | 否（**故意**：对照必须失败） |
| 惰性阈值预测 vs 实测（P2） | $M_{\rm th}=3.42$ | 实测 $M=4$ 起惰性 | 预测落在 $(1,4]$ | 是 |

**显式结论**：主线判据「惰性 → 与无限制器逐位相同 → 观测阶=设计阶」在 P2 成立
（EOC 2.925 vs 3 = 差 **2.5% < 3%**，且 $M\ge4$ 时**逐位相同**，误差为零无歧义）。
P1 @ $32^2$ 的最差场 3.116% **超 3% 门**，来源是 P1 自身分辨率而非限制器（限制器在该网格与无限制器
逐位相同），如实记录不作验收点。honest-FAIL 不粉饰。

## 交付物
`benchmarks/limiter_2d_order_study.py` · `benchmarks/report_limiter_2d_order.py` ·
`tests/test_limiter_2d_order.py` · `validation/limiter_2d_order/*.json` ·
`docs/limiter_2d_order_verification.md` · `docs/figures/limiter_2d_order/*.png`
"""


def main():
    token = login()
    payload = {
        "title": "TensorDG m3714：二维限制器光滑阶数验证——关闭「二维未验证」欠账",
        "summary": ("用等熵涡精确解（P1/P2，8²→24²/32²）验证 m2549 修复后的二维 TVB/WENO 限制器："
                    "M≥4 时与无限制器逐位相同、P2 观测阶 2.925（设计阶 3，差 2.5%<3%）；"
                    "M≤1 与 M=0 掉阶至 1.0–1.4（阴性对照成立）；惰性阈值实测 M∈(1,4] "
                    "与精确解预测 3.42 吻合。新回归 7 例。"),
        "scenario_key": os.environ.get("TENSOR_SCENARIO_KEY", "tensordg"),
        "mission_id": 3714,
        "tags": ["TensorDG", "limiter", "TVB", "WENO", "verification", "isentropic-vortex",
                 "order-of-accuracy", "m2549-debt"],
        "markdown": MARKDOWN,
        "figures": [
            figure("order_ladder", "阶数梯子：相对 L² 误差 vs 网格数（双对数）。无限限制器与 "
                                   "TVB(M=4/10/100)、WENO(M=10) 完全重合于设计阶 h³；"
                                   "TVB(M=0/M=1)、WENO(M=0) 跌向 h¹。"),
            figure("fields_limit", "场量云图（P2，16²）：上两行为 ρ 与 |u|，第三行为 ρ−ρ_exact；"
                                   "左列为无限制器，右列为 TVB M=0（严格 TVD）——光滑涡心被削平。"),
        ],
    }
    raw = json.dumps(payload).encode()
    print("payload bytes:", len(raw))
    data = _request("/tensor/reports", payload, token)
    print("report published:", json.dumps(data)[:300])


if __name__ == "__main__":
    main()