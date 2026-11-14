# CASE-LAX-DG limiter arms -- machine summary

generated: 2026-11-14T08:28:09Z

| arm | C1 | C2 | C3 | R3 | R5 | verdict |
|---|---|---|---|---|---|---|
| none | PASS | PASS | FAIL | FAIL | PASS | FAIL |
| tvb | PASS | PASS | FAIL | FAIL | FAIL | FAIL |
| weno | PASS | PASS | FAIL | FAIL | FAIL | FAIL |

| arm | degree | whole-field order | analytic order | R3 requirement | R3 |
|---|---|---|---|---|---|
| none | p1 | 0.9142 | 2.6963 | 1.70 | PASS |
| none | p2 | 1.027 | 1.6665 | 2.55 | FAIL |
| none | p3 | 0.484 | 0.2738 | 3.40 | FAIL |
| tvb | p1 | 0.7792 | 2.1 | 1.70 | PASS |
| tvb | p2 | 0.7626 | 2.0113 | 2.55 | FAIL |
| tvb | p3 | 0.7593 | 1.9758 | 3.40 | FAIL |
| weno | p1 | 0.7377 | 1.6384 | 1.70 | FAIL |
| weno | p2 | 0.7146 | 1.6386 | 2.55 | FAIL |
| weno | p3 | 0.7166 | 1.6849 | 3.40 | FAIL |

reproduction (arm=none vs tracked pool record): PASS max rel diff = 0.0
