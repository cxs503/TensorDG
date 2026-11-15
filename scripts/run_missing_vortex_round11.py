"""Run the 8 missing LADDER checkpoints for CASE-DG-EULER-VORTEX-ERR (round-7 debt)."""
import json, sys
from pathlib import Path
from benchmarks import case_euler_vortex_errconserve as ec

MISSING = [
    {"task":"dg","degree":3,"cells":32,"final_time":1.0,"cfl":0.2},
    {"task":"dg","degree":3,"cells":32,"final_time":2.0,"cfl":0.2},
    {"task":"dg","degree":3,"cells":32,"final_time":5.0,"cfl":0.2},
    {"task":"dg","degree":3,"cells":32,"final_time":10.0,"cfl":0.2},
    {"task":"dg","degree":1,"cells":64,"final_time":2.0,"cfl":0.2},
    {"task":"dg","degree":1,"cells":64,"final_time":5.0,"cfl":0.2},
    {"task":"dg","degree":1,"cells":64,"final_time":10.0,"cfl":0.2},
    {"task":"dg","degree":2,"cells":32,"final_time":20.0,"cfl":0.2},
]
out = Path("case_euler_vortex_errconserve/results")
out.mkdir(parents=True, exist_ok=True)
for cfg in MISSING:
    key = ec.task_key(cfg)
    if (out / f"{key}.json").exists():
        print(f"skip existing {key}", flush=True); continue
    rec = ec.run_task(cfg)
    rec["task_key"] = key
    (out / f"{key}.json").write_text(json.dumps(rec, indent=2))
    print(f"landed {key}: max%={rec['max_percent']:.4f} steps={rec['steps']} wall={rec['wall_seconds']:.1f}s", flush=True)
print("DONE")
