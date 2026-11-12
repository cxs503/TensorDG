import json, torch
torch.set_num_threads(1)
import tensordg.kovasznay as kv
x,y,coeff,meta = kv.solve_kovasznay_dg_2d(ncells=12, degree=1, sound_speed=30.0,
                                          final_time=0.4, chunks=10, cfl=0.5)
json.dump(meta, open("validation/kovasznay/probe_time_p1_12.json","w"), indent=1, default=str)
print(json.dumps([{k: (round(v,6) if isinstance(v,float) else v) for k,v in e.items()} for e in meta["history"]], indent=1))
