"""Two independently scheduled walls with a hold and a reversal on the unit slab."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import torch
from tensordg import heat_error, solve_heat_1d_wall_history
from benchmarks.ramped_wall_heat_1d import reference


TIMES = (0., .035, .06, .11, .14)
WALLS = ((0.,0.), (.07,-.035), (.07,-.035), (-.03,.015), (-.03,.075))


AMPLITUDES = ((.25,-.10), (0.,0.), (-.15,.30), (.05,0.))
WAVES = (math.pi,2*math.pi)


def exact(x, t, heat=False, pulses=False):
    # Superpose delayed changes of slope. The mirrored (2,-1) ramp has
    # wall rates (-1,2), and reflection reverses the sign of rightward heat.
    result = torch.zeros_like(x)
    last_left = last_right = 0.
    for i, start in enumerate(TIMES[:-1]):
        if t <= start:
            break
        duration = TIMES[i+1]-start
        left, right = ((b-a)/duration for a,b in zip(WALLS[i],WALLS[i+1]))
        dl, dr = left-last_left, right-last_right
        forward, mirrored = (2*dl+dr)/3, (dl+2*dr)/3
        result += forward*reference(x,t-start,heat)
        result += (-mirrored if heat else mirrored)*reference(1-x,t-start,heat)
        if pulses:
            # Convolve each switched sine forcing with its scalar heat kernel.
            age = t-start
            for mode,wave in enumerate(WAVES):
                change = AMPLITUDES[i][mode] - (AMPLITUDES[i-1][mode] if i else 0.)
                decay = math.expm1(-wave**2*age)
                if heat:
                    result -= change*wave*(age+decay/wave**2)*torch.cos(wave*x)
                else:
                    result -= change*decay*torch.sin(wave*x)
        last_left, last_right = left, right
    return result


def measure(n, degree, pulses):
    def generation(amplitudes):
        return lambda x: sum(a*w*w*torch.sin(w*x) for a,w in zip(amplitudes,WAVES))
    sources = [generation(a) for a in AMPLITUDES] if pulses else None
    x, states, meta = solve_heat_1d_wall_history(TIMES,WALLS,n,degree,initial_temperature=torch.zeros_like,
                                                  backend="exact-time",integrate_face_fluxes=True,interval_sources=sources)
    faces = torch.linspace(0.,1.,n+1,dtype=states.dtype)
    cumulative = torch.stack([exact(faces,t,heat=True,pulses=pulses) for t in TIMES])
    heat = states.new_tensor(meta["interval_face_heat"])
    error = max(heat_error(x,states[i],t,reference=lambda x,t: exact(x,t,pulses=pulses))["absolute_l2"] for i,t in enumerate(TIMES) if i)
    residual = states[:,:,0].diff(dim=0)/n+heat.diff(dim=1)
    if pulses:
        for i,amplitudes in enumerate(AMPLITUDES):
            for a,w in zip(amplitudes,WAVES):
                residual[i] += (TIMES[i+1]-TIMES[i])*a*w*torch.cos(w*faces).diff()
    return dict(cells=n,degree=degree,max_knot_l2=error,
                max_interval_face_heat_error=float((heat-cumulative.diff(dim=0)).abs().max()),
                max_total_face_heat_error=float((heat.sum(dim=0)-cumulative[-1]).abs().max()),
                max_cell_balance=float(residual.abs().max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pulses",action="store_true",help="switch two volumetric heating modes at the wall knots")
    pulses = parser.parse_args().source_pulses
    rows = []
    for degree in (1,2,3):
        previous = None
        grids = (8,16,32,64) if pulses else (8,16,32)
        for n in grids:
            row = measure(n,degree,pulses)
            error = row["max_knot_l2"]
            order = None if previous is None else math.log2(previous/error)
            passed = (order is None or order>degree+.7) and row["max_cell_balance"]<1e-10
            if n==grids[-1]:
                passed &= max(row["max_interval_face_heat_error"],row["max_total_face_heat_error"]) < (1e-4,5e-8,5e-11)[degree-1]
            rows.append(dict(**row,temperature_order=order,passed=passed))
            previous = error
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,source_amplitudes=AMPLITUDES if pulses else None,times=TIMES,walls=WALLS,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
