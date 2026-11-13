import os, math, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
from tensordg.elastic_1d_layered import (
    solve_elastic_1d_layered, exact_rightgoing_scattering,
    scattering_coefficients, rightgoing_pulse_initial_state,
    layered_energy, layered_stress_profile, impedance, scattering_valid_time)
from tensordg.elastic_1d import solve_elastic_1d, gaussian_pulse

torch.set_num_threads(1)

# ---------------- 1. uniform reduction -------------------------------------
N, p, T = 50, 2, 0.2
v0 = lambda x: gaussian_pulse(x, 0.5, 0.05)
s0 = lambda x: torch.zeros_like(x)
c1, co1, m1 = solve_elastic_1d(ncells=N, degree=p, final_time=T,
                               initial_velocity=v0, initial_stress=s0)
c2, co2, m2 = solve_elastic_1d_layered(ncells=N, degree=p, final_time=T,
                                       rho=(1.0,), young=(1.0,),
                                       initial_velocity=v0, initial_stress=s0)
print("uniform reduction max|diff| =", float((co1 - co2).abs().max()),
      "steps", m1["steps"], m2["steps"])

# ---------------- 2. scattering measurement ---------------------------------
def peak_in(x, s, lo, hi):
    m = (x >= lo) & (x <= hi)
    xs, ss = x[m], s[m]
    i = int(torch.argmax(ss.abs()))
    if 0 < i < len(ss) - 1:
        y0, y1, y2 = float(ss[i-1].abs()), float(ss[i].abs()), float(ss[i+1].abs())
        denom = (y0 - 2*y1 + y2)
        delta = 0.0 if abs(denom) < 1e-300 else 0.5*(y0 - y2)/denom
        hstep = float(xs[i+1] - xs[i])
        # |s| along the parabola: |s|(delta) = y1 - (y0-y2)/2*delta*... use standard max value
        peak_abs = y1 - 0.25*(y0 - y2)*delta
        return float(torch.sign(ss[i])) * peak_abs, float(xs[i] + delta*hstep)
    return float(ss[i]), float(xs[i])

def run_case(name, rho2, e2, N=400, p=2, width=0.05, center=1.0, iface=2.0, L=4.0):
    z1, z2 = impedance(1.0, 1.0), impedance(rho2, e2)
    R, Tt = scattering_coefficients(1.0, 1.0, rho2, e2)
    c2 = math.sqrt(e2/rho2)
    n1 = int(round(iface/L*N))
    rho = [1.0]*n1 + [rho2]*(N - n1)
    ey = [1.0]*n1 + [e2]*(N - n1)
    v_init, s_init = rightgoing_pulse_initial_state(iface, 1.0, 1.0, center, width)
    t1, t2 = 0.5, 1.4
    x1, cf1, mm1 = solve_elastic_1d_layered(ncells=N, degree=p, final_time=t1,
                                            length=L, rho=rho, young=ey, bc="fixed-free",
                                            initial_velocity=v_init, initial_stress=s_init)
    x1s, v1s, s1s = layered_stress_profile(x1, cf1, 8)
    x2, cf2, mm2 = solve_elastic_1d_layered(ncells=N, degree=p, final_time=t2,
                                            length=L, rho=rho, young=ey, bc="fixed-free",
                                            initial_velocity=v_init, initial_stress=s_init)
    xs, vs, ss = layered_stress_profile(x2, cf2, 8)
    inc_peak, inc_x = peak_in(x1s, s1s, center + t1 - 0.4, center + t1 + 0.4)
    x_ref = 2*iface - center - t2
    x_tra = iface + c2*(t2 - (iface - center)/1.0)
    ref_peak, ref_x = peak_in(xs, ss, max(0.0, x_ref - 0.4), iface - 0.05)
    tra_peak, tra_x = peak_in(xs, ss, iface + 0.05, x_tra + 0.4)
    Rm, Tm = ref_peak/inc_peak, tra_peak/inc_peak
    ve, se = exact_rightgoing_scattering(xs, t2, iface, L, 1.0, 1.0, rho2, e2, center, width)
    l2 = float(torch.sqrt(torch.sum((ss - se)**2))) / float(torch.sqrt(torch.sum(se**2)))
    e0 = layered_energy(cf1, L/N, rho, ey)
    e2t = layered_energy(cf2, L/N, rho, ey)
    print(f"[{name}] Z1={z1:.4f} Z2={z2:.4f} c2={c2:.4f} R_th={R:+.6f} T_th={Tt:+.6f} "
          f"| R_meas={Rm:+.6f} ({abs(Rm-R)/max(abs(R),1e-12):.3%}) T_meas={Tm:+.6f} "
          f"({abs(Tm-Tt)/abs(Tt):.3%}) | L2={l2:.3%} peaks inc={inc_peak:.6f} "
          f"ref={ref_peak:.6f} tra={tra_peak:.6f} | steps={mm2['steps']}")
    return dict(name=name, R_theory=R, T_theory=Tt, R_meas=Rm, T_meas=Tm,
                err_R=abs(Rm-R), err_T=abs(Tm-Tt)/abs(Tt),
                l2=l2, steps=mm2["steps"], c2=c2, z1=z1, z2=z2)

t0 = time.time()
res = []
res.append(run_case("contrast  rho2=4,E2=4", 4.0, 4.0))
res.append(run_case("matched   rho2=2,E2=.5", 2.0, 0.5))
res.append(run_case("soft      rho2=1,E2=.25", 1.0, 0.25))
print("elapsed", round(time.time()-t0, 2), "s")
print(json.dumps(res, indent=1))