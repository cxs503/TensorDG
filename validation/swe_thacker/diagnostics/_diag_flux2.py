"""Scratch diagnostic 8: manual HLL recomputation to locate flux bookkeeping."""
import torch

from tensordg.swe_1d import hll_flux, physical_flux, sound_speed, GRAVITY

# synthetic interface state: still water, h jump (as produced by the DG traces)
left = torch.tensor([[0.7304, 0.0], [1.1523, 0.0]], dtype=torch.float64)
right = torch.tensor([[0.7520, 0.0], [1.3332, 0.0]], dtype=torch.float64)
print("code hll_flux:", hll_flux(left, right, "hll"))

h_l, h_r = left[:, 0], right[:, 0]
u_l, u_r = left[:, 1] / h_l, right[:, 1] / h_r
c_l, c_r = sound_speed(left), sound_speed(right)
s_l = torch.minimum(u_l - c_l, u_r - c_r)
s_r = torch.maximum(u_l + c_l, u_r + c_r)
fl, fr = physical_flux(left), physical_flux(right)
manual = (s_r[:, None] * fl - s_l[:, None] * fr + s_l[:, None] * s_r[:, None] * (right - left)) / (s_r - s_l)[:, None]
print("s_l,s_r:", s_l, s_r)
print("fl,fr:", fl, fr)
print("manual:", manual)
print("dissipation-only mass term:", (s_l * s_r * (h_r - h_l) / (s_r - s_l)))