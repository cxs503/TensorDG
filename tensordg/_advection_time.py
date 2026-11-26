"""Time integration for spatially uniform Cartesian DG transport velocities."""
import math

from .runge_kutta import rk_step


def velocity_components(values, dimension):
    components = tuple(float(a) for a in values)
    if len(components) != dimension or not all(map(math.isfinite, components)):
        raise ValueError(f"velocity must contain {dimension} finite components")
    return components


def advance_transport(coeff, spatial_rhs, spacing, degree, tableau, velocity, start_time,
                      final_time, cfl, max_speed, max_dt, velocity_jump_times):
    if max_dt is not None and (not math.isfinite(max_dt) or max_dt <= 0.0):
        raise ValueError("max_dt must be positive and finite")

    def components_of(values):
        return velocity_components(values, len(spacing))

    variable = callable(velocity)
    fixed = None if variable else components_of(velocity)
    jumps = sorted(float(t) for t in velocity_jump_times)
    if jumps and (not variable or len(set(jumps)) != len(jumps)
                  or not all(start_time < t < final_time for t in jumps)):
        raise ValueError("velocity_jump_times requires distinct interior times and a velocity callback")
    boundaries = [start_time, *jumps, final_time]
    # A right-continuous event belongs to the following interval. Clamp all
    # evaluations on the preceding interval to the last float on its side.
    intervals = [(a, b, math.nextafter(b, a) if b != final_time else b)
                 for a, b in zip(boundaries, boundaries[1:])]

    if max_speed is not None:
        bounds = components_of(max_speed)
        if min(bounds) < 0:
            raise ValueError("max_speed must contain nonnegative bounds")
        if not variable and any(abs(a) > b for a, b in zip(fixed, bounds)):
            raise ValueError("constant velocity exceeds max_speed")
    elif variable:
        samples = []
        for begin, end, last in intervals:
            for i in range(5):
                t = last if i == 4 else min(begin + i * ((end - begin) / 4), last)
                samples.append(components_of(velocity(t)))
        bounds = tuple(max(map(abs, axis)) for axis in zip(*samples))
    else:
        bounds = tuple(map(abs, fixed))

    wave_rate = sum(a / h for a, h in zip(bounds, spacing))
    if variable and final_time > start_time and not wave_rate and max_speed is None and max_dt is None:
        raise ValueError("all sampled velocities vanish; supply max_speed or max_dt to resolve the interval")
    steps = 0
    largest_dt = 0.0

    def rhs(state, t=None):
        components = components_of(velocity(min(max(t, begin), last))) if variable else fixed
        if variable and max_speed is not None and any(abs(a) > b for a, b in zip(components, bounds)):
            raise ValueError("stage velocity exceeds max_speed")
        if variable and max_speed is None:
            stage_rate = sum(abs(a) / h for a, h in zip(components, spacing))
            stage_cfl = dt * (2 * degree + 1) * stage_rate
            if stage_cfl > cfl and not math.isclose(stage_cfl, cfl, rel_tol=1e-12):
                raise ValueError("stage velocity exceeds the requested CFL; supply max_speed or reduce max_dt")
        return spatial_rhs(state, components)

    for begin, end, last in intervals:
        span = end - begin
        segment_steps = math.ceil(span / (cfl / ((2 * degree + 1) * wave_rate))) if wave_rate else 0
        if max_dt is not None:
            segment_steps = max(segment_steps, math.ceil(span / max_dt))
        dt = span / segment_steps if segment_steps else 0.
        steps += segment_steps
        largest_dt = max(largest_dt, dt)
        for step in range(segment_steps):
            if variable:
                coeff = rk_step(rhs, coeff, dt, tableau, time=begin + step * dt)
            elif tableau.name == "rk4":
                # Preserve the summation order of the established default.
                k1 = rhs(coeff)
                k2 = rhs(coeff + 0.5 * dt * k1)
                k3 = rhs(coeff + 0.5 * dt * k2)
                k4 = rhs(coeff + dt * k3)
                coeff = coeff + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                coeff = rk_step(rhs, coeff, dt, tableau)

    return coeff, ("time-dependent" if variable else fixed), bounds, steps, largest_dt
