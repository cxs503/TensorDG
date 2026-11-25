"""SIPG Poisson on the unit square with per-side Dirichlet or Neumann data.

``tensordg.poisson_2d`` hard-codes homogeneous Dirichlet data on all four sides
and the single manufactured solution ``sin(pi x) sin(pi y)``.  This module keeps
that Dirichlet assembly bit-comparable (it reuses the very same face operators)
and adds the capability the repository was missing: the **natural (Neumann)
boundary condition**, chosen independently on each side, together with an
arbitrary forcing ``f(x, y)``.

The weak form of ``-Delta u = f`` reads

    a(u, v) = int_Omega grad(u).grad(v)
              - sum_{Gamma_D} ( {grad(u).n} [v] + {grad(v).n} [u] )
              + sum_{Gamma_D} (sigma/h) [u] [v]
            = int_Omega f v + sum_{Gamma_N} int_{Gamma_N} g_N v ,

so a Neumann face changes only the right-hand side: it contributes the
consistency load ``int g_N v`` and *no* trace/penalty terms (Arnold, Brezzi,
Cockburn & Marini, SIAM J. Numer. Anal. 39 (2002), eq. (2.9)).  ``g_N`` is the
**outward** normal derivative ``du/dn``.

Pure-Neumann data leave the discrete operator with its one-dimensional constant
nullspace.  That case is detected, checked (row sums must vanish, i.e.
``A @ 1 == 0``) and solved with the rank-one gauge

    A + (1/ndofs) 1 1^T

which fixes the spatial mean to zero for a compatible right-hand side. The
constant vector in the modal basis has only its cell-mean entries set to one.
The assembled source and outward normal derivative must sum to zero when
tested against this vector.
"""
from __future__ import annotations



import numpy as np
import torch

from .poisson_2d import (
    _add_block,
    _face_basis,
    _operators,
    dirichlet_face_operator,
    interior_face_operator,
)

BC_SIDES = ("left", "right", "bottom", "top")
_ALLOWED = ("dirichlet", "neumann", "robin")


def _resolve_bc(bc) -> dict[str, str]:
    """Normalise a partial ``{side: kind}`` mapping (default: Dirichlet)."""
    resolved = {side: "dirichlet" for side in BC_SIDES}
    if not bc:
        return resolved
    for side, kind in bc.items():
        if side not in resolved:
            raise ValueError(f"unknown boundary side {side!r}; use one of {BC_SIDES}")
        if kind not in _ALLOWED:
            raise ValueError(f"boundary kind must be one of {_ALLOWED}, got {kind!r}")
        resolved[side] = kind
    return resolved


def _boundary_samples(data, side, centers, nodes, dtype=None):
    """One row per face, one column per tangential quadrature point."""
    along = centers[:, None] + nodes[None, :]/(2*len(centers))
    fixed = torch.full_like(along, float(side in ("right", "top")))
    x, y = (fixed, along) if side in ("left", "right") else (along, fixed)
    values = data(x, y) if callable(data) else data
    return torch.broadcast_to(torch.as_tensor(values, device=along.device, dtype=dtype or along.dtype), along.shape)


def _surface_capacity_samples(data,side,centers,nodes):
    def checked(x,y):
        value = data(x,y) if callable(data) else data
        if torch.is_complex(torch.as_tensor(value)):
            raise ValueError("wall capacity must be real")
        return value
    capacity = _boundary_samples(checked,side,centers,nodes)
    if not torch.isfinite(capacity).all() or (capacity<0).any():
        raise ValueError("wall capacity must be finite and nonnegative")
    return capacity


def _check_storage_sides(capacities,types):
    if any(types.get(side)!="robin" for side in capacities):
        raise ValueError("wall capacity requires a Robin side")


def _wall_storage(grid,types,capacities,initial_temperature=None):
    from scipy.sparse import coo_matrix

    _check_storage_sides(capacities,types)
    x,y,degree = grid["x"],grid["y"],grid["degree"]
    nodes,w,phi,_,_,pl,pr,dl,dr = _operators(degree,x.device,x.dtype)
    local = (degree+1)**2
    dofs = np.arange(len(x)*len(y)*local).reshape(len(x),len(y),local)
    rows,cols,values = [],[],[]
    load = np.zeros(dofs.size) if initial_temperature is not None else None
    for side,data in capacities.items():
        horizontal = side in ("left","right")
        low = side in ("left","bottom")
        centers = y if horizontal else x
        samples = _surface_capacity_samples(data,side,centers,nodes)
        trace,_ = _face_basis(phi,pl if low else pr,dl if low else dr,"x" if horizontal else "y",1.)
        blocks = torch.einsum("iq,q,qk,ql->ikl",samples,w,trace,trace)/(2*len(centers))
        index = 0 if low else -1
        ids = dofs[index,:] if horizontal else dofs[:,index]
        if load is not None:
            values_on_wall = _boundary_samples(initial_temperature,side,centers,nodes)
            load[ids] += (((values_on_wall*samples*w) @ trace)/(2*len(centers))).numpy()
        rows.append(np.broadcast_to(ids[:,:,None],blocks.shape).ravel())
        cols.append(np.broadcast_to(ids[:,None,:],blocks.shape).ravel())
        values.append(blocks.numpy().ravel())
    return coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                      shape=(dofs.size,dofs.size)).tocsc(),load


def _film_samples(films, side, centers, nodes, load_dtype=None):
    if films is None or side not in films:
        raise ValueError(f"Robin side {side} requires (coefficient, ambient temperature)")
    coefficient, ambient = films[side]
    h = _boundary_samples(coefficient, side, centers, nodes)
    temperature = _boundary_samples(ambient, side, centers, nodes, load_dtype)
    if not torch.isfinite(h).all() or torch.any(h < 0) or not torch.isfinite(temperature).all():
        raise ValueError("Robin coefficients must be finite and nonnegative and ambient data finite")
    return h, temperature


def _contact_faces(data,nx,ny,device,dtype):
    faces = {}
    for axis,value in (data or {}).items():
        if axis not in ("x","y") or torch.is_complex(torch.as_tensor(value)):
            raise ValueError("contact resistance requires real x or y face data")
        field = torch.as_tensor(value,device=device,dtype=dtype)
        shape = (nx-1,ny) if axis=="x" else (nx,ny-1)
        if field.shape not in ((),shape) or not torch.isfinite(field).all() or (field<0).any():
            raise ValueError(f"{axis} contact resistance must be finite, nonnegative and scalar or shaped {shape}")
        faces[axis] = field.expand(shape)
    return faces


def _trace_face_data(data,nx,ny,device,dtype):
    """Validate one constant per face on each of its two traces."""
    for axis,pair in (data or {}).items():
        if axis not in ("x","y") or len(pair)!=2:
            raise ValueError("trace data require an x or y pair in increasing coordinate order")
        shape = (nx-1,ny) if axis=="x" else (nx,ny-1)
        fields = []
        for value in pair:
            if torch.is_complex(torch.as_tensor(value)) and not dtype.is_complex:
                raise ValueError("complex trace data require a complex load")
            field = torch.as_tensor(value,device=device,dtype=dtype)
            if field.shape not in ((),shape) or not torch.isfinite(field).all():
                raise ValueError(f"{axis} trace data must be finite and scalar or shaped {shape}")
            fields.append(field.expand(shape))
        yield axis,fields


def _add_trace_power(load,data):
    nx,ny,modes,_ = load.shape
    # A face-constant source tests only the constant tangential Legendre
    # mode. Normal basis values at the two endpoints are 1 and (-1)^p.
    low = (-1.)**torch.arange(modes,dtype=load.real.dtype,device=load.device)
    for axis,(lower,upper) in _trace_face_data(data,nx,ny,load.device,load.dtype):
        if axis=="x":
            load[:-1,:,:,0] += lower[...,None]/ny
            load[1:,:,:,0] += upper[...,None]*low/ny
        else:
            load[:,:-1,0,:] += lower[...,None]/nx
            load[:,1:,0,:] += upper[...,None]*low/nx


def _cell_conductivity(value, nx, ny, device, dtype):
    k = torch.as_tensor(value,device=device,dtype=dtype)
    cross = None
    if k.shape in ((),(nx,ny)):
        xx = yy = k.expand(nx,ny)
    elif k.shape in ((2,),(nx,ny,2)):
        xx,yy = k.expand(nx,ny,2).unbind(-1)
    elif k.shape in ((3,),(nx,ny,3)):
        xx,yy,cross = k.expand(nx,ny,3).unbind(-1)
    else:
        raise ValueError("supply cellwise or uniform conductivity with one, two or three components")
    if not torch.isfinite(k).all() or (xx<=0).any() or (yy<=0).any():
        raise ValueError("conductivity must be finite with positive diagonal entries")
    if cross is not None and (cross.abs()/xx.sqrt() >= yy.sqrt()).any():
        raise ValueError("conductivity tensor must be positive definite")
    return xx,yy,cross


def _legendre_derivative(phi):
    # Coefficient of P_i in dP_j/dxi for j>i with odd j-i is 2*i+1.
    i = torch.arange(phi.shape[1],device=phi.device)
    gap = i[None,:]-i[:,None]
    return phi @ ((2*i[:,None]+1)*((gap>0) & (gap%2==1))).to(phi.dtype)


def _tangent_basis(dphi,endpoint,axis,scale):
    values = (endpoint[None,:,None]*dphi[:,None,:] if axis=="x" else
              dphi[:,:,None]*endpoint[None,None,:])
    return scale*values.flatten(1)


def _interface_conductivity(left,right):
    # Harmonic weighting of k*grad(u) gives equal coefficients for the two gradients.
    low,high = torch.minimum(left,right),torch.maximum(left,right)
    return low/(.5+.5*(low/high))


def assemble_poisson_2d_bc(
    ncells: int | tuple[int, int] = 16,
    degree: int = 1,
    penalty: float = 20.0,
    bc=None,
    forcing=None,
    neumann_flux=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, dirichlet_values=None, robin_films=None, sparse=False, reaction=0., conductivity=1., complex_load=False, contact_resistance=None, trace_power=None,
):
    """Assemble ``-div(k grad u) + reaction*u = f`` with per-side Dirichlet/Neumann data.

    dirichlet_values maps prescribed sides to scalar values or callables of
    (x, y). Unspecified Dirichlet traces are zero. neumann_flux supplies inward heat flux n·K grad(u).
    conductivity is a positive scalar or (nx,ny) field of element constants.
    A pair (kx,ky) or an (nx,ny,2) field specifies a diagonal conductivity
    tensor in grid coordinates. Use (Kxx,Kyy,Kxy) or (nx,ny,3) for a symmetric
    positive-definite tensor with a cross term.
    contact_resistance maps x or y to nonnegative face resistances (m² K/W),
    scalar or arrays shaped (nx-1,ny) or (nx,ny-1). Positive faces enforce
    q=(T_left-T_right)/R; zero uses the ordinary SIPG interface.
    trace_power maps x or y to a pair of heat inputs in W/m², applied to
    the lower- and higher-coordinate elements respectively. Each entry is
    constant on a face and has the same shape options as contact_resistance.
    robin_films maps each Robin side to (h, ambient), implementing outward
    Fourier flux h*(u-ambient); both entries accept scalars or callables.
    reaction is a finite nonnegative scalar or callable of (x,y). Positive
    reaction on part of the domain fixes the mean even with all-Neumann data.
    Returns ``(matrix, rhs, grid, metadata)`` where ``grid`` holds the element
    centres and sizes needed by :func:`l2_error_2d`. With sparse=True the
    CPU operator is a SciPy CSR matrix. complex_load=True assembles complex
    volume and boundary loads with real material and geometry data. The default returns a torch tensor.
    """
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else tuple(ncells)
    if nx < 2 or ny < 2 or degree not in (1, 2, 3):
        raise ValueError("ncells must be >= 2 and SIPG degree must be in [1, 3]")
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    if forcing is None or not callable(forcing):
        raise TypeError("forcing must be a callable f(x, y) -> tensor")

    types = _resolve_bc(bc)
    fluxes = neumann_flux or {}
    traces = dirichlet_values or {}
    for side in (*fluxes, *traces):
        if side not in BC_SIDES:
            raise ValueError(f"unknown boundary side {side!r}; use one of {BC_SIDES}")
    for side in traces:
        if types[side] != "dirichlet":
            raise ValueError(f"Dirichlet trace supplied on {types[side]} side {side}")

    for side in robin_films or {}:
        if types.get(side) != "robin":
            raise ValueError(f"film data require a Robin side: {side}")
    anchored = "dirichlet" in types.values()
    dev = torch.device(device)
    hx, hy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * hx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * hy
    (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _operators(degree, dev, dtype)
    modes = degree + 1
    local_dofs = modes * modes
    ndofs = nx * ny * local_dofs
    if sparse:
        from scipy.sparse import coo_matrix

        if dev.type != "cpu" or dtype not in (torch.float32, torch.float64):
            raise ValueError("sparse assembly requires CPU float32 or float64")
        rows, columns, values = [], [], []

        def add(indices, block):
            idx = indices.numpy()
            data = block.numpy()
            i, j = np.nonzero(data)
            rows.append(idx[i])
            columns.append(idx[j])
            values.append(data[i, j])
    else:
        matrix = torch.zeros((ndofs, ndofs), device=dev, dtype=dtype)

        def add(indices, block):
            _add_block(matrix, indices, block)

    local = torch.arange(local_dofs, device=dev)

    def dofs(ix: int, iy: int):
        return (ix * ny + iy) * local_dofs + local

    # ---- volume term -----------------------------------------------------
    volume_x = (hy / hx) * torch.einsum("ki,lj->klij", stiffness, mass).reshape(
        local_dofs, local_dofs,
    )
    volume_y = (hx / hy) * torch.einsum("ki,lj->klij", mass, stiffness).reshape(
        local_dofs, local_dofs,
    )
    k_x,k_y,cross = _cell_conductivity(conductivity,nx,ny,dev,dtype)
    if cross is not None:
        dphi = _legendre_derivative(phi)
        mixed = (dphi.T*weights) @ phi
        xy = torch.kron(mixed,mixed.T.contiguous())
        volume_xy = xy+xy.T
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    coefficient = reaction(xq, yq) if callable(reaction) else reaction
    coefficient = torch.as_tensor(coefficient, device=dev, dtype=dtype)
    if not torch.isfinite(coefficient).all() or (coefficient < 0).any():
        raise ValueError("reaction must be finite and nonnegative")
    if coefficient.ndim == 0:
        if coefficient > 0:
            anchored = True
        reaction_block = (hx*hy/4)*coefficient*torch.kron(mass,mass)
    else:
        coefficient = torch.broadcast_to(coefficient,(nx,ny,len(nodes),len(nodes)))
        anchored = anchored or bool((coefficient > 0).any())
        basis = (phi[:,None,:,None]*phi[None,:,None,:]).reshape(-1,local_dofs)
        product_weights = weights[:,None]*weights
    for ix in range(nx):
        if coefficient.ndim:
            weighted = (coefficient[ix]*product_weights).flatten(1)
            blocks = (hx*hy/4)*(basis.T @ (weighted[...,None]*basis))
        for iy in range(ny):
            block = k_x[ix,iy]*volume_x + k_y[ix,iy]*volume_y + (blocks[iy] if coefficient.ndim else reaction_block)
            if cross is not None:
                block = block+cross[ix,iy]*volume_xy
            add(dofs(ix, iy), block)

    # ---- interior faces (identical to the Dirichlet-only benchmark) ------
    measure_x, measure_y = 0.5 * hy, 0.5 * hx
    contacts = _contact_faces(contact_resistance,nx,ny,dev,dtype)
    for axis,normal,h,measure in (("x",k_x,hx,measure_x),("y",k_y,hy,measure_y)):
        face = interior_face_operator(phi,weights,phi_left,phi_right,dphi_left,dphi_right,
                                      axis,h,measure,penalty,modes)
        sx,sy = int(axis=="x"),int(axis=="y")
        if cross is not None or axis in contacts:
            tr,_ = _face_basis(phi,phi_right,dphi_right,axis,2/h)
            tl,_ = _face_basis(phi,phi_left,dphi_left,axis,2/h)
            jump = torch.cat((tr,-tl),dim=1)
        if axis in contacts:
            contact_block = measure*(jump.T*weights) @ jump
        if cross is not None:
            tangent_r = _tangent_basis(dphi,phi_right,axis,1/measure)
            tangent_l = _tangent_basis(dphi,phi_left,axis,1/measure)
            zero = torch.zeros_like(tangent_l)
            left_block = -measure/2*(jump.T*weights) @ torch.cat((tangent_r,zero),dim=1)
            right_block = -measure/2*(jump.T*weights) @ torch.cat((zero,tangent_l),dim=1)
            left_block = left_block+left_block.T
            right_block = right_block+right_block.T
        a,b = normal[:nx-sx,:ny-sy],normal[sx:,sy:]
        harmonic = _interface_conductivity(a,b)
        for ix in range(nx-sx):
            for iy in range(ny-sy):
                jx,jy = ix+sx,iy+sy
                if axis in contacts and contacts[axis][ix,iy]>0:
                    add(torch.cat((dofs(ix,iy),dofs(jx,jy))),contact_block/contacts[axis][ix,iy])
                    continue
                block = face
                if cross is not None:
                    block = block+(cross[ix,iy]/a[ix,iy])*left_block+(cross[jx,jy]/b[ix,iy])*right_block
                add(torch.cat((dofs(ix,iy),dofs(jx,jy))),harmonic[ix,iy]*block)

    # ---- boundary faces --------------------------------------------------
    load_dtype = torch.complex(torch.empty((),dtype=dtype),torch.empty((),dtype=dtype)).dtype if complex_load else dtype
    load = torch.zeros(ndofs, device=dev, dtype=load_dtype).view(nx, ny, local_dofs)
    if trace_power:
        _add_trace_power(load.view(nx,ny,modes,modes),trace_power)
    sides = (
        ("left", "x", 0.0, 0, phi_left, dphi_left, -1.0, hx, measure_x),
        ("right", "x", 1.0, nx - 1, phi_right, dphi_right, 1.0, hx, measure_x),
        ("bottom", "y", 0.0, 0, phi_left, dphi_left, -1.0, hy, measure_y),
        ("top", "y", 1.0, ny - 1, phi_right, dphi_right, 1.0, hy, measure_y),
    )
    for side, axis, position, index, endpoint, d_endpoint, outward, h_normal, measure in sides:
        centers = y if axis == "x" else x
        normal_k = k_x[index,:] if axis=="x" else k_y[:,index]
        if types[side] == "dirichlet":
            block, trace = dirichlet_face_operator(
                phi, weights, endpoint, d_endpoint, axis, outward,
                h_normal, measure, penalty, modes,
            )
            if cross is not None:
                side_cross = cross[index,:] if axis=="x" else cross[:,index]
                tangent = outward*_tangent_basis(dphi,endpoint,axis,1/measure)
                coupling = -measure*(trace.T*weights) @ tangent
                coupling = coupling+coupling.T
            for line in range(ny if axis == "x" else nx):
                ix, iy = (index, line) if axis == "x" else (line, index)
                boundary = normal_k[line]*block
                if cross is not None:
                    boundary = boundary+side_cross[line]*coupling
                add(dofs(ix, iy), boundary)
            data = traces.get(side)
            if data is None:
                continue
            _, derivative = _face_basis(phi, endpoint, d_endpoint, axis, 2/h_normal, outward)
            test = penalty*modes**2/h_normal*trace - derivative
        elif types[side] == "robin":
            h, ambient = _film_samples(robin_films, side, centers, nodes, load_dtype)
            anchored = anchored or bool((h > 0).any())
            test, _ = _face_basis(phi, endpoint, d_endpoint, axis, 2/h_normal, outward)
            blocks = measure*torch.einsum("iq,q,qk,ql->ikl", h, weights, test, test)
            for line, block in enumerate(blocks):
                ix, iy = (index, line) if axis == "x" else (line, index)
                add(dofs(ix, iy), block)
            data = h*ambient
        else:
            data = fluxes.get(side)
            if data is None:
                continue
            test, _ = _face_basis(phi, endpoint, d_endpoint, axis, 2/h_normal, outward)
        samples = _boundary_samples(data, side, centers, nodes, load_dtype)
        contribution = measure*((samples*weights) @ test.to(load_dtype))
        if types[side] == "dirichlet":
            contribution *= normal_k[:,None]
            if cross is not None:
                contribution -= measure*side_cross[:,None]*((samples*weights) @ tangent.to(load_dtype))
        if axis == "x":
            load[index] += contribution
        else:
            load[:,index] += contribution

    # ---- forcing ---------------------------------------------------------
    f = torch.broadcast_to(torch.as_tensor(forcing(xq,yq),device=dev,dtype=load_dtype),
                           (nx,ny,len(nodes),len(nodes)))
    load += (0.25 * hx * hy * torch.einsum(
        "xyqr,q,r,qk,rl->xykl", f, weights.to(load_dtype), weights.to(load_dtype), phi.to(load_dtype), phi.to(load_dtype),
    )).reshape(nx, ny, local_dofs)

    if sparse:
        matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                            shape=(ndofs, ndofs)).tocsr()
        matrix.eliminate_zeros()
        symmetry_error = float(np.max(np.abs((matrix - matrix.T).data), initial=0))
    else:
        symmetry_error = float((matrix - matrix.T).abs().max())
    grid = {"x": x, "y": y, "hx": hx, "hy": hy, "degree": degree}
    metadata = {
        "ncells": (nx, ny), "degree": degree, "dofs": ndofs, "penalty": penalty,
        "bc": dict(types), "temperature_anchored": anchored,
        "matrix_symmetry_error": symmetry_error,
        "device": str(dev), "backend": "scipy-csr" if sparse else "torch-dense",
    }
    return matrix, load.reshape(-1), grid, metadata


def solve_poisson_2d_bc(
    ncells: int | tuple[int, int] = 16,
    degree: int = 1,
    penalty: float = 20.0,
    bc=None,
    forcing=None,
    neumann_flux=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, dirichlet_values=None, robin_films=None, sparse=False, reaction=0., conductivity=1., contact_resistance=None, trace_power=None,
):
    """Solve the Poisson problem with dense torch or CPU SciPy sparse direct algebra.

    Returns ``(x_centers, y_centers, modal_coefficients, metadata)`` with the
    modal coefficients shaped ``(nx, ny, degree + 1, degree + 1)``.
    """
    matrix, rhs, grid, metadata = assemble_poisson_2d_bc(
        ncells=ncells, degree=degree, penalty=penalty, bc=bc,
        forcing=forcing, neumann_flux=neumann_flux, device=device, dtype=dtype,
        dirichlet_values=dirichlet_values, robin_films=robin_films, sparse=sparse, reaction=reaction, conductivity=conductivity, contact_resistance=contact_resistance, trace_power=trace_power,
    )
    nx, ny = metadata["ncells"]
    local_dofs = (degree + 1) ** 2
    if not metadata["temperature_anchored"]:
        ones = torch.zeros_like(rhs)
        ones.view(nx, ny, local_dofs)[:, :, 0] = 1.0
        if sparse:
            scale = float(np.max(np.abs(matrix.data)))
            nullspace = float(np.max(np.abs(matrix @ ones.numpy())))
        else:
            scale = float(matrix.abs().max())
            nullspace = float((matrix @ ones).abs().max())
        metadata["nullspace_residual"] = nullspace / max(scale, 1e-300)
        if metadata["nullspace_residual"] > max(1e-10, 8*torch.finfo(rhs.dtype).eps):
            raise RuntimeError(
                "pure-Neumann operator does not annihilate constants "
                f"(relative residual {metadata['nullspace_residual']:.3e}); "
                "the assembly is inconsistent"
            )
        means = rhs.view(nx, ny, local_dofs)[:, :, 0]
        imbalance = float(means.sum().item())
        metadata["net_neumann_load"] = imbalance
        # Scale by the full assembled load to include moments of nonconstant
        # data when the cell-mean components nearly cancel.
        tolerance = 128*torch.finfo(rhs.dtype).eps*float(rhs.abs().sum().item())
        if abs(imbalance) > tolerance:
            raise ValueError("incompatible pure Neumann data: integrated source plus "
                             f"outward normal derivative is {imbalance:.6g} (tolerance {tolerance:.3g})")
        if sparse:
            metadata["gauge"] = "pin first mean mode and shift to zero spatial mean"
        else:
            matrix = matrix + (1.0 / matrix.shape[0]) * torch.outer(ones, ones)
            metadata["gauge"] = "rank-one 1 1^T / ndofs"
    else:
        metadata["nullspace_residual"] = 0.0
        metadata["gauge"] = "none"

    if sparse:
        from scipy.sparse.linalg import spsolve

        b = rhs.numpy()
        if metadata["temperature_anchored"]:
            u = spsolve(matrix, b)
        else:
            u = np.zeros_like(b)
            u[1:] = spsolve(matrix[1:, 1:], b[1:])
            u[::local_dofs] -= u[::local_dofs].mean()
        solution = torch.from_numpy(u)
        residual = float(np.max(np.abs(matrix @ u - b)))
    else:
        solution = torch.linalg.solve(matrix, rhs)
        residual = float((matrix @ solution - rhs).abs().max())
    metadata["linear_residual"] = residual / max(float(torch.max(torch.abs(rhs)).item()), 1e-300)
    coeff = solution.reshape(nx, ny, degree + 1, degree + 1)
    return grid["x"], grid["y"], coeff, metadata


def poisson_interface_traces_2d(coeff, axis, face, *, conductivity=1., points=(0.,)):
    """Sample the two material traces on a grid-aligned interior cut of the unit square.

    Return temperature (K), coordinate-normal derivative (K/m) and conductive
    flux density (W/m²), all shaped (2, tangential cells, sample points).
    Side 0 belongs to the lower-coordinate material and side 1 to the higher.
    Both fluxes are positive along increasing axis. The material gradients
    give one-sided physical fluxes; poisson_face_powers_2d integrates the SIPG
    numerical flux. Face indices start at 1 for the first interior cut. Sample
    points are local tangential Legendre coordinates in [-1, 1] per face.
    """
    if coeff.ndim != 4 or coeff.shape[2] != coeff.shape[3] or coeff.shape[2] not in (2, 3, 4):
        raise ValueError("expected square P1 to P3 modal blocks")
    if axis not in ("x", "y") or not (coeff.is_floating_point() or coeff.is_complex()):
        raise ValueError("expected axis x or y and floating or complex coefficients")
    nx, ny, modes, _ = coeff.shape
    cells = coeff if axis == "x" else coeff.transpose(0, 1)
    count, tangent_count = cells.shape[:2]
    if not isinstance(face, (int, np.integer)) or not 0 < face < count:
        raise ValueError("face must index an interior cut")
    raw = torch.as_tensor(points, device=coeff.device)
    if raw.is_complex():
        raise ValueError("sample points must be real")
    z = torch.as_tensor(points, device=coeff.device, dtype=coeff.real.dtype)
    if z.ndim != 1 or not z.numel() or not ((z >= -1) & (z <= 1)).all():
        raise ValueError("provide a nonempty vector of sample points in [-1, 1]")
    values = [torch.ones_like(z), z]
    for j in range(2, modes):
        values.append(((2*j-1)*z*values[-1] - (j-1)*values[-2])/j)
    phi = torch.stack(values, dim=1)
    dphi = _legendre_derivative(phi)
    _, _, _, _, _, pl, pr, dl, dr = _operators(modes-1, coeff.device, coeff.real.dtype)
    xx, yy, cross = _cell_conductivity(conductivity, nx, ny, coeff.device, coeff.real.dtype)
    kn = xx if axis == "x" else yy.T
    if cross is not None and axis == "y":
        cross = cross.T
    temperatures, gradients, fluxes = [], [], []
    for index, endpoint, derivative in ((face-1, pr, dr), (face, pl, dl)):
        block = cells[index].flatten(1)
        trace, normal = _face_basis(phi, endpoint, derivative, axis, 2*count)
        gradient = block @ normal.T.to(coeff.dtype)
        flux = -kn[index, :, None]*gradient
        if cross is not None:
            tangent = _tangent_basis(dphi, endpoint, axis, 2*tangent_count)
            flux -= cross[index, :, None]*(block @ tangent.T.to(coeff.dtype))
        temperatures.append(block @ trace.T.to(coeff.dtype))
        gradients.append(gradient)
        fluxes.append(flux)
    return torch.stack(temperatures), torch.stack(gradients), torch.stack(fluxes)


def poisson_face_powers_2d(coeff, *, penalty=20., bc=None, dirichlet_values=None, neumann_flux=None, robin_films=None, conductivity=1.,
                           wall_areal_heat_capacity=None, temperature_rate=None, contact_resistance=None):
    """Integrate SIPG numerical heat flux over each face for unit depth.

    Returns arrays (nx+1, ny) and (nx, ny+1) oriented in increasing x and y.
    Supply the same conductivity, penalty and boundary data as the solve.
    With wall_areal_heat_capacity, supply modal temperature_rate (iωU for harmonic
    fields, U0 for first-order coefficients). Returned wall power then includes
    storage and measures heat crossing into the material. In steady state,
    each cell’s net outward power equals its integrated source minus reaction loss.
    """
    nx, ny, modes, last = coeff.shape
    if modes != last or modes not in (2, 3, 4) or min(nx, ny) < 2 or penalty <= 0:
        raise ValueError("expected square P1 to P3 modal blocks and positive penalty")
    capacities = wall_areal_heat_capacity or {}
    types = _resolve_bc(bc)
    _check_storage_sides(capacities,types)
    if capacities:
        if temperature_rate is None or temperature_rate.shape!=coeff.shape:
            raise ValueError("wall capacity requires matching temperature_rate coefficients")
        coeff = coeff.to(torch.promote_types(coeff.dtype,temperature_rate.dtype))
    nodes, weights, phi, _, _, pl, pr, dl, dr = _operators(modes-1, coeff.device, coeff.real.dtype)
    k_x,k_y,cross = _cell_conductivity(conductivity,nx,ny,coeff.device,coeff.real.dtype)
    contacts = _contact_faces(contact_resistance,nx,ny,coeff.device,coeff.real.dtype)
    fields = coeff.flatten(2)
    outputs = []
    for axis, count, tangential, low, high in (("x", nx, ny, "left", "right"),
                                               ("y", ny, nx, "bottom", "top")):
        u = fields if axis == "x" else fields.transpose(0, 1)
        tl, gl = _face_basis(phi, pl, dl, axis, 2*count)
        tr, gr = _face_basis(phi, pr, dr, axis, 2*count)
        tl,gl,tr,gr = (v.to(coeff.dtype) for v in (tl,gl,tr,gr))
        scale = penalty*modes*modes*count
        interior = -.5*(u[:-1] @ gr.T + u[1:] @ gl.T) + scale*(u[:-1] @ tr.T-u[1:] @ tl.T)
        material = k_x if axis=="x" else k_y.T
        if cross is not None:
            dphi = _legendre_derivative(phi)
            tangent_l = _tangent_basis(dphi,pl,axis,2*tangential)
            tangent_r = _tangent_basis(dphi,pr,axis,2*tangential)
            tangent_l,tangent_r = tangent_l.to(coeff.dtype),tangent_r.to(coeff.dtype)
            offdiag = cross if axis=="x" else cross.T
            ratio = (offdiag/material)[...,None]
            interior -= .5*(ratio[:-1]*(u[:-1] @ tangent_r.T)+ratio[1:]*(u[1:] @ tangent_l.T))
        interior *= _interface_conductivity(material[:-1],material[1:])[...,None]
        if axis in contacts:
            resistance = contacts[axis] if axis=="x" else contacts[axis].T
            mask = resistance>0
            jump = u[:-1] @ tr.T-u[1:] @ tl.T
            interior[mask] = jump[mask]/resistance[mask][:,None]
        result = coeff.new_empty((count+1, tangential))
        result[1:-1] = (interior @ weights.to(coeff.dtype))/(2*tangential)
        centers = (torch.arange(tangential, device=coeff.device, dtype=coeff.real.dtype)+.5)/tangential
        for side, index, sign, trace, gradient in ((low, 0, -1, tl, gl), (high, -1, 1, tr, gr)):
            if types[side] == "robin":
                h, ambient = _film_samples(robin_films, side, centers, nodes, coeff.dtype)
                flux = sign*h*(u[index] @ trace.T-ambient)
                if side in capacities:
                    surface = _surface_capacity_samples(capacities[side],side,centers,nodes)
                    boundary_rate = temperature_rate[index,:] if axis=="x" else temperature_rate[:,index]
                    flux += sign*surface*(boundary_rate.flatten(1).to(coeff.dtype) @ trace.T)
                result[index] = (flux @ weights.to(coeff.dtype))/(2*tangential)
                continue
            neumann = types[side] == "neumann"
            data = ((neumann_flux if neumann else dirichlet_values) or {}).get(side)
            prescribed = torch.zeros_like(result[index, :, None]) if data is None else _boundary_samples(
                data, side, centers, nodes, coeff.dtype)
            if neumann:
                flux = -sign*prescribed.expand(tangential, len(nodes))
            else:
                flux = material[index,:,None]*(-(u[index] @ gradient.T) + sign*scale*(u[index] @ trace.T-prescribed))
                if cross is not None:
                    tangent = tangent_l if index==0 else tangent_r
                    flux -= offdiag[index,:,None]*(u[index] @ tangent.T)
            result[index] = (flux @ weights.to(coeff.dtype))/(2*tangential)
        outputs.append(result if axis == "x" else result.T)
    return tuple(outputs)


def l2_error_2d(x, y, coeff, exact, remove_constant: bool = False) -> dict[str, float | complex]:
    """Relative L2 error of a modal field against a callable ``exact(x, y)``.

    ``remove_constant`` subtracts the mean of the pointwise difference before
    taking the norm; that is the correct comparison for a pure-Neumann
    solution, which the discrete problem only fixes up to a constant.
    """
    degree = coeff.shape[-1] - 1
    nx, ny = coeff.shape[:2]
    hx, hy = 1.0 / nx, 1.0 / ny
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 4, 8))
    nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.real.dtype)
    weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.real.dtype)
    basis = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=coeff.device, dtype=coeff.dtype,
        ))
    phi = torch.stack(basis, dim=1)
    numerical = torch.einsum("xykl,qk,rl->xyqr", coeff, phi, phi)
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    reference = exact(xq, yq)
    difference = numerical - reference
    quadrature = weights[None, None, :, None] * weights[None, None, None, :]
    weight = 0.25 * hx * hy
    offset = 0.0
    reference = reference.expand_as(difference)
    if remove_constant:
        total = nx * ny * float(torch.sum(weight * quadrature).item())
        offset = torch.sum(weight * quadrature * difference).item() / total
        difference = difference - offset
    error2 = weight * torch.sum(quadrature * difference.abs() ** 2)
    if remove_constant:
        reference = reference - (weight * torch.sum(quadrature * reference))/total
    norm2 = weight * torch.sum(quadrature * reference.abs()**2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {
        "absolute_l2": absolute,
        "exact_l2": norm,
        "relative_l2": absolute / norm,
        "constant_offset": offset,
        "max_abs_error": float(torch.max(torch.abs(difference)).item()),
    }