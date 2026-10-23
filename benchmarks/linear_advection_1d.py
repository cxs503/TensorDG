"""Run the TensorDG smooth periodic advection convergence benchmark."""
import argparse
import math
from tensordg import solve_advection, l2_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument('--cells', type=int, nargs='+', default=[20, 40, 80, 160])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cfl', type=float, default=0.08)
    args = parser.parse_args()
    print(f"1D periodic linear advection | DG degree={args.degree} | PyTorch")
    print("cells       L2 error       EOC       steps")
    previous_error, previous_h = None, None
    for n in args.cells:
        centers, coeff, meta = solve_advection(ncells=n, degree=args.degree,
            final_time=1.0, speed=1.0, cfl=args.cfl, device=args.device)
        error = l2_error(centers, coeff, final_time=1.0, speed=1.0)
        h = 1.0 / n
        eoc = math.log(previous_error / error) / math.log(previous_h / h) if previous_error else float('nan')
        print(f"{n:5d}  {error:12.5e}  {eoc:7.3f}  {meta['steps']:7d}")
        previous_error, previous_h = error, h

if __name__ == '__main__':
    main()
