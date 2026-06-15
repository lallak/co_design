import sys
from pathlib import Path

# Add hydrax to path so internal imports work
hydrax_path = Path(__file__).parent / "hydrax"
sys.path.insert(0, str(hydrax_path))

import argparse
import numpy as np
from cartpole_codesign.optimize import run_codesign
from cartpole_codesign.visualize import plot_convergence, render_best_design
from cartpole_codesign.evaluator import evaluate_design, generate_heatmap_2d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole Co-Design Experiment")
    parser.add_argument('mode', choices=['optimize', 'visualize', 'render', 'test'],
                        help="Mode to run")
    parser.add_argument('--theta', nargs=2, type=float,
                        default=[0.8, 1.0],
                        help="Design parameters [pole_length, cart_mass] for render/test mode")
    args = parser.parse_args()

    if args.mode == 'optimize':
        theta_opt, history = run_codesign()
        plot_convergence()
        render_best_design(theta_opt)

    elif args.mode == 'visualize':
        plot_convergence()

    elif args.mode == 'render':
        theta = np.array(args.theta)
        render_best_design(theta)

    elif args.mode == 'test':
        # Quick sanity check: evaluate the default design
        theta = np.array(args.theta)
        reward = evaluate_design(theta, seed=0)
        generate_heatmap_2d(resolution=args.resolution)
